import os
import re
from pathlib import Path

import numpy as np
import torch
from chatterbox.mtl_tts import SUPPORTED_LANGUAGES, ChatterboxMultilingualTTS
from chatterbox.tts import ChatterboxTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS
from huggingface_hub import snapshot_download

from ovl.audio import load_audio, save_audio
from ovl.backends.base import Backend as BaseBackend
from ovl.backends.base import LoadInfo, Request, resolve_device, seed_everything, split_speaker_script

SAMPLE_RATE = 24000
LANGUAGE_CODES = {name: code for code, name in SUPPORTED_LANGUAGES.items()}
MAX_CHUNK_CHARS = 300
GAP_SECONDS = 0.15

TURBO_FILES = ["ve.safetensors", "s3gen_meanflow.safetensors", "*.json", "*.txt", "conds.pt"]
MTL_FILES = ["ve.pt", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"]
ENGLISH_FILES = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]


def _resolve(model_id):
    """Return (kind, local_dir, extra) for a variant id or a local checkpoint folder."""
    if os.path.isdir(model_id):
        d = Path(model_id)
        if (d / "t3_nano_v1.safetensors").exists():
            return "nano", d, None
        if (d / "t3_turbo_v1.safetensors").exists():
            return "turbo", d, None
        mtl = sorted(d.glob("t3_mtl*.safetensors"))
        if mtl:
            return "multilingual", d, mtl[-1].name
        return "english", d, None
    repo, _, variant = model_id.partition(":")
    if repo.endswith("-nano") or repo.endswith("-turbo"):
        kind = "nano" if repo.endswith("-nano") else "turbo"
        t3 = f"t3_{kind}_v1.safetensors"
        return kind, Path(snapshot_download(repo, allow_patterns=TURBO_FILES + [t3])), None
    if variant.startswith("multilingual"):
        t3 = f"t3_mtl23ls_{variant.rpartition('-')[2] or 'v3'}.safetensors"
        return "multilingual", Path(snapshot_download(repo, allow_patterns=MTL_FILES + [t3])), t3
    return "english", Path(snapshot_download(repo, allow_patterns=ENGLISH_FILES)), None


def _chunks(text, limit=MAX_CHUNK_CHARS):
    """Split long text at sentence boundaries; each generation is capped at ~40 s of speech."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s*", text) if s.strip()]
    out = []
    for s in sentences:
        if out and len(out[-1]) + len(s) + 1 <= limit:
            out[-1] += " " + s
        else:
            out.append(s)
    return out


class Backend(BaseBackend):
    def load(self, model_id, device, options):
        self.device = resolve_device(device)
        self.progress(None, "Downloading Chatterbox weights...")
        self.kind, ckpt_dir, t3_file = _resolve(model_id)
        self.progress(None, f"Loading Chatterbox ({self.kind})...")
        if self.kind in ("turbo", "nano"):
            self.model = ChatterboxTurboTTS.from_local(ckpt_dir, self.device, nano=self.kind == "nano")
            # Upstream's loudness normalisation returns float64, which MPS can't take.
            norm = self.model.norm_loudness
            self.model.norm_loudness = lambda wav, sr, **kw: np.asarray(norm(wav, sr, **kw), dtype=np.float32)
        elif self.kind == "multilingual":
            self.model = ChatterboxMultilingualTTS.from_local(ckpt_dir, self.device, t3_model=t3_file)
        else:
            self.model = ChatterboxTTS.from_local(ckpt_dir, self.device)
        self.default_conds = self.model.conds
        self.voice_conds = {}
        languages = sorted(LANGUAGE_CODES, key=lambda n: (n != "English", n)) if self.kind == "multilingual" else ["English"]
        return LoadInfo(languages=languages, sample_rate=self.model.sr)

    def _conds(self, voice, exaggeration):
        if not voice.audio:
            if self.default_conds is None:
                raise ValueError("This checkpoint has no built-in voice. Add a reference clip.")
            return self.default_conds
        if voice.audio not in self.voice_conds:
            if self.kind in ("turbo", "nano"):
                wav, sr = load_audio(voice.audio)
                if len(wav) / sr <= 5.0:
                    raise ValueError("Chatterbox Turbo/Nano need a reference clip longer than 5 seconds.")
            self.model.prepare_conditionals(voice.audio, exaggeration=exaggeration)
            self.voice_conds[voice.audio] = self.model.conds
        return self.voice_conds[voice.audio]

    def generate(self, request: Request):
        p = request.params
        kwargs = dict(
            temperature=float(p.get("temperature", 0.8)),
            repetition_penalty=float(p.get("repetition_penalty", 1.2)),
        )
        exaggeration = float(p.get("exaggeration", 0.5))
        if self.kind in ("turbo", "nano"):
            kwargs.update(exaggeration=0.0, cfg_weight=0.0)
        else:
            kwargs.update(exaggeration=exaggeration, cfg_weight=float(p.get("cfg_weight", 0.5)))
        if self.kind == "multilingual":
            language = request.language or "English"
            if language not in LANGUAGE_CODES:
                raise ValueError(f"Chatterbox Multilingual doesn't support {language}")
            kwargs["language_id"] = LANGUAGE_CODES[language]

        voices = request.voices or [request.voice]
        lines = split_speaker_script(request.text, len(voices)) if len(voices) > 1 else [(0, request.text)]
        jobs = [(spk, chunk) for spk, line in lines for chunk in _chunks(line)]
        if not jobs:
            raise ValueError("Enter some text to synthesize")

        seed_everything(request.seed)
        gap = np.zeros(int(GAP_SECONDS * self.model.sr), dtype=np.float32)
        pieces = []
        with torch.inference_mode():
            for i, (spk, chunk) in enumerate(jobs):
                self.progress(i / len(jobs), f"Generating chunk {i + 1}/{len(jobs)}")
                self.model.conds = self._conds(voices[spk], exaggeration)
                wav = self.model.generate(chunk, **kwargs)
                pieces += [wav.squeeze(0).float().cpu().numpy(), gap]
        save_audio(request.output_path, np.concatenate(pieces[:-1]), self.model.sr)
        return {"chunks": len(jobs)}

    def unload(self):
        self.model = self.default_conds = None
        self.voice_conds = {}
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
