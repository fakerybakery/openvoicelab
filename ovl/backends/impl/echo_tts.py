import hashlib
import random
import re
import shutil
import sys
import types
import urllib.request
from pathlib import Path

import numpy as np
import torch

from ovl.audio import load_audio, save_audio
from ovl.backends.base import Backend as BaseBackend
from ovl.backends.base import LoadInfo, Request, resolve_device

SAMPLE_RATE = 44100
REPO = "jordandare/echo-tts"
COMMIT = "2ed95fce62d33bf7b56f835fd9ec0f0b6fb9155e"
FILES = {
    "inference.py": "8df068bdb8ce02edefa9ef1ecd1d711bf75b6a90cc3797fcf5567d1c5c5ce80a",
    "model.py": "aaf8ff3f9a24cb7081ee2e7411290aa05df9d01092e6f9d708f0ecf9b1bd7b0f",
    "autoencoder.py": "248c161d8e140c557d0d6e7bd4eb8325bdda99b0c44a734aabb1825353f842c7",
}
MAX_REF_SECONDS = 300
GAP_SECONDS = 0.2
MAX_LATENTS = 640  # ~30 s, the training length; ~21.5 latents per second


def _initial_length(text: str) -> int:
    """Generous latent-length guess for short text (shorter sequences sample much faster)."""
    return min(MAX_LATENTS, max(192, -(-len(text) * 3 // 64) * 64))


def _fetch_source() -> Path:
    """Download the commit-pinned upstream modules into <env>/src (the repo isn't a package)."""
    dest = Path(sys.prefix) / "src" / f"echo-tts-{COMMIT[:12]}"
    if all((dest / f).exists() for f in FILES):
        return dest
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for name, sha in FILES.items():
        url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{name}"
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        if hashlib.sha256(data).hexdigest() != sha:
            raise RuntimeError(f"Checksum mismatch for {url}")
        (tmp / name).write_bytes(data)
    shutil.rmtree(dest, ignore_errors=True)
    tmp.replace(dest)
    return dest


def _import_echo():
    src = str(_fetch_source())
    if src not in sys.path:
        sys.path.insert(0, src)
    # inference.py imports torchcodec only for its own file loader, which we don't use.
    if "torchcodec" not in sys.modules:
        stub = types.ModuleType("torchcodec.decoders")
        stub.AudioDecoder = None
        sys.modules["torchcodec"] = types.ModuleType("torchcodec")
        sys.modules["torchcodec.decoders"] = stub
    import inference

    return inference


def chunk_text(text: str, max_chars: int):
    """Split text at sentence (then clause, then word) boundaries into chunks of <= max_chars."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return [text] if text else []
    pieces = []
    for sentence in re.split(r"(?<=[.!?…])\s+", text):
        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue
        for clause in re.split(r"(?<=[,;:—])\s+", sentence):
            while len(clause) > max_chars:
                cut = clause.rfind(" ", 0, max_chars)
                cut = cut if cut > 0 else max_chars
                pieces.append(clause[:cut])
                clause = clause[cut:].strip()
            if clause:
                pieces.append(clause)
    chunks, current = [], ""
    for p in pieces:
        if current and len(current) + 1 + len(p) > max_chars:
            chunks.append(current)
            current = p
        else:
            current = f"{current} {p}".strip()
    if current:
        chunks.append(current)
    return chunks


class Backend(BaseBackend):
    def load(self, model_id, device, options):
        self.device = resolve_device(device)
        self.progress(None, "Fetching Echo-TTS code...")
        self.echo = _import_echo()
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self.progress(None, "Loading Echo-TTS...")
        self.model = self.echo.load_model_from_hf(
            model_id, device=self.device, dtype=dtype, delete_blockwise_modules=True
        )
        self.progress(None, "Loading Fish S1-DAC codec...")
        self.fish_ae = self.echo.load_fish_ae_from_hf(device=self.device, dtype=torch.float32)
        self.pca_state = self.echo.load_pca_state_from_hf(model_id, device=self.device)
        self._speaker_cache = (None, None)
        return LoadInfo(languages=["English"], sample_rate=SAMPLE_RATE)

    def _speaker(self, path):
        if path is None:
            latent = torch.zeros((1, 4, 80), device=self.device, dtype=self.model.dtype)
            return latent, torch.zeros((1, 4), device=self.device, dtype=torch.bool)
        key = (path, Path(path).stat().st_mtime)
        if self._speaker_cache[0] != key:
            wav, _ = load_audio(path, sr=SAMPLE_RATE)
            audio = torch.from_numpy(wav[: MAX_REF_SECONDS * SAMPLE_RATE]).unsqueeze(0)
            audio = audio / max(audio.abs().max().item(), 1.0)
            self.progress(None, "Encoding reference voice...")
            latent_mask = self.echo.get_speaker_latent_and_mask(
                self.fish_ae, self.pca_state, audio.to(self.device, self.fish_ae.dtype)
            )
            self._speaker_cache = (key, latent_mask)
        return self._speaker_cache[1]

    def generate(self, request: Request):
        p = request.params
        chunks = chunk_text(request.text, int(p.get("chunk_chars", 400)))
        if not chunks:
            raise ValueError("Enter some text to synthesize")
        speaker_latent, speaker_mask = self._speaker(request.voice.audio)

        seed = request.seed if request.seed is not None and request.seed >= 0 else random.randint(0, 2**31 - 1)
        kv_scale = float(p.get("force_speaker", 1.0))
        rescale_k = float(p.get("rescale_k", 1.0))
        sample_kwargs = dict(
            num_steps=int(p.get("num_steps", 40)),
            cfg_scale_text=float(p.get("cfg_scale_text", 3.0)),
            cfg_scale_speaker=float(p.get("cfg_scale_speaker", 8.0)),
            cfg_min_t=0.5,
            cfg_max_t=1.0,
            truncation_factor=float(p.get("truncation", 1.0)),
            rescale_k=rescale_k if rescale_k != 1.0 else None,
            rescale_sigma=3.0,
            speaker_kv_scale=kv_scale if kv_scale != 1.0 else None,
            speaker_kv_max_layers=24,
            speaker_kv_min_t=0.9,
        )

        gap = np.zeros(int(GAP_SECONDS * SAMPLE_RATE), dtype=np.float32)
        pieces = []
        with torch.inference_mode():
            for i, chunk in enumerate(chunks):
                self.progress(i / len(chunks), f"Generating chunk {i + 1}/{len(chunks)}")
                text_ids, text_mask = self.echo.get_text_input_ids_and_mask(
                    [chunk], max_length=768, device=self.device, pad_to_max=False
                )
                length = _initial_length(chunk)
                while True:
                    latent = self.echo.sample_euler_cfg_independent_guidances(
                        self.model, speaker_latent, speaker_mask, text_ids, text_mask, seed,
                        sequence_length=length, **sample_kwargs,
                    )
                    end = self.echo.find_flattening_point(latent[0])
                    # Speech ran to the end of a shortened window: redo it at full length.
                    if end < length - 8 or length == MAX_LATENTS:
                        break
                    length = MAX_LATENTS
                audio = self.echo.ae_decode(self.fish_ae, self.pca_state, latent[:, :end])
                pieces += [audio[0, 0].float().cpu().numpy(), gap]
        if self.device == "mps":
            torch.mps.empty_cache()
        save_audio(request.output_path, np.concatenate(pieces[:-1]), SAMPLE_RATE)
        return {"chunks": len(chunks), "seed": seed}

    def unload(self):
        self.model = self.fish_ae = self.pca_state = None
        self._speaker_cache = (None, None)
