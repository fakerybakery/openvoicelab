import gc

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel

from ovl.audio import load_audio, save_audio
from ovl.backends.base import Backend as BaseBackend
from ovl.backends.base import LoadInfo, Request, resolve_device, seed_everything, split_speaker_script

GEN_KEYS = ("temperature", "top_p", "top_k", "repetition_penalty", "subtalker_temperature", "max_new_tokens")
INT_KEYS = ("top_k", "max_new_tokens")
PAUSE_SECONDS = 0.25


def _pretty(name):
    return "_".join(p.capitalize() for p in name.split("_"))


class Backend(BaseBackend):
    def load(self, model_id, device, options):
        self.device = resolve_device(device)
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self.progress(None, "Loading Qwen3-TTS...")
        self.tts = Qwen3TTSModel.from_pretrained(
            model_id, device_map=self.device, dtype=dtype, attn_implementation="sdpa"
        )
        model = self.tts.model
        # Checkpoint type comes from the config, so local/fine-tuned paths work too.
        self.kind = model.tts_model_type
        self.presets = {p: _pretty(p) for p in sorted(model.get_supported_speakers() or [])}
        languages = ["Auto"] + [l.capitalize() for l in model.get_supported_languages() if l != "auto"]

        if self.kind == "custom_voice":
            caps = dict(presets=True, instruct=model.tts_model_size != "0b6", clone=False, voice_design=False)
        elif self.kind == "voice_design":
            caps = dict(presets=False, instruct=False, clone=False, voice_design=True)
        elif self.kind == "base":
            caps = dict(presets=False, instruct=False, clone=True, clone_needs_transcript=True, voice_design=False)
        else:
            raise ValueError(f"Unknown Qwen3-TTS checkpoint type {self.kind!r}")
        caps["languages"] = languages
        return LoadInfo(
            presets=list(self.presets.values()),
            languages=languages,
            capabilities=caps,
            sample_rate=self.tts.model.speech_tokenizer.get_output_sample_rate(),
            notes=f"{self.kind} checkpoint ({model.tts_model_size})",
        )

    def _gen_kwargs(self, params):
        kw = {k: params[k] for k in GEN_KEYS if params.get(k) is not None}
        for k in INT_KEYS:
            if k in kw:
                kw[k] = int(kw[k])
        return kw

    def _synth(self, text, voice, request, kw, clone_cache):
        language = request.language or "Auto"
        if self.kind == "custom_voice":
            if not voice.preset:
                raise ValueError(f"Pick a Qwen3-TTS voice: {', '.join(self.presets.values())}")
            wavs, sr = self.tts.generate_custom_voice(
                text=text, speaker=voice.preset, language=language, instruct=request.instruct or None, **kw
            )
        elif self.kind == "voice_design":
            design = ". ".join(s.strip() for s in (voice.description, request.instruct) if s and s.strip())
            if not design:
                raise ValueError("Describe the voice you want (e.g. 'A calm middle-aged man with a deep voice')")
            wavs, sr = self.tts.generate_voice_design(text=text, instruct=design, language=language, **kw)
        else:
            if not voice.audio:
                raise ValueError("Upload a reference clip to clone (this is a Qwen3-TTS Base checkpoint)")
            key = (voice.audio, voice.transcript)
            if key not in clone_cache:
                transcript = (voice.transcript or "").strip()
                clone_cache[key] = self.tts.create_voice_clone_prompt(
                    ref_audio=load_audio(voice.audio), ref_text=transcript or None, x_vector_only_mode=not transcript
                )
            wavs, sr = self.tts.generate_voice_clone(
                text=text, language=language, voice_clone_prompt=clone_cache[key], **kw
            )
        return np.asarray(wavs[0], dtype=np.float32), sr

    def generate(self, request: Request):
        seed_everything(request.seed)
        kw = self._gen_kwargs(request.params)
        voices = request.voices or [request.voice]
        lines = split_speaker_script(request.text, len(voices)) if len(voices) > 1 else [(0, request.text.strip())]
        if not any(t for _, t in lines):
            raise ValueError("Enter some text to speak")

        chunks, clone_cache, sr = [], {}, 24000
        for i, (spk, text) in enumerate(lines):
            self.progress(i / len(lines), f"Generating line {i + 1}/{len(lines)}" if len(lines) > 1 else "Generating...")
            wav, sr = self._synth(text, voices[spk], request, kw, clone_cache)
            if chunks:
                chunks.append(np.zeros(int(PAUSE_SECONDS * sr), dtype=np.float32))
            chunks.append(wav)
        audio = np.concatenate(chunks)
        save_audio(request.output_path, audio, sr)

        metrics = {"audio_seconds": round(len(audio) / sr, 2)}
        if self.kind == "base" and any(v.audio and not (v.transcript or "").strip() for v in voices):
            metrics["note"] = "No transcript given: cloned the timbre only (add the transcript for closer cloning)"
        return metrics

    def unload(self):
        self.tts = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
