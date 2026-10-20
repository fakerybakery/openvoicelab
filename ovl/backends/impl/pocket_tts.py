import numpy as np
import torch
from pocket_tts import TTSModel
from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

from ovl.audio import load_audio, save_audio
from ovl.backends.base import Backend as BaseBackend
from ovl.backends.base import LoadInfo, Request, seed_everything, split_speaker_script

CLONE_GATED = (
    "Voice cloning for Pocket TTS needs the gated kyutai/pocket-tts weights. Accept the terms at "
    "https://huggingface.co/kyutai/pocket-tts, log in with `hf auth login` (or set HF_TOKEN), then reload "
    "the model. Preset voices work without it."
)
MAX_PROMPT_SECONDS = 30
PAUSE_SECONDS = 0.25


class Backend(BaseBackend):
    def load(self, model_id, device, options):
        # Small autoregressive model with per-frame sync points: CPU is faster than MPS.
        self.progress(None, f"Loading Pocket TTS ({model_id})...")
        self.model = TTSModel.load_model(language=model_id)
        self.sample_rate = self.model.sample_rate
        self.states = {}
        language = model_id.split("_")[0].capitalize()
        notes = "" if self.model.has_voice_cloning else CLONE_GATED
        return LoadInfo(
            presets=sorted(_ORIGINS_OF_PREDEFINED_VOICES),
            languages=[language],
            capabilities={} if self.model.has_voice_cloning else {"clone": False},
            sample_rate=self.sample_rate,
            notes=notes,
        )

    def _state(self, voice):
        if voice.audio:
            if not self.model.has_voice_cloning:
                raise ValueError(CLONE_GATED)
            key = ("audio", voice.audio)
        elif voice.preset:
            if voice.preset not in _ORIGINS_OF_PREDEFINED_VOICES:
                raise ValueError(f"Unknown Pocket TTS voice {voice.preset!r}")
            key = ("preset", voice.preset)
        else:
            raise ValueError("Pick a Pocket TTS voice or upload a reference clip")
        if key not in self.states:
            self.progress(None, "Encoding voice...")
            if voice.audio:
                wav, _ = load_audio(voice.audio, sr=self.sample_rate)
                wav = torch.from_numpy(wav[: MAX_PROMPT_SECONDS * self.sample_rate]).unsqueeze(0)
                self.states[key] = self.model.get_state_for_audio_prompt(wav)
            else:
                self.states[key] = self.model.get_state_for_audio_prompt(voice.preset)
        return self.states[key]

    def generate(self, request: Request):
        if not request.text.strip():
            raise ValueError("Enter some text to synthesize")
        voices = request.voices or []
        if not voices or any(v.is_empty for v in voices):
            raise ValueError("Pick a Pocket TTS voice or upload a reference clip")
        states = [self._state(v) for v in voices]
        lines = split_speaker_script(request.text, len(voices)) if len(voices) > 1 else [(0, request.text)]

        params = request.params
        self.model.temp = float(params.get("temperature", 0.3))
        self.model.sampler_decode_steps = int(params.get("decode_steps", 1))
        self.model.eos_threshold = float(params.get("eos_threshold", -4.0))
        seed_everything(request.seed)

        pause = np.zeros(int(PAUSE_SECONDS * self.sample_rate), dtype=np.float32)
        chunks = []
        for i, (speaker, line) in enumerate(lines):
            self.progress(i / len(lines), f"Generating line {i + 1}/{len(lines)}")
            audio = self.model.generate_audio(states[speaker], line, copy_state=True)
            if chunks:
                chunks.append(pause)
            chunks.append(audio.float().cpu().numpy().reshape(-1))
        if not chunks:
            raise ValueError("Pocket TTS produced no audio for this text")
        save_audio(request.output_path, np.concatenate(chunks), self.sample_rate)
        return {}

    def unload(self):
        self.model = None
        self.states = {}
