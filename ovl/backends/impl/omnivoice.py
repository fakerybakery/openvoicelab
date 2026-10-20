import gc

import torch
from omnivoice import OmniVoice
from omnivoice.utils.lang_map import LANG_NAMES

from ovl.audio import save_audio
from ovl.backends.base import Backend as BaseBackend
from ovl.backends.base import LoadInfo, Request, resolve_device, seed_everything

GEN_KEYS = ("num_step", "guidance_scale", "denoise", "postprocess_output")


class Backend(BaseBackend):
    model = None

    def load(self, model_id, device, options):
        self.device = resolve_device(device)
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.progress(None, "Loading OmniVoice...")
        # On MPS the audio codec stays on CPU (handled upstream: its conv output exceeds MPS limits).
        self.model = OmniVoice.from_pretrained(model_id, device_map=self.device, dtype=dtype, attn_implementation="sdpa")
        self.sample_rate = self.model.sampling_rate
        languages = ["Auto"] + sorted(name.title() for name in LANG_NAMES)
        return LoadInfo(languages=languages, sample_rate=self.sample_rate)

    def generate(self, request: Request):
        voice = request.voice
        p = request.params
        kwargs = {k: p[k] for k in GEN_KEYS if k in p}
        if p.get("duration"):
            kwargs["duration"] = float(p["duration"])
        elif p.get("speed", 1.0) != 1.0:
            kwargs["speed"] = float(p["speed"])
        if voice.audio:
            if voice.transcript is None:
                self.progress(None, "Transcribing reference clip...")
            kwargs["voice_clone_prompt"] = self.model.create_voice_clone_prompt(voice.audio, voice.transcript or None)
        elif voice.description or request.instruct:
            kwargs["instruct"] = voice.description or request.instruct

        language = request.language if request.language and request.language.lower() != "auto" else None
        seed_everything(request.seed)
        self.progress(None, "Generating...")
        audio = self.model.generate(text=request.text.strip(), language=language, **kwargs)[0]
        save_audio(request.output_path, audio, self.sample_rate)
        return {}

    def unload(self):
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
