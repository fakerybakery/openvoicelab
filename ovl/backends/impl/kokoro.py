import numpy as np
import torch
from huggingface_hub import list_repo_files
from kokoro import KModel, KPipeline

from ovl.audio import save_audio
from ovl.backends.base import Backend as BaseBackend
from ovl.backends.base import LoadInfo, Request, fix_espeak_data_path, resolve_device

fix_espeak_data_path()

SAMPLE_RATE = 24000

# First letter of a Kokoro voice name -> pipeline lang_code.
LANGUAGES = {
    "English (US)": "a",
    "English (UK)": "b",
    "Spanish": "e",
    "French": "f",
    "Hindi": "h",
    "Italian": "i",
    "Portuguese (BR)": "p",
}


class Backend(BaseBackend):
    def load(self, model_id, device, options):
        self.device = resolve_device(device)
        self.repo_id = model_id
        self.progress(None, "Loading Kokoro...")
        self.model = KModel(repo_id=model_id).to(self.device).eval()
        self.pipelines = {}
        voices = sorted(
            f[len("voices/") : -len(".pt")]
            for f in list_repo_files(model_id)
            if f.startswith("voices/") and f.endswith(".pt")
        )
        # Japanese/Chinese voices need extra misaki dependencies; only list what works.
        voices = [v for v in voices if v[0] in LANGUAGES.values()]
        return LoadInfo(presets=voices, languages=list(LANGUAGES), sample_rate=SAMPLE_RATE)

    def _pipeline(self, lang_code):
        if lang_code not in self.pipelines:
            self.pipelines[lang_code] = KPipeline(lang_code=lang_code, repo_id=self.repo_id, model=self.model)
        return self.pipelines[lang_code]

    def generate(self, request: Request):
        voice = request.voice.preset
        if not voice:
            raise ValueError("Pick a Kokoro voice")
        # The voice prefix decides the phonemizer unless a language is chosen explicitly.
        lang_code = LANGUAGES.get(request.language) or voice[0]
        pipeline = self._pipeline(lang_code)

        chunks = []
        with torch.inference_mode():
            for i, result in enumerate(pipeline(request.text, voice=voice, speed=request.params.get("speed", 1.0))):
                if result.audio is not None:
                    chunks.append(result.audio.cpu().numpy())
                self.progress(None, f"Synthesized {i + 1} chunk(s)")
        if not chunks:
            raise ValueError("Kokoro produced no audio for this text")
        save_audio(request.output_path, np.concatenate(chunks), SAMPLE_RATE)
        return {}
