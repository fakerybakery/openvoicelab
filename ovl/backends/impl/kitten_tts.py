import numpy as np
from kittenml import KittenTTS
from kittenml.kittentts_legacy import onnx_model  # noqa: F401  (sets the long espeak path on import)

from ovl.audio import save_audio
from ovl.backends.base import Backend as BaseBackend
from ovl.backends.base import LoadInfo, Request, fix_espeak_data_path

fix_espeak_data_path()

SAMPLE_RATE = 24000


class Backend(BaseBackend):
    def load(self, model_id, device, options):
        self.progress(None, "Loading Kitten TTS...")
        self.model = KittenTTS(model_id, backend="cpu")
        return LoadInfo(presets=list(self.model.available_voices), languages=["English"], sample_rate=SAMPLE_RATE)

    def generate(self, request: Request):
        voice = request.voice.preset
        if not voice:
            raise ValueError("Pick a Kitten TTS voice")
        if not request.text.strip():
            raise ValueError("Enter some text to synthesize")
        params = request.params
        chunks = []
        stream = self.model.model.generate_stream(
            request.text,
            voice=voice,
            speed=float(params.get("speed", 1.0)),
            clean_text=bool(params.get("clean_text", True)),
        )
        for i, audio in enumerate(stream):
            chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
            self.progress(None, f"Synthesized {i + 1} chunk(s)")
        if not chunks:
            raise ValueError("Kitten TTS produced no audio for this text")
        save_audio(request.output_path, np.concatenate(chunks), SAMPLE_RATE)
        return {}

    def unload(self):
        self.model = None
