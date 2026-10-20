"""Whisper transcription for the voice library and the dataset builder."""

import logging
import threading
from typing import Optional

from ovl.audio import load_audio

logger = logging.getLogger(__name__)

WHISPER_MODELS = [
    "openai/whisper-large-v3-turbo",
    "openai/whisper-large-v3",
    "distil-whisper/distil-large-v3.5",
    "openai/whisper-medium",
    "openai/whisper-small",
    "openai/whisper-base",
    "openai/whisper-tiny",
]
DEFAULT_MODEL = WHISPER_MODELS[0]

_pipe = None
_pipe_id = None
_lock = threading.Lock()


def device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_pipeline(model_id: str = DEFAULT_MODEL):
    global _pipe, _pipe_id
    with _lock:
        if _pipe is None or _pipe_id != model_id:
            import torch
            from transformers import pipeline

            dev = device()
            logger.info("Loading %s on %s", model_id, dev)
            _pipe = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                dtype=torch.float16 if dev != "cpu" else torch.float32,
                device=dev,
            )
            _pipe_id = model_id
        return _pipe


def unload():
    global _pipe, _pipe_id
    with _lock:
        _pipe = _pipe_id = None
    import gc

    gc.collect()


def transcribe(path: str, model_id: str = DEFAULT_MODEL, language: Optional[str] = None) -> str:
    wav, sr = load_audio(path, sr=16000)
    kwargs = {"generate_kwargs": {"language": language}} if language else {}
    result = get_pipeline(model_id)(
        {"raw": wav, "sampling_rate": sr}, chunk_length_s=30, batch_size=8, **kwargs
    )
    return result["text"].strip()
