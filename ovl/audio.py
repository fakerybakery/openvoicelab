"""Audio I/O helpers.

Only depends on numpy + soundfile (+ soxr for resampling) so it can be imported
both by the UI process and by model workers running in their own environments.
Formats libsndfile can't read (m4a, aac, ...) are decoded with the ffmpeg CLI.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf


def _decode_with_ffmpeg(path: str, sr: Optional[int]) -> Tuple[np.ndarray, int]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(f"Cannot decode {Path(path).name}: install ffmpeg or convert it to WAV/FLAC/MP3 first")
    target_sr = sr or 44100
    cmd = [ffmpeg, "-nostdin", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(target_sr), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy(), target_sr


def resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav
    import soxr

    return soxr.resample(wav, orig_sr, target_sr).astype(np.float32)


def load_audio(path: str, sr: Optional[int] = None, mono: bool = True) -> Tuple[np.ndarray, int]:
    """Load audio as float32. Returns (wav, sample_rate); wav is 1-D if mono else (samples, channels)."""
    path = str(path)
    try:
        wav, file_sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        wav, file_sr = _decode_with_ffmpeg(path, sr)
        return wav, file_sr

    wav = wav.mean(axis=1) if mono else wav
    if sr is not None and sr != file_sr:
        if mono:
            wav = resample(wav, file_sr, sr)
        else:
            wav = np.stack([resample(wav[:, c], file_sr, sr) for c in range(wav.shape[1])], axis=1)
        file_sr = sr
    return wav, file_sr


def save_audio(path: str, wav, sr: int) -> str:
    """Save a 1-D (or samples x channels) array/tensor as 16-bit WAV."""
    if hasattr(wav, "detach"):  # torch tensor
        wav = wav.detach().float().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32)
    wav = np.squeeze(wav)
    if wav.ndim == 2 and wav.shape[0] < wav.shape[1]:  # (channels, samples) -> (samples, channels)
        wav = wav.T
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(wav, -1.0, 1.0), sr, subtype="PCM_16")
    return path


def duration(path: str) -> float:
    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        wav, sr = load_audio(path)
        return len(wav) / sr
