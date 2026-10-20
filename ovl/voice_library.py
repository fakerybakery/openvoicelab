"""The user's voice library: reference clips for cloning, with optional transcripts.

Each voice is an audio file in paths.voices_dir(); its transcript, if known, sits
next to it as <name>.txt (several models clone better, or only, with one).
"""

import re
import shutil
from pathlib import Path
from typing import List, Optional

from ovl import paths

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus")


def _clean_name(name: str) -> str:
    name = re.sub(r"[^\w\- ]+", "", name).strip()
    if not name:
        raise ValueError("Voice name must contain letters or numbers")
    return name


def list_voices() -> List[str]:
    return sorted(
        (p.stem for p in paths.voices_dir().iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS),
        key=str.lower,
    )


def audio_path(name: str) -> Path:
    for ext in AUDIO_EXTENSIONS:
        p = paths.voices_dir() / f"{name}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Voice '{name}' not found")


def transcript(name: str) -> str:
    p = paths.voices_dir() / f"{name}.txt"
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def set_transcript(name: str, text: Optional[str]):
    p = paths.voices_dir() / f"{name}.txt"
    if text and text.strip():
        p.write_text(text.strip() + "\n", encoding="utf-8")
    else:
        p.unlink(missing_ok=True)


def add_voice(name: str, source: str, text: Optional[str] = None, overwrite: bool = False) -> str:
    name = _clean_name(name)
    if name in list_voices():
        if not overwrite:
            raise ValueError(f"A voice named '{name}' already exists")
        delete_voice(name)
    ext = Path(source).suffix.lower()
    if ext not in AUDIO_EXTENSIONS:
        ext = ".wav"
    shutil.copy(source, paths.voices_dir() / f"{name}{ext}")
    set_transcript(name, text)
    return name


def delete_voice(name: str):
    audio_path(name).unlink()
    (paths.voices_dir() / f"{name}.txt").unlink(missing_ok=True)
