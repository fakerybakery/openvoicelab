"""Worker-side backend interface.

Implementations live in ovl/backends/impl/ and run inside the backend's own
environment (see ovl/runtime/worker.py), so they may import their model
libraries at module level. They must not import gradio or any other UI module.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

ProgressFn = Callable[[Optional[float], str], None]


@dataclass
class Voice:
    """A voice to synthesize with. Exactly one of the fields is normally set."""

    preset: Optional[str] = None  # name of a built-in voice
    audio: Optional[str] = None  # path to a reference clip for cloning
    transcript: Optional[str] = None  # transcript of `audio`, if known
    description: Optional[str] = None  # natural-language voice design prompt

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Voice":
        return cls(**(d or {}))

    @property
    def is_empty(self) -> bool:
        return not (self.preset or self.audio or self.description)


@dataclass
class Request:
    text: str
    output_path: str
    # One voice per speaker. Speaker N in the script uses voices[N - 1].
    voices: List[Voice] = field(default_factory=list)
    language: Optional[str] = None
    instruct: Optional[str] = None  # style / emotion instruction
    seed: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Request":
        d = dict(d)
        d["voices"] = [Voice.from_dict(v) for v in d.get("voices") or []]
        return cls(**d)

    @property
    def voice(self) -> Voice:
        return self.voices[0] if self.voices else Voice()


@dataclass
class LoadInfo:
    """Returned by load(); tells the UI what this particular checkpoint offers."""

    presets: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    # Overrides for spec capabilities that depend on the checkpoint (e.g. Qwen3-TTS
    # CustomVoice vs Base vs VoiceDesign). Keys are Capabilities field names.
    capabilities: Dict[str, Any] = field(default_factory=dict)
    sample_rate: int = 24000
    notes: str = ""


class Backend(ABC):
    def __init__(self, progress: ProgressFn):
        self.progress = progress

    @abstractmethod
    def load(self, model_id: str, device: str, options: Dict[str, Any]) -> LoadInfo:
        """Load weights. `options` may contain e.g. {"lora_path": ...}."""

    @abstractmethod
    def generate(self, request: Request) -> Dict[str, Any]:
        """Synthesize request.text, write a WAV to request.output_path.

        Returns a dict of extra metrics to show the user (may be empty)."""

    def unload(self):
        pass


def resolve_device(device: Optional[str]) -> str:
    import torch

    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def fix_espeak_data_path():
    """espeak-ng silently ignores data paths longer than ~160 chars and then exit()s
    the process. When the env lives somewhere deep, use a copy of the data at a short
    path (phonemizer resolves symlinks, so a link isn't enough)."""
    import shutil
    from importlib.metadata import version
    from pathlib import Path

    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    try:  # misaki sets the (long) path at import time; import it first so we win.
        import misaki.espeak  # noqa: F401
    except ImportError:
        pass

    data = Path(espeakng_loader.get_data_path())
    if len(str(data)) > 120:
        short = Path.home() / ".cache" / "openvoicelab" / f"espeak-{version('espeakng-loader')}"
        if not (short / "phontab").exists():
            tmp = short.with_name(short.name + ".tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.copytree(data, tmp)
            tmp.replace(short)
        data = short
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(str(data))


def seed_everything(seed: Optional[int]):
    if seed is None or seed < 0:
        return
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_speaker_script(text: str, num_speakers: int) -> List[tuple]:
    """Parse a "Speaker N: line" script into [(speaker_index, line)], 0-based.

    Speaker labels may be 0- or 1-based; they are renumbered in order of first
    appearance. Unlabelled lines continue with the previous speaker. If the script
    has no labels at all, lines rotate through the speakers.
    """
    import re

    pattern = re.compile(r"^\s*speaker\s*(\d+)\s*[:：]\s*(.*)$", re.IGNORECASE)
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not any(pattern.match(l) for l in lines):
        return [(i % max(num_speakers, 1), l) for i, l in enumerate(lines)]

    mapping: Dict[str, int] = {}
    out = []
    current = 0
    for line in lines:
        m = pattern.match(line)
        if m:
            label, line = m.group(1), m.group(2).strip()
            if label not in mapping:
                mapping[label] = len(mapping)
            current = mapping[label]
            if current >= num_speakers:
                raise ValueError(
                    f"The script uses {len(mapping)} speakers but only {num_speakers} voice(s) are set. "
                    f"Increase the number of speakers."
                )
        if line:
            out.append((current, line))
    return out
