"""Declarative descriptions of TTS backends.

A backend spec is plain data: it tells the UI what a model family can do and what
to install, without importing any of the model's code. The implementation lives
in a separate module that is only imported inside the backend's own environment.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class Param:
    """A generation parameter rendered as a control in the UI."""

    name: str
    label: str
    default: Any
    kind: str = "float"  # float | int | bool | choice | text
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    choices: Sequence[str] = ()
    info: str = ""


@dataclass(frozen=True)
class ModelVariant:
    repo_id: str
    label: str
    size: str = ""
    notes: str = ""


@dataclass(frozen=True)
class Capabilities:
    # Zero-shot cloning from a reference clip.
    clone: bool = False
    # Cloning works better/only with a transcript of the reference clip.
    clone_needs_transcript: bool = False
    # Built-in named voices (listed by the worker after load).
    presets: bool = False
    # Describe a voice in natural language ("a calm older man with a British accent").
    voice_design: bool = False
    # Style / emotion instruction on top of a voice.
    instruct: bool = False
    # Max number of speakers in one generation ("Speaker 1: ..." scripts).
    max_speakers: int = 1
    # Can generate without any voice input.
    no_voice: bool = False
    languages: Sequence[str] = ("English",)
    # Can load LoRA adapters produced by the Training tab.
    lora: bool = False
    finetune: bool = False


@dataclass(frozen=True)
class BackendSpec:
    id: str
    name: str
    description: str
    module: str  # import path of the implementation, e.g. "ovl.backends.impl.qwen3_tts"
    requirements: Sequence[str]
    variants: Sequence[ModelVariant]
    capabilities: Capabilities
    params: Sequence[Param] = ()
    python: str = "3.12"
    license: str = ""
    homepage: str = ""
    sample_rate: int = 24000
    # Extra `uv pip install` arguments, e.g. ["--prerelease=allow"].
    install_args: Sequence[str] = ()
    # Packages installed with --no-deps after the main install (for over-pinned repos).
    no_deps_requirements: Sequence[str] = ()
    supports_mps: bool = True
    supports_cpu: bool = True
    tips: str = ""
    # Bump to force existing environments to reinstall after requirement changes.
    env_version: int = 1

    @property
    def default_variant(self) -> ModelVariant:
        return self.variants[0]

    def default_params(self) -> Dict[str, Any]:
        return {p.name: p.default for p in self.params}


# Installed into every backend environment so workers can use ovl.audio.
BASE_REQUIREMENTS: List[str] = ["numpy", "soundfile", "soxr"]
