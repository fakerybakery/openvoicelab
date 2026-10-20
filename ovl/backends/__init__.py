"""Registry of TTS backends.

Each backend is described by a spec in ovl/backends/catalog/<id>.py (plain data,
cheap to import) and implemented in ovl/backends/impl/<id>.py (imported only in the
backend's own environment by the worker process).
"""

import importlib
import logging
import pkgutil
from typing import Dict, List

from ovl.backends.spec import BackendSpec, Capabilities, ModelVariant, Param

logger = logging.getLogger(__name__)

# Display order in the UI; backends not listed here are appended alphabetically.
ORDER = [
    "vibevoice",
    "qwen3_tts",
    "voxcpm",
    "chatterbox",
    "omnivoice",
    "echo_tts",
    "moss_tts",
    "kitten_tts",
    "pocket_tts",
    "kokoro",
]

_registry: Dict[str, BackendSpec] = {}


def _load():
    if _registry:
        return
    from ovl.backends import catalog

    found = {}
    for info in pkgutil.iter_modules(catalog.__path__):
        try:
            mod = importlib.import_module(f"{catalog.__name__}.{info.name}")
            found[mod.SPEC.id] = mod.SPEC
        except Exception:
            logger.exception("Failed to load backend spec %s", info.name)
    for key in ORDER + sorted(k for k in found if k not in ORDER):
        if key in found:
            _registry[key] = found[key]


def all_backends() -> List[BackendSpec]:
    _load()
    return list(_registry.values())


def get(backend_id: str) -> BackendSpec:
    _load()
    return _registry[backend_id]


__all__ = ["BackendSpec", "Capabilities", "ModelVariant", "Param", "all_backends", "get"]
