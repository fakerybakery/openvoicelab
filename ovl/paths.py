"""Filesystem layout for OpenVoiceLab.

Everything the app writes (datasets, outputs, training runs, model environments,
user voices) lives under a single home directory. It defaults to the current
working directory so existing installs keep their `data/`, `training_runs/`, etc.,
and can be moved anywhere with the OVL_HOME environment variable.
"""

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
BUILTIN_VOICES_DIR = PACKAGE_DIR / "voices"


def home() -> Path:
    return Path(os.environ.get("OVL_HOME", os.getcwd())).resolve()


def _dir(name: str) -> Path:
    path = home() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    return _dir("data")


def outputs_dir() -> Path:
    return _dir("outputs")


def runs_dir() -> Path:
    return _dir("training_runs")


def envs_dir() -> Path:
    return _dir("envs")


def logs_dir() -> Path:
    return _dir("logs")


def voices_dir() -> Path:
    """User voice library. Seeded from the bundled voices on first use."""
    path = home() / "voices"
    if not path.exists():
        import shutil

        path.mkdir(parents=True)
        for pattern in ("*.wav", "*.txt"):
            for src in BUILTIN_VOICES_DIR.glob(pattern):
                shutil.copy(src, path / src.name)
    return path
