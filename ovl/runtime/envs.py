"""Per-backend Python environments, managed with uv.

Each backend gets envs/<backend_id>/ with its own interpreter and pinned
dependencies, so models with conflicting requirements (transformers 4.x vs 5.x,
old torch pins, ...) can all be installed side by side. uv hard-links packages
from its global cache, so shared wheels like torch are only stored once.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterator, List, Optional

from ovl import paths
from ovl.backends.spec import BASE_REQUIREMENTS, BackendSpec

logger = logging.getLogger(__name__)

MARKER = "ovl-env.json"


def uv_bin() -> str:
    try:
        import uv

        return uv.find_uv_bin()
    except Exception:
        found = shutil.which("uv")
        if found:
            return found
    raise RuntimeError("uv is not installed. Run `pip install uv` in the OpenVoiceLab environment.")


def env_dir(spec: BackendSpec) -> Path:
    return paths.envs_dir() / spec.id


def env_python(spec: BackendSpec) -> Path:
    d = env_dir(spec)
    return d / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _fingerprint(spec: BackendSpec) -> dict:
    return {
        "env_version": spec.env_version,
        "python": spec.python,
        "requirements": list(spec.requirements),
        "no_deps_requirements": list(spec.no_deps_requirements),
    }


def status(spec: BackendSpec) -> str:
    """'missing', 'outdated' or 'ready'."""
    marker = env_dir(spec) / MARKER
    if not env_python(spec).exists() or not marker.exists():
        return "missing"
    try:
        if json.loads(marker.read_text()) != _fingerprint(spec):
            return "outdated"
    except Exception:
        return "outdated"
    return "ready"


def _run(cmd: List[str], log: Callable[[str], None], env: Optional[dict] = None):
    log("$ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, **(env or {})},
    )
    for line in proc.stdout:
        log(line.rstrip())
    if proc.wait() != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def install(spec: BackendSpec, log: Callable[[str], None] = logger.info, reinstall: bool = False):
    """Create (or update) the environment for a backend."""
    uv = uv_bin()
    d = env_dir(spec)
    if reinstall and d.exists():
        shutil.rmtree(d)
    (d / MARKER).unlink(missing_ok=True)

    uv_env = {"UV_TORCH_BACKEND": os.environ.get("UV_TORCH_BACKEND", "auto")}
    if not env_python(spec).exists():
        _run([uv, "venv", "--seed", "--python", spec.python, str(d)], log, uv_env)

    py = str(env_python(spec))
    reqs = list(BASE_REQUIREMENTS) + list(spec.requirements)
    _run([uv, "pip", "install", "--python", py, *spec.install_args, *reqs], log, uv_env)
    if spec.no_deps_requirements:
        _run([uv, "pip", "install", "--python", py, "--no-deps", *spec.no_deps_requirements], log, uv_env)

    (d / MARKER).write_text(json.dumps(_fingerprint(spec), indent=2))
    log(f"✓ {spec.name} environment ready")


def remove(spec: BackendSpec):
    d = env_dir(spec)
    if d.exists():
        shutil.rmtree(d)


def worker_command(spec: BackendSpec) -> List[str]:
    return [str(env_python(spec)), "-m", "ovl.runtime.worker", spec.module]


def _ovl_import_root() -> str:
    """A directory containing only the `ovl` package, for workers' PYTHONPATH.

    In a source checkout that's the repo root. When installed into site-packages we
    must not expose the whole site-packages dir to workers, so link the package
    into a dedicated folder instead.
    """
    if (paths.REPO_ROOT / "pyproject.toml").exists():
        return str(paths.REPO_ROOT)
    root = paths.envs_dir() / "_ovl"
    link = root / "ovl"
    if not link.exists():
        root.mkdir(parents=True, exist_ok=True)
        try:
            link.symlink_to(paths.PACKAGE_DIR, target_is_directory=True)
        except OSError:  # e.g. Windows without symlink permission
            shutil.copytree(paths.PACKAGE_DIR, link, ignore=shutil.ignore_patterns("voices", "__pycache__"))
    return str(root)


def worker_env() -> dict:
    """Environment for worker processes: make the ovl package importable."""
    env = dict(os.environ)
    root = _ovl_import_root()
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["PYTHONUNBUFFERED"] = "1"
    env["OVL_HOME"] = str(paths.home())
    return env
