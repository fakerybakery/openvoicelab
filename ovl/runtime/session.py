"""The currently loaded model, shared by the UI and the CLI."""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ovl import paths
from ovl.audio import duration
from ovl.backends import BackendSpec
from ovl.runtime import envs
from ovl.runtime.client import ProgressCallback, WorkerClient

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    spec: BackendSpec
    model_id: str
    device: str
    options: Dict[str, Any]
    info: Dict[str, Any]
    client: WorkerClient

    def capability(self, name: str):
        """Checkpoint-specific capability overrides win over the backend spec."""
        overrides = self.info.get("capabilities") or {}
        return overrides.get(name, getattr(self.spec.capabilities, name))


@dataclass
class GenerationResult:
    audio_path: str
    duration: float
    generation_time: float
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def rtf(self) -> float:
        return self.generation_time / self.duration if self.duration else float("inf")


class Session:
    def __init__(self):
        self.current: Optional[LoadedModel] = None
        self.loading = False
        self._lock = threading.Lock()

    def load(
        self,
        spec: BackendSpec,
        model_id: str,
        device: str = "auto",
        options: Optional[Dict[str, Any]] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> LoadedModel:
        if envs.status(spec) != "ready":
            raise RuntimeError(f"{spec.name} is not installed yet. Install it from the Models tab first.")
        with self._lock:
            self.loading = True
            try:
                self._unload_locked()
                if on_progress:
                    on_progress(None, f"Starting {spec.name} worker...")
                client = WorkerClient(spec)
                try:
                    info = client.call(
                        "load",
                        {"model_id": model_id, "device": device, "options": options or {}},
                        on_progress,
                    )
                except Exception:
                    client.close(timeout=2)
                    raise
                self.current = LoadedModel(spec, model_id, device, options or {}, info, client)
                return self.current
            finally:
                self.loading = False

    def _unload_locked(self):
        if self.current is not None:
            logger.info("Unloading %s", self.current.model_id)
            self.current.client.close()
            self.current = None

    def unload(self):
        with self._lock:
            self._unload_locked()

    def cancel(self):
        """Abort an in-flight generation by killing the worker. The model must be reloaded."""
        if self.current is not None:
            self.current.client.proc.kill()

    def generate(
        self,
        text: str,
        voices: Optional[List[Dict[str, Any]]] = None,
        language: Optional[str] = None,
        instruct: Optional[str] = None,
        seed: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> GenerationResult:
        model = self.current
        if model is None:
            raise RuntimeError("No model loaded")
        if not model.client.alive:
            self.current = None
            raise RuntimeError("The model worker stopped (crashed or was cancelled). Load the model again.")
        if output_path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            output_path = str(paths.outputs_dir() / f"{model.spec.id}_{stamp}.wav")
        start = time.time()
        result = model.client.call(
            "generate",
            {
                "text": text,
                "output_path": output_path,
                "voices": voices or [],
                "language": language,
                "instruct": instruct,
                "seed": seed,
                "params": params or {},
            },
            on_progress,
        )
        elapsed = time.time() - start
        metrics = result.get("metrics") or {}
        return GenerationResult(
            audio_path=result["audio_path"],
            duration=duration(result["audio_path"]),
            generation_time=metrics.pop("generation_time", elapsed),
            metrics=metrics,
        )


session = Session()
