"""UI-side handle to a model worker process (see worker.py)."""

import collections
import json
import logging
import queue
import subprocess
import threading
from typing import Any, Callable, Dict, Optional

from ovl.backends.spec import BackendSpec
from ovl.runtime import envs

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[Optional[float], str], None]


class WorkerError(RuntimeError):
    def __init__(self, message: str, details: str = ""):
        super().__init__(message)
        self.details = details


class WorkerClient:
    def __init__(self, spec: BackendSpec):
        self.spec = spec
        self.log_lines = collections.deque(maxlen=400)
        self._messages: "queue.Queue[dict]" = queue.Queue()
        self._next_id = 1
        self._lock = threading.Lock()  # one request at a time
        self.proc = subprocess.Popen(
            envs.worker_command(spec),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=envs.worker_env(),
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        self._wait(0, None)  # ready handshake

    def _read_stdout(self):
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._messages.put(json.loads(line))
            except json.JSONDecodeError:
                self.log_lines.append(line)
        self._messages.put({"id": None, "dead": True})

    def _read_stderr(self):
        for line in self.proc.stderr:
            line = line.rstrip()
            # tqdm writes carriage-return updates; keep only the latest state.
            if "\r" in line:
                line = line.split("\r")[-1]
            self.log_lines.append(line)
            logger.debug("[%s] %s", self.spec.id, line)

    def recent_log(self, n: int = 40) -> str:
        return "\n".join(list(self.log_lines)[-n:])

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def _wait(self, msg_id: int, on_progress: Optional[ProgressCallback]) -> Dict[str, Any]:
        while True:
            msg = self._messages.get()
            if msg.get("dead"):
                self.proc.wait()
                raise WorkerError(
                    f"{self.spec.name} worker exited unexpectedly (code {self.proc.returncode})",
                    self.recent_log(),
                )
            if msg.get("id") != msg_id:
                continue
            if msg.get("event") == "progress":
                if on_progress:
                    on_progress(msg.get("value"), msg.get("desc", ""))
                continue
            if "error" in msg:
                raise WorkerError(msg["error"], msg.get("traceback", ""))
            return msg.get("result", {})

    def call(self, method: str, params: Dict[str, Any], on_progress: Optional[ProgressCallback] = None) -> Dict[str, Any]:
        with self._lock:
            if not self.alive:
                raise WorkerError(f"{self.spec.name} worker is not running", self.recent_log())
            msg_id = self._next_id
            self._next_id += 1
            self.proc.stdin.write(json.dumps({"id": msg_id, "method": method, "params": params}) + "\n")
            self.proc.stdin.flush()
            return self._wait(msg_id, on_progress)

    def close(self, timeout: float = 10):
        if not self.alive:
            return
        try:
            self.proc.stdin.write(json.dumps({"id": -1, "method": "shutdown"}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=timeout)
        except Exception:
            pass
        if self.alive:
            self.proc.kill()
            self.proc.wait()
