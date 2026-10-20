"""Model worker: runs inside a backend's environment, driven by the UI over stdio.

Protocol: one JSON object per line.
  request:  {"id": 1, "method": "load" | "generate" | "ping", "params": {...}}
  replies:  {"id": 1, "event": "progress", "value": 0.5, "desc": "..."}
            {"id": 1, "result": {...}}
            {"id": 1, "error": "message", "traceback": "..."}

The real stdout is reserved for the protocol; anything the model libraries print
(including from C extensions) is redirected to stderr, which the UI logs.

Usage: python -m ovl.runtime.worker <module.path.of.Backend>
"""

import importlib
import json
import os
import sys
import time
import traceback


def _claim_stdout():
    proto_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(proto_fd, "w", buffering=1, encoding="utf-8")


def main():
    proto = _claim_stdout()
    if sys.platform == "darwin":
        # transformers 5 copies weights to the device from a thread pool, which deadlocks on MPS.
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

    def send(msg):
        proto.write(json.dumps(msg) + "\n")
        proto.flush()

    module_path = sys.argv[1]
    try:
        module = importlib.import_module(module_path)
        backend_cls = getattr(module, "Backend")
    except Exception as e:
        send({"id": 0, "error": f"Failed to import {module_path}: {e}", "traceback": traceback.format_exc()})
        return 1

    from ovl.backends.base import Request

    current_id = {"id": 0}

    def progress(value, desc=""):
        send({"id": current_id["id"], "event": "progress", "value": value, "desc": desc})

    backend = backend_cls(progress)
    send({"id": 0, "result": {"ready": True, "pid": os.getpid()}})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        current_id["id"] = msg.get("id", 0)
        method, params = msg.get("method"), msg.get("params") or {}
        try:
            if method == "ping":
                result = {"pong": True}
            elif method == "load":
                info = backend.load(params["model_id"], params.get("device") or "auto", params.get("options") or {})
                result = info.__dict__
            elif method == "generate":
                request = Request.from_dict(params)
                start = time.time()
                metrics = backend.generate(request) or {}
                metrics.setdefault("generation_time", time.time() - start)
                result = {"audio_path": request.output_path, "metrics": metrics}
            elif method == "shutdown":
                backend.unload()
                send({"id": current_id["id"], "result": {}})
                return 0
            else:
                raise ValueError(f"Unknown method {method!r}")
            send({"id": current_id["id"], "result": result})
        except (Exception, SystemExit) as e:
            # Some libraries sys.exit() on failure (e.g. auto-downloads); report instead of dying.
            traceback.print_exc()
            send({"id": current_id["id"], "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
