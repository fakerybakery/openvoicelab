"""VibeVoice LoRA fine-tuning, run as a subprocess in the VibeVoice environment."""

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ovl import paths
from ovl.training.data_utils import prepare_training_data

logger = logging.getLogger(__name__)

TENSORBOARD_PORT = 6006


@dataclass
class TrainingConfig:
    model_path: str
    dataset_path: str
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2.5e-5
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = False
    bf16: bool = True
    save_steps: int = 500
    logging_steps: int = 10
    voice_prompt_drop_rate: float = 0.2
    # Defaults from the official VibeVoice finetuning example.
    train_diffusion_head: bool = True
    gradient_clipping: bool = True
    ddpm_batch_mul: int = 4
    diffusion_loss_weight: float = 1.4
    ce_loss_weight: float = 0.04
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    max_grad_norm: float = 0.8
    gradient_accumulation_steps: int = 1


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class TrainingManager:
    def __init__(self, runs_dir: Optional[str] = None):
        self.runs_dir = Path(runs_dir) if runs_dir else paths.runs_dir()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_process: Optional[subprocess.Popen] = None
        self._processes: Dict[str, subprocess.Popen] = {}

    def _run_info(self, run_dir: Path) -> Optional[Dict[str, Any]]:
        info_file = run_dir / "info.json"
        if not info_file.exists():
            return None
        info = json.loads(info_file.read_text())
        info["run_id"] = run_dir.name
        info["run_dir"] = str(run_dir)
        pid_file = run_dir / "train.pid"
        running = pid_file.exists() and _pid_alive(int(pid_file.read_text().strip()))
        info["status"] = "running" if running else info.get("final_status", "stopped")
        lora = run_dir / "checkpoints" / "lora"
        info["lora_path"] = str(lora) if lora.is_dir() else None
        return info

    def list_runs(self) -> List[Dict[str, Any]]:
        runs = (self._run_info(d) for d in sorted(self.runs_dir.iterdir(), reverse=True) if d.name.startswith("run_"))
        return [r for r in runs if r]

    def get_active_run(self) -> Optional[Dict[str, Any]]:
        return next((r for r in self.list_runs() if r["status"] == "running"), None)

    def start_training(self, config: TrainingConfig) -> str:
        from ovl.backends import get
        from ovl.runtime import envs

        if self.get_active_run():
            raise RuntimeError("A training run is already in progress")
        spec = get("vibevoice")
        if envs.status(spec) != "ready":
            raise RuntimeError("Install VibeVoice on the Models tab first; training runs in its environment.")

        jsonl_path = prepare_training_data(config.dataset_path)
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = self.runs_dir / run_id
        output_dir = run_dir / "checkpoints"
        log_dir = run_dir / "logs"
        output_dir.mkdir(parents=True)
        log_dir.mkdir()
        (run_dir / "info.json").write_text(
            json.dumps({"created_at": datetime.now().isoformat(timespec="seconds"), "config": asdict(config)}, indent=2)
        )

        c = config
        cmd = [
            str(envs.env_python(spec)), "-m", "vibevoice.finetune.train_vibevoice",
            "--model_name_or_path", c.model_path,
            "--dataset_name", "json",
            "--train_jsonl", jsonl_path,
            "--text_column_name", "text",
            "--audio_column_name", "audio",
            "--output_dir", str(output_dir),
            "--num_train_epochs", str(c.num_epochs),
            "--per_device_train_batch_size", str(c.batch_size),
            "--gradient_accumulation_steps", str(c.gradient_accumulation_steps),
            "--learning_rate", str(c.learning_rate),
            "--lora_r", str(c.lora_r),
            "--lora_alpha", str(c.lora_alpha),
            "--lora_dropout", str(c.lora_dropout),
            "--lora_target_modules", c.lora_target_modules,
            "--save_steps", str(c.save_steps),
            "--logging_steps", str(c.logging_steps),
            "--warmup_ratio", str(c.warmup_ratio),
            "--voice_prompt_drop_rate", str(c.voice_prompt_drop_rate),
            "--lr_scheduler_type", c.lr_scheduler_type,
            "--max_grad_norm", str(c.max_grad_norm),
            "--ddpm_batch_mul", str(c.ddpm_batch_mul),
            "--diffusion_loss_weight", str(c.diffusion_loss_weight),
            "--ce_loss_weight", str(c.ce_loss_weight),
            "--logging_dir", str(log_dir),
            "--report_to", "tensorboard",
            "--remove_unused_columns", "False",
            "--do_train",
        ]
        if c.bf16:
            cmd.append("--bf16")
        if c.gradient_checkpointing:
            cmd.append("--gradient_checkpointing")
        if c.gradient_clipping:
            cmd.append("--gradient_clipping")
        if c.train_diffusion_head:
            cmd += ["--train_diffusion_head", "True"]

        logger.info("Starting training run %s: %s", run_id, " ".join(cmd))
        popen_kwargs = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        )
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(run_dir),
            env=envs.worker_env(),
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **popen_kwargs,
        )
        (run_dir / "train.pid").write_text(str(process.pid))
        self._processes[run_id] = process

        def tail_logs():
            with open(run_dir / "train.log", "w", encoding="utf-8") as f:
                for line in process.stdout:
                    f.write(line)
                    f.flush()
            code = process.wait()
            info = json.loads((run_dir / "info.json").read_text())
            info["final_status"] = "finished" if code == 0 else f"failed (exit {code})"
            (run_dir / "info.json").write_text(json.dumps(info, indent=2))
            (run_dir / "train.pid").unlink(missing_ok=True)

        threading.Thread(target=tail_logs, daemon=True).start()
        self.ensure_tensorboard()
        return run_id

    def stop_training(self, run_id: str):
        run_dir = self.runs_dir / run_id
        pid_file = run_dir / "train.pid"
        if not pid_file.exists():
            return
        pid = int(pid_file.read_text().strip())
        logger.info("Stopping training run %s (PID %s)", run_id, pid)
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            # SIGINT lets the HF Trainer stop cleanly; escalate if it doesn't.
            for sig, wait in ((signal.SIGINT, 10), (signal.SIGTERM, 5), (signal.SIGKILL, 0)):
                try:
                    os.killpg(pid, sig)
                except OSError:
                    break
                deadline = time.time() + wait
                while time.time() < deadline and _pid_alive(pid):
                    time.sleep(0.5)
                if not _pid_alive(pid):
                    break
        pid_file.unlink(missing_ok=True)

    def ensure_tensorboard(self) -> str:
        if self.tensorboard_process is None or self.tensorboard_process.poll() is not None:
            self.tensorboard_process = subprocess.Popen(
                [sys.executable, "-m", "tensorboard.main", "--logdir", str(self.runs_dir),
                 "--port", str(TENSORBOARD_PORT), "--host", "127.0.0.1"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return f"http://localhost:{TENSORBOARD_PORT}"

    def is_tensorboard_ready(self) -> bool:
        if self.tensorboard_process is None or self.tensorboard_process.poll() is not None:
            return False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            return sock.connect_ex(("127.0.0.1", TENSORBOARD_PORT)) == 0

    def get_training_log(self, run_id: str, lines: int = 200) -> str:
        log_file = self.runs_dir / run_id / "train.log"
        if not log_file.exists():
            return ""
        return "".join(log_file.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-lines:])
