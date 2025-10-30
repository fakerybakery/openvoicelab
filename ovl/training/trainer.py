"""Training manager for VibeVoice fine-tuning with TensorBoard logging"""

import json
import logging
import os
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ovl.training.data_utils import prepare_training_data

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training"""

    model_path: str
    dataset_path: str
    output_dir: str
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
    eval_steps: int = 500
    warmup_steps: int = 100
    voice_prompt_drop_rate: float = 0.2
    # Critical parameters from official training example
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


class TrainingManager:
    """Manages training processes"""

    def __init__(self, runs_dir: str = "training_runs"):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.runs_dir / "state.json"
        self.tensorboard_process = None
        self.training_process = None

    def _load_state(self) -> Dict[str, Any]:
        """Load training state"""
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        return {}

    def _save_state(self, state: Dict[str, Any]):
        """Save training state"""
        self.state_file.write_text(json.dumps(state, indent=2))

    def get_active_run(self) -> Optional[Dict[str, Any]]:
        """Get currently active training run"""
        state = self._load_state()
        if state.get("active_run"):
            run_id = state["active_run"]
            run_dir = self.runs_dir / run_id
            if run_dir.exists():
                info_file = run_dir / "info.json"
                if info_file.exists():
                    info = json.loads(info_file.read_text())
                    info["run_id"] = run_id
                    info["run_dir"] = str(run_dir)

                    # Check if process is still running
                    pid_file = run_dir / "train.pid"
                    if pid_file.exists():
                        pid = int(pid_file.read_text().strip())
                        try:
                            os.kill(pid, 0)  # Check if process exists
                            info["status"] = "running"
                        except OSError:
                            info["status"] = "stopped"
                    else:
                        info["status"] = "stopped"

                    return info
        return None

    def list_runs(self) -> list[Dict[str, Any]]:
        """List all training runs"""
        runs = []
        for run_dir in sorted(self.runs_dir.iterdir(), reverse=True):
            if run_dir.is_dir() and run_dir.name.startswith("run_"):
                info_file = run_dir / "info.json"
                if info_file.exists():
                    info = json.loads(info_file.read_text())
                    info["run_id"] = run_dir.name

                    # Check if process is still running
                    pid_file = run_dir / "train.pid"
                    if pid_file.exists():
                        pid = int(pid_file.read_text().strip())
                        try:
                            os.kill(pid, 0)
                            info["status"] = "running"
                        except OSError:
                            info["status"] = "stopped"
                    else:
                        info["status"] = "stopped"

                    runs.append(info)
        return runs

    def start_training(self, config: TrainingConfig, progress_callback: Optional[Callable] = None) -> str:
        """Start training in background process"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp}"
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting training run: {run_id}")
        logger.info(f"Model: {config.model_path}, Dataset: {config.dataset_path}")
        info = {
            "created_at": datetime.now().isoformat(),
            "config": {
                "model_path": config.model_path,
                "dataset_path": config.dataset_path,
                "output_dir": config.output_dir,
                "num_epochs": config.num_epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "lora_r": config.lora_r,
                "lora_alpha": config.lora_alpha,
                "voice_prompt_drop_rate": config.voice_prompt_drop_rate,
            },
        }
        (run_dir / "info.json").write_text(json.dumps(info, indent=2))

        # Prepare training script arguments
        output_dir = run_dir / "checkpoints"
        output_dir.mkdir(exist_ok=True)

        try:
            logger.info(f"Preparing training data from {config.dataset_path}")
            jsonl_path = prepare_training_data(config.dataset_path)
            logger.info(f"Training data prepared: {jsonl_path}")
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            raise ValueError(f"Failed to prepare training data: {e}")

        # Build command
        train_script = Path(__file__).parent / "train_script.py"
        cmd = [
            sys.executable,
            str(train_script),
            "--model_name_or_path",
            config.model_path,
            "--dataset_name",
            "json",
            "--train_jsonl",
            jsonl_path,
            "--text_column_name",
            "text",
            "--audio_column_name",
            "audio",
            "--output_dir",
            str(output_dir),
            "--num_train_epochs",
            str(config.num_epochs),
            "--per_device_train_batch_size",
            str(config.batch_size),
            "--gradient_accumulation_steps",
            str(config.gradient_accumulation_steps),
            "--learning_rate",
            str(config.learning_rate),
            "--lora_r",
            str(config.lora_r),
            "--lora_alpha",
            str(config.lora_alpha),
            "--lora_dropout",
            str(config.lora_dropout),
            "--lora_target_modules",
            config.lora_target_modules,
            "--save_steps",
            str(config.save_steps),
            "--logging_steps",
            str(config.logging_steps),
            "--warmup_steps",
            str(config.warmup_steps),
            "--warmup_ratio",
            str(config.warmup_ratio),
            "--voice_prompt_drop_rate",
            str(config.voice_prompt_drop_rate),
            "--lr_scheduler_type",
            config.lr_scheduler_type,
            "--max_grad_norm",
            str(config.max_grad_norm),
            "--ddpm_batch_mul",
            str(config.ddpm_batch_mul),
            "--diffusion_loss_weight",
            str(config.diffusion_loss_weight),
            "--ce_loss_weight",
            str(config.ce_loss_weight),
            "--logging_dir",
            str(run_dir / "logs"),
            "--report_to",
            "tensorboard",
            "--remove_unused_columns",
            "False",
            "--do_train",
        ]

        if config.bf16:
            cmd.append("--bf16")
        if config.gradient_checkpointing:
            cmd.append("--gradient_checkpointing")
        if config.gradient_clipping:
            cmd.append("--gradient_clipping")
        if config.train_diffusion_head:
            cmd.append("--train_diffusion_head")
            cmd.append("True")

        log_file = run_dir / "train.log"
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(Path.cwd()),
            start_new_session=True,
            text=True,
            bufsize=1,
        )

        def tail_logs():
            """Tail training logs to both file and console"""
            with open(log_file, "w") as f:
                for line in process.stdout:
                    f.write(line)
                    f.flush()
                    print(line, end="")

        threading.Thread(target=tail_logs, daemon=True).start()
        (run_dir / "train.pid").write_text(str(process.pid))
        logger.info(f"Training process started with PID {process.pid}")

        state = self._load_state()
        state["active_run"] = run_id
        self._save_state(state)
        log_dir = run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        if self.tensorboard_process is None or self.tensorboard_process.poll() is not None:
            logger.info("Starting TensorBoard on port 6006")
            tb_log_file = run_dir / "tensorboard.log"
            with open(tb_log_file, "w") as tb_log:
                self.tensorboard_process = subprocess.Popen(
                    [
                        "tensorboard",
                        "--logdir",
                        str(log_dir),
                        "--port",
                        "6006",
                        "--bind_all",
                    ],
                    stdout=tb_log,
                    stderr=subprocess.STDOUT,
                )
            logger.info(f"TensorBoard started with PID {self.tensorboard_process.pid}")

        logger.info(f"Training run {run_id} setup complete")
        return run_id

    def stop_training(self, run_id: Optional[str] = None):
        """Stop training process with proper cleanup"""
        if run_id is None:
            active = self.get_active_run()
            if not active:
                logger.warning("No active training run to stop")
                return
            run_id = active["run_id"]

        logger.info(f"Stopping training run: {run_id}")
        run_dir = self.runs_dir / run_id
        pid_file = run_dir / "train.pid"

        if pid_file.exists():
            pid = int(pid_file.read_text().strip())
            logger.info(f"Sending interrupt signals to PID {pid}")

            for attempt in range(3):
                try:
                    os.kill(pid, signal.SIGINT)
                    import time

                    time.sleep(2)

                    try:
                        os.kill(pid, 0)
                    except OSError:
                        logger.info(f"Training process {pid} stopped successfully")
                        break

                    if attempt == 2:
                        logger.warning(f"Process {pid} did not respond to SIGINT, sending SIGTERM")
                        os.kill(pid, signal.SIGTERM)

                except OSError:
                    logger.info(f"Process {pid} already terminated")
                    break

            try:
                pid_file.unlink()
            except Exception as e:
                logger.error(f"Failed to remove PID file: {e}")

    def get_tensorboard_url(self, run_id: Optional[str] = None, port: int = 6006) -> str:
        """Get TensorBoard URL for a run"""
        if run_id is None:
            active = self.get_active_run()
            if not active:
                return ""
            run_id = active["run_id"]

        run_dir = self.runs_dir / run_id
        log_dir = run_dir / "logs"

        if not log_dir.exists():
            return ""

        if self.tensorboard_process is None or self.tensorboard_process.poll() is not None:
            self.tensorboard_process = subprocess.Popen(
                [
                    "tensorboard",
                    "--logdir",
                    str(log_dir),
                    "--port",
                    str(port),
                    "--bind_all",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        return f"http://localhost:{port}"

    def get_training_log(self, run_id: Optional[str] = None, lines: int = 100) -> str:
        """Get training log"""
        if run_id is None:
            active = self.get_active_run()
            if not active:
                return ""
            run_id = active["run_id"]

        run_dir = self.runs_dir / run_id
        log_file = run_dir / "train.log"

        if not log_file.exists():
            return ""

        with open(log_file) as f:
            all_lines = f.readlines()
            return "".join(all_lines[-lines:])

    def is_tensorboard_ready(self, run_id: Optional[str] = None) -> bool:
        """Check if TensorBoard is ready"""
        import socket

        if self.tensorboard_process is None or self.tensorboard_process.poll() is not None:
            return False

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("localhost", 6006))
            sock.close()
            return result == 0
        except:
            return False
