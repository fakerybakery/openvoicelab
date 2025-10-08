import platform

import gradio as gr
import torch

from ovl import __version__
from ovl.ui.data import data_tab
from ovl.ui.inference import inference_tab
from ovl.ui.settings import settings_tab
from ovl.ui.training import training_tab
from ovl.ui.voices import voices_tab
from ovl.utils import setup_logging

setup_logging(level="INFO", log_file="logs/openvoicelab.log")


def get_system_info():
    """Get system information for footer"""
    info_parts = [f"OpenVoiceLab v{__version__}"]

    # Detect available devices
    devices = []
    if torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(0)
        devices.append(f"CUDA ({cuda_name})")
    if torch.backends.mps.is_available():
        devices.append("MPS")
    if not devices:
        devices.append("CPU")

    info_parts.append(f"Devices: {', '.join(devices)}")

    # Python version
    python_version = platform.python_version()
    info_parts.append(f"Python {python_version}")

    # PyTorch version
    torch_version = torch.__version__
    info_parts.append(f"PyTorch {torch_version}")

    return " • ".join(info_parts)


with gr.Blocks() as app:
    gr.TabbedInterface(
        [
            inference_tab,
            voices_tab,
            data_tab,
            training_tab,
            settings_tab,
        ],
        [
            "Inference",
            "Voices",
            "Data",
            "Training",
            "Settings",
        ],
    )

    gr.Markdown(
        f"<div style='text-align: center; margin-top: 20px; padding: 10px; opacity: 0.6; font-size: 0.9em;'>{get_system_info()}</div>",
        elem_classes="version-footer",
    )
