import platform

import gradio as gr

from ovl import __version__, paths
from ovl.utils import setup_logging

CSS = """
.ovl-footer { text-align: center; opacity: 0.6; font-size: 0.85em; margin-top: 12px; }
"""


def system_info() -> str:
    import torch

    devices = []
    if torch.cuda.is_available():
        devices.append(f"CUDA ({torch.cuda.get_device_name(0)})")
    if torch.backends.mps.is_available():
        devices.append("Apple MPS")
    devices.append("CPU")
    return f"OpenVoiceLab v{__version__} · {', '.join(devices)} · Python {platform.python_version()} · data in {paths.home()}"


def create_app() -> gr.Blocks:
    from ovl.ui import data, inference, models, training, voices

    with gr.Blocks(title="OpenVoiceLab", analytics_enabled=False) as app:
        gr.Markdown("# OpenVoiceLab")
        with gr.Tabs():
            with gr.Tab("Generate"):
                inference.build(app)
            with gr.Tab("Voices"):
                voices.build(app)
            with gr.Tab("Models"):
                models.build(app)
            with gr.Tab("Data"):
                data.build(app)
            with gr.Tab("Training"):
                training.build(app)
        gr.Markdown(system_info(), elem_classes="ovl-footer")
    return app


def launch(**kwargs):
    import atexit

    from ovl.runtime.session import session

    setup_logging(level="DEBUG" if kwargs.get("debug") else "INFO", log_file=str(paths.logs_dir() / "openvoicelab.log"))
    atexit.register(session.unload)
    app = create_app()
    app.queue(default_concurrency_limit=1)
    app.launch(
        theme=gr.themes.Soft(primary_hue="indigo"),
        css=CSS,
        footer_links=["gradio", "settings"],
        allowed_paths=[str(paths.home())],
        **kwargs,
    )
