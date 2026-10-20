"""Training tab: LoRA fine-tuning for VibeVoice."""

import gradio as gr

from ovl.training import TrainingManager
from ovl.training.trainer import TrainingConfig
from ovl.ui.data import dataset_builder

training_manager = TrainingManager()


def dataset_choices():
    return [(f"{ds['name']} ({ds['num_samples']} clips)", ds["path"]) for ds in dataset_builder.list_datasets()]


def refresh_datasets():
    choices = dataset_choices()
    return gr.update(choices=choices, value=choices[0][1] if choices else None)


def runs_markdown() -> str:
    runs = training_manager.list_runs()
    if not runs:
        return "No training runs yet."
    lines = []
    for run in runs:
        icon = {"running": "🟢", "finished": "✅"}.get(run["status"], "⚫")
        cfg = run.get("config", {})
        lines.append(f"{icon} **{run['run_id']}**: {run['status']} · {cfg.get('model_path', '')} on "
                     f"`{cfg.get('dataset_path', '')}`")
        if run.get("lora_path"):
            lines.append(f"  - LoRA: `{run['lora_path']}` (pick it on the Generate tab)")
    return "\n".join(lines)


def hardware_note() -> str:
    import torch

    if torch.cuda.is_available():
        return ""
    return ("⚠️ No NVIDIA GPU detected. VibeVoice fine-tuning is designed for CUDA GPUs (≈16 GB+ VRAM for 1.5B); "
            "on other hardware it will be very slow or fail.")


def start_training(dataset_path, model_path, num_epochs, batch_size, grad_accum, learning_rate, lora_r,
                   voice_prompt_drop_rate, gradient_checkpointing):
    if not dataset_path:
        raise gr.Error("Select a dataset (build one on the Data tab)")
    import torch

    config = TrainingConfig(
        model_path=model_path,
        dataset_path=dataset_path,
        num_epochs=int(num_epochs),
        batch_size=int(batch_size),
        gradient_accumulation_steps=int(grad_accum),
        learning_rate=float(learning_rate),
        lora_r=int(lora_r),
        lora_alpha=int(lora_r) * 4,
        voice_prompt_drop_rate=float(voice_prompt_drop_rate),
        gradient_checkpointing=bool(gradient_checkpointing),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )
    try:
        run_id = training_manager.start_training(config)
    except Exception as e:
        raise gr.Error(f"Failed to start training: {e}")
    gr.Success(f"Training started: {run_id}")
    return f"Training **{run_id}**. TensorBoard will appear below once the first metrics are logged.", runs_markdown()


def stop_training():
    active = training_manager.get_active_run()
    if not active:
        raise gr.Error("No training is running")
    training_manager.stop_training(active["run_id"])
    return f"Stopped {active['run_id']}", runs_markdown()


def tensorboard_html():
    url = training_manager.ensure_tensorboard()
    if not training_manager.is_tensorboard_ready():
        return "<p>TensorBoard is starting... refresh in a few seconds.</p>"
    return f'<iframe src="{url}" width="100%" height="720px" style="border:0"></iframe>'


def view_logs():
    active = training_manager.get_active_run() or next(iter(training_manager.list_runs()), None)
    return training_manager.get_training_log(active["run_id"]) if active else ""


def build(demo: gr.Blocks):
    gr.Markdown("Fine-tune VibeVoice on a dataset from the **Data** tab. The result is a LoRA adapter you can load "
                "on the **Generate** tab. Training runs in the VibeVoice environment, so install it on the **Models** tab first.")
    note = gr.Markdown()
    with gr.Row():
        with gr.Column(scale=1):
            dataset = gr.Dropdown(label="Dataset", choices=dataset_choices())
            refresh_ds = gr.Button("Refresh datasets", size="sm")
            model_path = gr.Dropdown(
                ["vibevoice/VibeVoice-1.5B", "vibevoice/VibeVoice-7B"],
                value="vibevoice/VibeVoice-1.5B",
                label="Base model",
                allow_custom_value=True,
            )
            with gr.Accordion("Training parameters", open=False):
                num_epochs = gr.Slider(1, 50, value=3, step=1, label="Epochs")
                batch_size = gr.Slider(1, 16, value=8, step=1, label="Batch size", info="Official recommendation: 8")
                grad_accum = gr.Slider(1, 64, value=1, step=1, label="Gradient accumulation steps")
                learning_rate = gr.Number(label="Learning rate", value=2.5e-5)
                lora_r = gr.Slider(4, 128, value=8, step=4, label="LoRA rank", info="alpha is set to 4× rank")
                voice_prompt_drop_rate = gr.Slider(
                    0.0, 1.0, value=0.2, step=0.05, label="Voice prompt drop rate",
                    info="0.2 keeps voice cloning working; 1.0 trains a single fixed voice (more natural, no cloning)",
                )
                gradient_checkpointing = gr.Checkbox(label="Gradient checkpointing", value=False,
                                                     info="Saves memory at the cost of speed")
            with gr.Row():
                start_btn = gr.Button("Start training", variant="primary")
                stop_btn = gr.Button("Stop", variant="stop")
            status = gr.Markdown()
            gr.Markdown("### Runs")
            runs = gr.Markdown()
            refresh_runs = gr.Button("Refresh runs", size="sm")
        with gr.Column(scale=2):
            tensorboard = gr.HTML("<p>Start a run to see TensorBoard.</p>")
            refresh_tb = gr.Button("Show TensorBoard", size="sm")
            with gr.Accordion("Training log", open=False):
                logs = gr.Textbox(lines=20, max_lines=30, show_label=False, interactive=False, autoscroll=True)
                refresh_logs = gr.Button("Refresh log", size="sm")

    refresh_ds.click(refresh_datasets, None, dataset)
    start_btn.click(
        start_training,
        [dataset, model_path, num_epochs, batch_size, grad_accum, learning_rate, lora_r, voice_prompt_drop_rate,
         gradient_checkpointing],
        [status, runs],
    )
    stop_btn.click(stop_training, None, [status, runs])
    refresh_runs.click(runs_markdown, None, runs)
    refresh_tb.click(tensorboard_html, None, tensorboard)
    refresh_logs.click(view_logs, None, logs)
    demo.load(lambda: (hardware_note(), refresh_datasets(), runs_markdown()), None, [note, dataset, runs])
