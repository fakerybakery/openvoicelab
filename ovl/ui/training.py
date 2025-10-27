import gradio as gr

from ovl.data import DatasetBuilder
from ovl.training import TrainingManager
from ovl.training.trainer import TrainingConfig

# Global state
training_manager = TrainingManager()
dataset_builder = DatasetBuilder()


def refresh_datasets():
    """Refresh available datasets"""
    datasets = dataset_builder.list_datasets()
    if not datasets:
        return gr.update(choices=[], value=None)

    choices = [f"{ds['name']} ({ds['num_samples']} samples)" for ds in datasets]
    return gr.update(choices=choices, value=choices[0] if choices else None)


def refresh_runs():
    """Refresh training runs list"""
    runs = training_manager.list_runs()

    if not runs:
        return gr.update(value="No training runs yet")

    runs_text = "## Training Runs\n\n"
    for run in runs:
        status_emoji = "🟢" if run["status"] == "running" else "⚫"
        runs_text += f"{status_emoji} **{run['run_id']}**\n"
        runs_text += f"- Created: {run['created_at']}\n"
        runs_text += f"- Status: {run['status']}\n"
        runs_text += f"- Dataset: {run['config']['dataset_path']}\n\n"

    return gr.update(value=runs_text)


def start_training(
    dataset_choice,
    model_path,
    num_epochs,
    batch_size,
    learning_rate,
    lora_r,
    voice_prompt_drop_rate,
    progress=gr.Progress(),
):
    """Start training"""
    if not dataset_choice:
        raise gr.Error("Please select a dataset")

    # Extract dataset name from choice
    dataset_name = dataset_choice.split(" (")[0]
    dataset_path = f"data/{dataset_name}"

    if not model_path:
        raise gr.Error("Please provide model path")

    # Create config
    config = TrainingConfig(
        model_path=model_path,
        dataset_path=dataset_path,
        output_dir=f"training_runs/output_{dataset_name}",
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lora_r=lora_r,
        voice_prompt_drop_rate=voice_prompt_drop_rate,
    )

    try:
        run_id = training_manager.start_training(config)

        return (
            gr.update(value=f"✅ Training started: {run_id}\n\nWaiting for TensorBoard..."),
            gr.update(value="<p>Waiting for TensorBoard to start... Click Refresh TensorBoard in a few seconds.</p>"),
            refresh_runs(),
        )
    except Exception as e:
        raise gr.Error(f"Failed to start training: {str(e)}")


def stop_training():
    """Stop current training"""
    active = training_manager.get_active_run()
    if not active:
        raise gr.Error("No active training to stop")

    training_manager.stop_training(active["run_id"])

    return (gr.update(value=f"⏹️ Training stopped: {active['run_id']}"), refresh_runs())


def view_tensorboard():
    """Load TensorBoard for active run"""
    active = training_manager.get_active_run()
    if not active:
        return gr.update(value="<p>No active training run</p>")

    # Check if TensorBoard is actually running
    if not training_manager.is_tensorboard_ready(active["run_id"]):
        return gr.update(value="<p>TensorBoard is starting... Please wait a few more seconds and refresh.</p>")

    tb_url = training_manager.get_tensorboard_url(active["run_id"])
    return gr.update(value=f'<iframe src="{tb_url}" width="100%" height="800px"></iframe>')


def view_logs():
    """View training logs"""
    active = training_manager.get_active_run()
    if not active:
        return "No active training run"

    logs = training_manager.get_training_log(active["run_id"])
    return logs


with gr.Blocks() as training_tab:
    gr.Markdown("# Training")
    gr.Markdown("Fine-tune VibeVoice on your datasets")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Configuration")

            dataset_dropdown = gr.Dropdown(label="Dataset", choices=[], info="Select dataset to train on")

            refresh_datasets_btn = gr.Button("🔄 Refresh Datasets", size="sm")

            model_path = gr.Textbox(
                label="Model Path",
                value="vibevoice/VibeVoice-1.5B",
                info="HuggingFace model ID or local path",
            )

            with gr.Accordion("Training Parameters", open=False):
                num_epochs = gr.Slider(label="Epochs", minimum=1, maximum=20, value=3, step=1)

                batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=16, value=4, step=1)

                learning_rate = gr.Number(label="Learning Rate", value=1e-4, precision=6)

                lora_r = gr.Slider(label="LoRA Rank (r)", minimum=4, maximum=64, value=8, step=4)

                voice_prompt_drop_rate = gr.Slider(
                    label="Voice Prompt Drop Rate",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.2,
                    step=0.1,
                    info="Default 0.2 for multi-speaker. Use 1.0 for single-speaker (disables voice cloning but improves naturalness)",
                )

            with gr.Row():
                start_btn = gr.Button("▶️ Start Training", variant="primary", size="lg")
                stop_btn = gr.Button("⏹️ Stop Training", variant="stop", size="lg")

            status_text = gr.Markdown("")

        with gr.Column(scale=2):
            gr.Markdown("### TensorBoard")

            tensorboard_html = gr.HTML("<p>Start training to view TensorBoard</p>")
            refresh_tb_btn = gr.Button("🔄 Refresh TensorBoard")

            with gr.Accordion("Training Logs", open=False):
                logs_text = gr.Textbox(label="Logs", lines=20, max_lines=30, interactive=False)
                refresh_logs_btn = gr.Button("🔄 Refresh Logs")

    with gr.Row():
        gr.Markdown("### Training History")

    with gr.Row():
        runs_list = gr.Markdown("No training runs yet")
        refresh_runs_btn = gr.Button("🔄 Refresh Runs")

    # Event handlers
    refresh_datasets_btn.click(fn=refresh_datasets, outputs=[dataset_dropdown])

    start_btn.click(
        fn=start_training,
        inputs=[
            dataset_dropdown,
            model_path,
            num_epochs,
            batch_size,
            learning_rate,
            lora_r,
            voice_prompt_drop_rate,
        ],
        outputs=[status_text, tensorboard_html, runs_list],
    )

    stop_btn.click(fn=stop_training, outputs=[status_text, runs_list])

    refresh_tb_btn.click(fn=view_tensorboard, outputs=[tensorboard_html])

    refresh_logs_btn.click(fn=view_logs, outputs=[logs_text])

    refresh_runs_btn.click(fn=refresh_runs, outputs=[runs_list])

    # Auto-load on tab load
    training_tab.load(fn=refresh_datasets, outputs=[dataset_dropdown])

    training_tab.load(fn=refresh_runs, outputs=[runs_list])
