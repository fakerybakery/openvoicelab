import os

import gradio as gr

from ovl.data import DatasetBuilder

# Global state
dataset_builder = DatasetBuilder()
processing_state = {"active": False, "progress": 0, "status": "", "dataset_name": ""}


def format_duration(seconds):
    """Format duration in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m ({seconds:.0f}s)"
    else:
        hours = seconds / 3600
        total_minutes = seconds / 60
        return f"{hours:.1f}h ({total_minutes:.0f}m)"


def start_processing(input_dir, dataset_name, whisper_model, min_duration, max_duration, progress=gr.Progress()):
    """Start dataset processing"""
    if not input_dir or not os.path.exists(input_dir):
        raise gr.Error("Please provide a valid input directory")

    if not dataset_name or dataset_name.strip() == "":
        raise gr.Error("Please provide a dataset name")

    def progress_callback(prog, status):
        """Update progress"""
        progress(prog, desc=status)

    try:
        # Run processing with progress tracking
        dataset_builder.process_dataset(
            input_dir=input_dir,
            dataset_name=dataset_name.strip(),
            whisper_model=whisper_model,
            progress_callback=progress_callback,
            min_segment_duration=min_duration,
            max_segment_duration=max_duration,
        )

        # Refresh datasets list after completion
        datasets = dataset_builder.list_datasets()
        if not datasets:
            datasets_text = "No datasets created yet"
        else:
            datasets_text = "## Created Datasets\n\n"
            for ds in datasets:
                datasets_text += f"**{ds['name']}**\n"
                datasets_text += f"- Samples: **{ds['num_samples']}**\n"
                duration = ds.get('total_duration', 0)
                datasets_text += f"- Duration: **{format_duration(duration)}**\n"
                datasets_text += f"- Created: {ds['created_at']}\n"
                datasets_text += f"- Location: `data/{ds['name']}/`\n\n"

        return gr.update(value=datasets_text)

    except Exception as e:
        raise gr.Error(f"Error processing dataset: {str(e)}")


def refresh_datasets():
    """Refresh the list of datasets"""
    datasets = dataset_builder.list_datasets()

    if not datasets:
        return gr.update(value="No datasets created yet")

    # Format datasets list with summary
    total_samples = sum(ds.get('num_samples', 0) for ds in datasets)
    total_duration = sum(ds.get('total_duration', 0) for ds in datasets)

    datasets_text = "## Created Datasets\n\n"
    datasets_text += f"**Summary:** {len(datasets)} dataset(s) • {total_samples} total samples • {format_duration(total_duration)} total\n\n"
    datasets_text += "---\n\n"

    for ds in datasets:
        datasets_text += f"**{ds['name']}**\n"
        datasets_text += f"- Samples: **{ds['num_samples']}**\n"
        duration = ds.get('total_duration', 0)
        datasets_text += f"- Duration: **{format_duration(duration)}**\n"
        datasets_text += f"- Created: {ds['created_at']}\n"
        datasets_text += f"- Location: `data/{ds['name']}/`\n\n"

    return gr.update(value=datasets_text)


with gr.Blocks() as data_tab:
    gr.Markdown("# Dataset Builder")
    gr.Markdown("Process audio files into LJSpeech format datasets for training")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")

            input_dir = gr.Textbox(
                label="Input Directory",
                placeholder="/path/to/audio/files",
                info="Directory containing audio files (.wav, .mp3, .flac, .m4a)",
            )

            dataset_name = gr.Textbox(
                label="Dataset Name",
                placeholder="my_dataset",
                info="Name for the output dataset",
            )

            whisper_model = gr.Dropdown(
                label="Whisper Model",
                choices=[
                    "openai/whisper-tiny",
                    "openai/whisper-base",
                    "openai/whisper-small",
                    "openai/whisper-medium",
                    "openai/whisper-large-v3",
                    "openai/whisper-large-v3-turbo",
                ],
                value="openai/whisper-base",
                info="Larger models are more accurate but slower",
            )

            with gr.Accordion("Segmentation Settings", open=False):
                min_duration = gr.Slider(
                    label="Min Segment Duration (seconds)",
                    minimum=3,
                    maximum=15,
                    value=7,
                    step=1,
                    info="Segments shorter than this will be merged together",
                )
                max_duration = gr.Slider(
                    label="Max Segment Duration (seconds)",
                    minimum=10,
                    maximum=30,
                    value=20,
                    step=1,
                    info="Segments longer than this will be split",
                )

            start_btn = gr.Button("Start Processing", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### Datasets")
            datasets_list = gr.Markdown("No datasets created yet")
            refresh_btn = gr.Button("Refresh")

    gr.Markdown(
        """
    ### How it works:
    1. **VAD Segmentation**: Audio files are split into coarse speech segments using Silero VAD
    2. **Whisper Transcription**: Each VAD segment is transcribed using Whisper with timestamp-based segmentation
    3. **Smart Merging**: Short segments (<7s default) are merged together, long segments (>20s default) are split
    4. **LJSpeech Format**: Output is saved in LJSpeech format at `data/{dataset_name}/`
        - `wavs/` - Audio files (7-20s each by default)
        - `metadata.csv` - Transcriptions in format: `filename|text|text`

    **Note**: The new segmentation approach prevents OOM errors during training by keeping segments in the optimal 7-20s range.
    """
    )

    # Event handlers
    start_btn.click(
        fn=start_processing,
        inputs=[input_dir, dataset_name, whisper_model, min_duration, max_duration],
        outputs=[datasets_list],
    )

    # Refresh datasets list
    refresh_btn.click(fn=refresh_datasets, outputs=[datasets_list])

    # Auto-refresh datasets on load
    data_tab.load(fn=refresh_datasets, outputs=[datasets_list])
