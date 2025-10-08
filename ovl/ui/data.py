import os

import gradio as gr

from ovl.data import DatasetBuilder

# Global state
dataset_builder = DatasetBuilder()
processing_state = {"active": False, "progress": 0, "status": "", "dataset_name": ""}


def start_processing(input_dir, dataset_name, whisper_model, progress=gr.Progress()):
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
        )

        # Refresh datasets list after completion
        datasets = dataset_builder.list_datasets()
        if not datasets:
            datasets_text = "No datasets created yet"
        else:
            datasets_text = "## Created Datasets\n\n"
            for ds in datasets:
                datasets_text += f"**{ds['name']}**\n"
                datasets_text += f"- Samples: {ds['num_samples']}\n"
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

    # Format datasets list
    datasets_text = "## Created Datasets\n\n"
    for ds in datasets:
        datasets_text += f"**{ds['name']}**\n"
        datasets_text += f"- Samples: {ds['num_samples']}\n"
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

            start_btn = gr.Button("Start Processing", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### Datasets")
            datasets_list = gr.Markdown("No datasets created yet")
            refresh_btn = gr.Button("Refresh")

    gr.Markdown(
        """
    ### How it works:
    1. **VAD Segmentation**: Audio files are split into speech segments using Silero VAD
    2. **Transcription**: Each segment is transcribed using Whisper
    3. **LJSpeech Format**: Output is saved in LJSpeech format at `data/{dataset_name}/`
        - `wavs/` - Audio files
        - `metadata.csv` - Transcriptions in format: `filename|text|text`
    """
    )

    # Event handlers
    start_btn.click(
        fn=start_processing,
        inputs=[input_dir, dataset_name, whisper_model],
        outputs=[datasets_list],
    )

    # Refresh datasets list
    refresh_btn.click(fn=refresh_datasets, outputs=[datasets_list])

    # Auto-refresh datasets on load
    data_tab.load(fn=refresh_datasets, outputs=[datasets_list])
