"""Data tab: build training datasets from raw recordings."""

import gradio as gr

from ovl.data import DatasetBuilder
from ovl.ui.common import format_duration

dataset_builder = DatasetBuilder()


def datasets_markdown() -> str:
    datasets = dataset_builder.list_datasets()
    if not datasets:
        return "No datasets yet."
    total = sum(ds.get("total_duration", 0) for ds in datasets)
    lines = [f"**{len(datasets)} dataset(s)** · {format_duration(total)} of audio", ""]
    for ds in datasets:
        lines.append(
            f"- **{ds['name']}**: {ds['num_samples']} clips, {format_duration(ds.get('total_duration', 0))} "
            f"· {ds['created_at'][:16].replace('T', ' ')} · `{ds['path']}`"
        )
    return "\n".join(lines)


def start_processing(input_dir, dataset_name, whisper_model, language, min_duration, max_duration,
                     progress=gr.Progress()):
    if not (input_dir or "").strip():
        raise gr.Error("Enter the folder containing your recordings")
    if not (dataset_name or "").strip():
        raise gr.Error("Give the dataset a name")
    if min_duration >= max_duration:
        raise gr.Error("Min segment duration must be below the max")
    try:
        info = dataset_builder.process_dataset(
            input_dir=input_dir.strip(),
            dataset_name=dataset_name.strip(),
            whisper_model=whisper_model,
            language=language or None,
            progress_callback=lambda p, desc: progress(p, desc=desc),
            min_segment_duration=min_duration,
            max_segment_duration=max_duration,
        )
    except ValueError as e:
        raise gr.Error(str(e))
    gr.Success(f"Created '{info['name']}' with {info['num_samples']} clips")
    return datasets_markdown()


def build(demo: gr.Blocks):
    from ovl.asr import DEFAULT_MODEL, WHISPER_MODELS

    gr.Markdown(
        "Turn long recordings of one speaker into a training dataset: speech is detected with Silero VAD, "
        "transcribed with Whisper, and cut into clips of a few seconds (LJSpeech format: `wavs/` + `metadata.csv`)."
    )
    with gr.Row():
        with gr.Column():
            input_dir = gr.Textbox(label="Recordings folder", placeholder="/path/to/recordings",
                                   info="Searched recursively for wav, mp3, flac, m4a, ogg and opus files")
            dataset_name = gr.Textbox(label="Dataset name", placeholder="my_voice")
            whisper_model = gr.Dropdown(WHISPER_MODELS, value=DEFAULT_MODEL, label="Whisper model",
                                        info="large-v3-turbo is fast and accurate; smaller ones use less memory")
            language = gr.Textbox(label="Language (optional)", placeholder="e.g. english; auto-detected if empty")
            with gr.Accordion("Segmentation", open=False):
                min_duration = gr.Slider(1, 15, value=7, step=1, label="Min clip length (s)",
                                         info="Shorter sentences are merged together")
                max_duration = gr.Slider(5, 30, value=20, step=1, label="Max clip length (s)",
                                         info="Longer ones are split. Long clips use more memory when training")
            start_btn = gr.Button("Build dataset", variant="primary", size="lg")
        with gr.Column():
            gr.Markdown("### Datasets")
            datasets = gr.Markdown()
            refresh_btn = gr.Button("Refresh", size="sm")

    start_btn.click(start_processing, [input_dir, dataset_name, whisper_model, language, min_duration, max_duration],
                    datasets, concurrency_id="gpu")
    refresh_btn.click(datasets_markdown, None, datasets)
    demo.load(datasets_markdown, None, datasets)
