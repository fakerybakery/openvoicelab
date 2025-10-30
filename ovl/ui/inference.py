import os
import threading
from datetime import datetime

import gradio as gr
import torch

from ovl.models.vibevoice import VibeVoiceModel
from ovl.models.vibevoice.voices import VoiceMapper

# Global state
model = None
model_loading = False
loaded_lora_path = None
voice_mapper = VoiceMapper()


def get_available_devices():
    """Get list of available devices"""
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")  # CPU always available
    return devices


def get_default_device():
    """Auto-detect best available device: cuda > mps > cpu"""
    devices = get_available_devices()
    return devices[0]  # First device is the best one


def load_model_async(model_path, device, load_lora, lora_path, progress=gr.Progress()):
    """Load the VibeVoice model asynchronously"""
    global model, model_loading, loaded_lora_path

    model_loading = True
    try:
        progress(0, desc="Loading model...")
        checkpoint_path = lora_path if load_lora and lora_path else None
        model = VibeVoiceModel(model_path=model_path, device=device, checkpoint_path=checkpoint_path)
        model_loading = False
        loaded_lora_path = checkpoint_path

        status_msg = f"✓ Model loaded on {device}"
        if checkpoint_path:
            status_msg += f"\n✓ LoRA loaded from {checkpoint_path}"

        return (
            status_msg,
            gr.update(value="Unload Model", variant="stop"),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(value=bool(checkpoint_path)),  # load_lora checkbox
            gr.update(value=checkpoint_path or ""),  # lora_path textbox
            gr.update(visible=bool(checkpoint_path)),  # lora_row visibility
        )
    except Exception as e:
        model_loading = False
        model = None
        loaded_lora_path = None
        return (
            f"✗ Error loading model: {str(e)}",
            gr.update(value="Load Model", variant="primary"),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(value=False),  # load_lora checkbox
            gr.update(value=""),  # lora_path textbox
            gr.update(visible=False),  # lora_row visibility
        )


def unload_model():
    """Unload the current model"""
    global model, loaded_lora_path
    model = None
    loaded_lora_path = None
    return (
        "Model unloaded",
        gr.update(value="Load Model", variant="primary"),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(value=False),  # load_lora checkbox
        gr.update(value=""),  # lora_path textbox
        gr.update(visible=False),  # lora_row visibility
    )


def toggle_model(model_path, device, load_lora, lora_path):
    """Toggle between load and unload"""
    global model, model_loading

    if model_loading:
        return (
            "⏳ Model is currently loading...",
            gr.update(value="Load Model", variant="primary", interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(),  # load_lora checkbox
            gr.update(),  # lora_path textbox
            gr.update(),  # lora_row visibility
        )

    if model is not None:
        return unload_model()
    else:
        return load_model_async(model_path, device, load_lora, lora_path)


def toggle_lora_path(load_lora):
    """Show/hide LoRA path input based on checkbox"""
    return gr.update(visible=load_lora)


def get_initial_state():
    """Get initial UI state based on model status"""
    global model, model_loading, loaded_lora_path

    if model_loading:
        return (
            "⏳ Model is loading...",
            gr.update(value="Load Model", variant="primary", interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(),  # load_lora checkbox
            gr.update(),  # lora_path textbox
            gr.update(),  # lora_row visibility
        )
    elif model is not None:
        status_msg = "✓ Model already loaded"
        if loaded_lora_path:
            status_msg += f"\n✓ LoRA: {loaded_lora_path}"
        return (
            status_msg,
            gr.update(value="Unload Model", variant="stop"),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(value=bool(loaded_lora_path)),  # load_lora checkbox
            gr.update(value=loaded_lora_path or ""),  # lora_path textbox
            gr.update(visible=bool(loaded_lora_path)),  # lora_row visibility
        )
    else:
        return (
            "",
            gr.update(value="Load Model", variant="primary"),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(value=False),  # load_lora checkbox
            gr.update(value=""),  # lora_path textbox
            gr.update(visible=False),  # lora_row visibility
        )


def generate_speech(
    num_speakers, text, voice_1, voice_2, voice_3, voice_4, cfg_scale, enable_voice_cloning, progress=gr.Progress()
):
    """Generate speech from text using gradio_demo.py approach"""
    if model is None:
        return None, "Please load the model first"

    if not text or text.strip() == "":
        return None, "Please enter some text"

    try:
        import re
        import librosa
        import numpy as np
        import soundfile as sf

        # Validate inputs
        if num_speakers < 1 or num_speakers > 4:
            return None, "Error: Number of speakers must be between 1 and 4."

        # Collect selected speakers
        selected_speakers = [voice_1, voice_2, voice_3, voice_4][:num_speakers]

        # Validate speaker selections only when voice cloning is enabled
        if enable_voice_cloning:
            for i, speaker in enumerate(selected_speakers):
                if not speaker:
                    return None, f"Error: Please select a valid voice for Speaker {i+1}."

        # Defend against common mistake
        script = text.replace("'", "'")

        # Parse script to assign speaker IDs (similar to gradio_demo.py lines 277-293)
        lines = script.strip().split("\n")
        formatted_script_lines = []
        speaker_ids_used = set()

        # First pass: collect all speaker IDs used in the script
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line already has speaker format (Speaker 0, Speaker 1, etc.)
            if line.startswith("Speaker ") and ":" in line:
                match = re.match(r"^Speaker\s+(\d+)\s*:", line, re.IGNORECASE)
                if match:
                    speaker_id = int(match.group(1))
                    speaker_ids_used.add(speaker_id)

        # Create a mapping to re-index speakers to start at 0
        if speaker_ids_used:
            sorted_speakers = sorted(speaker_ids_used)
            speaker_id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted_speakers)}

            # Validate that we don't have more unique speakers than configured
            if len(sorted_speakers) > num_speakers:
                raise gr.Error(
                    f"Script uses {len(sorted_speakers)} unique speakers ({', '.join(f'Speaker {s}' for s in sorted_speakers)}) "
                    f"but only {num_speakers} speaker(s) configured. "
                    f"Please increase 'Number of Speakers' to at least {len(sorted_speakers)}."
                )
        else:
            speaker_id_mapping = {}

        # Second pass: reformat with re-indexed speaker IDs
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line already has speaker format (Speaker 0, Speaker 1, etc.)
            if line.startswith("Speaker ") and ":" in line:
                match = re.match(r"^Speaker\s+(\d+)\s*:\s*(.*)$", line, re.IGNORECASE)
                if match:
                    old_speaker_id = int(match.group(1))
                    text_content = match.group(2)
                    new_speaker_id = speaker_id_mapping.get(old_speaker_id, old_speaker_id)
                    formatted_script_lines.append(f"Speaker {new_speaker_id}: {text_content}")
                else:
                    formatted_script_lines.append(line)
            else:
                # Auto-assign to speakers in rotation (0-indexed like gradio_demo.py)
                speaker_id = len(formatted_script_lines) % num_speakers
                formatted_script_lines.append(f"Speaker {speaker_id}: {line}")

        formatted_script = "\n".join(formatted_script_lines)

        # Load voice samples when voice cloning is enabled (similar to gradio_demo.py lines 257-267)
        voice_samples = None
        if enable_voice_cloning:
            voice_samples = []
            for speaker_name in selected_speakers:
                audio_path = voice_mapper.get_voice_path(speaker_name)
                # Read audio with same preprocessing as gradio_demo.py
                wav, sr = sf.read(audio_path)
                if len(wav.shape) > 1:
                    wav = np.mean(wav, axis=1)
                if sr != 24000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=24000)
                voice_samples.append(wav)

        status_speakers = (
            f"Using {num_speakers} speaker(s): {', '.join(selected_speakers)}"
            if enable_voice_cloning
            else "Voice cloning disabled"
        )

        # Generate output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"generated_{timestamp}.wav")

        # Wrapper for Gradio progress.tqdm to ignore unsupported kwargs
        class GradioTqdmWrapper:
            def __call__(self, iterable, desc=None, **kwargs):
                # Only pass supported arguments to Gradio's tqdm
                return progress.tqdm(iterable, desc=desc)

        # Generate using the same approach as gradio_demo.py
        result = model.generate(
            text=formatted_script,
            voice_samples=voice_samples,
            output_path=output_path,
            cfg_scale=cfg_scale,
            enable_voice_cloning=enable_voice_cloning,
            progress_callback=GradioTqdmWrapper(),
        )

        status = f"""✓ Generated successfully
{status_speakers}
Duration: {result.audio_duration:.2f}s
Generation time: {result.generation_time:.2f}s
RTF: {result.rtf:.2f}x"""

        return result.audio_path, status

    except Exception as e:
        import traceback

        return None, f"✗ Error: {str(e)}\n{traceback.format_exc()}"


def toggle_voice_selection(enable_cloning):
    """Show/hide voice selection based on voice cloning checkbox"""
    return gr.update(visible=enable_cloning)


def update_speaker_visibility(num_speakers):
    """Show/hide speaker dropdowns based on number of speakers"""
    updates = []
    for i in range(4):
        updates.append(gr.update(visible=(i < num_speakers)))
    return updates


def refresh_voices():
    """Refresh voice list"""
    voice_mapper.refresh()
    voices = voice_mapper.list_voices()
    return (
        gr.update(choices=voices, value=voices[0] if voices else None),
        gr.update(choices=voices, value=voices[1] if len(voices) > 1 else None),
        gr.update(choices=voices, value=voices[2] if len(voices) > 2 else None),
        gr.update(choices=voices, value=voices[3] if len(voices) > 3 else None),
    )


with gr.Blocks() as inference_tab:
    gr.Markdown("# VibeVoice Inference")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Model Settings")
            model_path = gr.Textbox(
                label="Model Path",
                value="vibevoice/VibeVoice-1.5B",
                placeholder="HuggingFace model ID or local path",
            )
            device = gr.Radio(
                label="Device",
                choices=get_available_devices(),
                value=get_default_device(),
            )
            load_lora = gr.Checkbox(label="Load LoRA Adapter", value=False)
            with gr.Row(visible=False) as lora_row:
                lora_path = gr.Textbox(
                    label="LoRA Path",
                    placeholder="Path to LoRA adapter or HuggingFace repo",
                    scale=1,
                )
            load_btn = gr.Button("Load Model", variant="primary")
            model_status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### Generate Speech")

            # Number of speakers slider
            num_speakers = gr.Slider(
                minimum=1,
                maximum=4,
                value=1,
                step=1,
                label="Number of Speakers",
            )

            text_input = gr.Textbox(
                label="Text",
                placeholder="Enter text to synthesize...\n\nFor multi-speaker:\nSpeaker 0: Hello there!\nSpeaker 1: Hi! How are you?\n\nOr enter plain text and speakers will be auto-assigned.",
                lines=8,
            )

            enable_cloning = gr.Checkbox(label="Enable Voice Cloning", value=True)

            with gr.Column(visible=True) as voice_row:
                gr.Markdown("**Voice Selection** (Speaker 0, Speaker 1, Speaker 2, Speaker 3)")
                with gr.Row():
                    voice_dropdown_1 = gr.Dropdown(
                        label="Speaker 0",
                        choices=voice_mapper.list_voices(),
                        value=(voice_mapper.list_voices()[0] if voice_mapper.list_voices() else None),
                        visible=True,
                    )
                    voice_dropdown_2 = gr.Dropdown(
                        label="Speaker 1",
                        choices=voice_mapper.list_voices(),
                        value=(voice_mapper.list_voices()[1] if len(voice_mapper.list_voices()) > 1 else None),
                        visible=False,
                    )
                with gr.Row():
                    voice_dropdown_3 = gr.Dropdown(
                        label="Speaker 2",
                        choices=voice_mapper.list_voices(),
                        value=(voice_mapper.list_voices()[2] if len(voice_mapper.list_voices()) > 2 else None),
                        visible=False,
                    )
                    voice_dropdown_4 = gr.Dropdown(
                        label="Speaker 3",
                        choices=voice_mapper.list_voices(),
                        value=(voice_mapper.list_voices()[3] if len(voice_mapper.list_voices()) > 3 else None),
                        visible=False,
                    )
                refresh_btn = gr.Button("🔄 Refresh Voices", size="sm")

            cfg_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=2.0, value=1.3, step=0.1)

            generate_btn = gr.Button("Generate Speech", variant="primary", size="lg")

            generation_status = gr.Textbox(label="Status", interactive=False)
            output_audio = gr.Audio(label="Generated Audio", interactive=False)

    # Event handlers
    load_btn.click(
        fn=toggle_model,
        inputs=[model_path, device, load_lora, lora_path],
        outputs=[
            model_status,
            load_btn,
            model_path,
            lora_path,
            load_lora,
            lora_path,
            lora_row,
        ],
    )

    # Update speaker visibility when num_speakers changes
    num_speakers.change(
        fn=update_speaker_visibility,
        inputs=[num_speakers],
        outputs=[voice_dropdown_1, voice_dropdown_2, voice_dropdown_3, voice_dropdown_4],
    )

    generate_btn.click(
        fn=generate_speech,
        inputs=[
            num_speakers,
            text_input,
            voice_dropdown_1,
            voice_dropdown_2,
            voice_dropdown_3,
            voice_dropdown_4,
            cfg_scale,
            enable_cloning,
        ],
        outputs=[output_audio, generation_status],
    )

    refresh_btn.click(
        fn=refresh_voices,
        inputs=[],
        outputs=[voice_dropdown_1, voice_dropdown_2, voice_dropdown_3, voice_dropdown_4],
    )

    enable_cloning.change(fn=toggle_voice_selection, inputs=[enable_cloning], outputs=[voice_row])

    load_lora.change(fn=toggle_lora_path, inputs=[load_lora], outputs=[lora_row])

    # Initialize state on load
    inference_tab.load(
        fn=get_initial_state,
        outputs=[
            model_status,
            load_btn,
            model_path,
            lora_path,
            load_lora,
            lora_path,
            lora_row,
        ],
    )
