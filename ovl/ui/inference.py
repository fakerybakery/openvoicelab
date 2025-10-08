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


def generate_speech(text, voice_name, cfg_scale, enable_voice_cloning, progress=gr.Progress()):
    """Generate speech from text"""
    if model is None:
        return None, "Please load the model first"

    if not text or text.strip() == "":
        return None, "Please enter some text"

    if enable_voice_cloning and not voice_name:
        return None, "Please select a voice"

    try:
        # Get voice path only if voice cloning is enabled
        voice_samples = None
        if enable_voice_cloning:
            voice_path = voice_mapper.get_voice_path(voice_name)
            voice_samples = [voice_path]

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

        # Generate
        result = model.generate(
            text=text,
            voice_samples=voice_samples,
            output_path=output_path,
            cfg_scale=cfg_scale,
            enable_voice_cloning=enable_voice_cloning,
            progress_callback=GradioTqdmWrapper(),
        )

        status = f"""✓ Generated successfully
Duration: {result.audio_duration:.2f}s
Generation time: {result.generation_time:.2f}s
RTF: {result.rtf:.2f}x"""

        return result.audio_path, status

    except Exception as e:
        return None, f"✗ Error: {str(e)}"


def toggle_voice_selection(enable_cloning):
    """Show/hide voice selection based on voice cloning checkbox"""
    return gr.update(visible=enable_cloning)


def refresh_voices():
    """Refresh voice list"""
    voice_mapper.refresh()
    voices = voice_mapper.list_voices()
    return gr.update(choices=voices, value=voices[0] if voices else None)


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
            text_input = gr.Textbox(label="Text", placeholder="Enter text to synthesize...", lines=5)

            enable_cloning = gr.Checkbox(label="Enable Voice Cloning", value=True)

            with gr.Row(visible=True) as voice_row:
                voice_dropdown = gr.Dropdown(
                    label="Voice",
                    choices=voice_mapper.list_voices(),
                    value=(voice_mapper.list_voices()[0] if voice_mapper.list_voices() else None),
                )
                refresh_btn = gr.Button("🔄", scale=0)

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

    generate_btn.click(
        fn=generate_speech,
        inputs=[text_input, voice_dropdown, cfg_scale, enable_cloning],
        outputs=[output_audio, generation_status],
    )

    refresh_btn.click(fn=refresh_voices, inputs=[], outputs=[voice_dropdown])

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
