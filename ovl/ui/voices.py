"""Voices tab: manage the reference clips used for voice cloning."""

import gradio as gr

from ovl import voice_library
from ovl.audio import duration


def _select(name):
    if not name:
        return None, "", ""
    path = voice_library.audio_path(name)
    return str(path), voice_library.transcript(name), f"<small>`{path}` · {duration(str(path)):.1f}s</small>"


def _refresh(selected=None):
    voices = voice_library.list_voices()
    selected = selected if selected in voices else (voices[0] if voices else None)
    return gr.update(choices=voices, value=selected), *_select(selected)


def transcribe(audio_path, model_id):
    if not audio_path:
        raise gr.Error("No audio to transcribe")
    from ovl import asr

    gr.Info("Transcribing with Whisper...")
    text = asr.transcribe(audio_path, model_id)
    asr.unload()
    return text


def add_voice(audio, name, text, auto, model_id):
    if not audio:
        raise gr.Error("Upload or record a clip first")
    if not (name or "").strip():
        raise gr.Error("Give the voice a name")
    if auto and not (text or "").strip():
        text = transcribe(audio, model_id)
    try:
        saved = voice_library.add_voice(name, audio, text)
    except ValueError as e:
        raise gr.Error(str(e))
    gr.Success(f"Added voice '{saved}'")
    return *_refresh(saved), None, "", ""


def save_transcript(name, text):
    if not name:
        raise gr.Error("Select a voice")
    voice_library.set_transcript(name, text)
    gr.Success("Transcript saved")


def delete_voice(name):
    if not name:
        raise gr.Error("Select a voice")
    voice_library.delete_voice(name)
    gr.Info(f"Deleted '{name}'")
    return _refresh()


def build(demo: gr.Blocks):
    from ovl.asr import DEFAULT_MODEL, WHISPER_MODELS

    gr.Markdown(
        "Reference clips for voice cloning. 5–20 seconds of clean speech from one speaker works best. "
        "Some models need a transcript of the clip; it's generated automatically if you leave it empty."
    )
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Your voices")
            voice = gr.Dropdown(label="Voice", choices=voice_library.list_voices())
            preview = gr.Audio(label="Preview", type="filepath", interactive=False)
            info = gr.Markdown()
            transcript = gr.Textbox(label="Transcript", lines=3)
            with gr.Row():
                save_btn = gr.Button("Save transcript")
                retranscribe_btn = gr.Button("Re-transcribe")
                delete_btn = gr.Button("Delete voice", variant="stop")
        with gr.Column():
            gr.Markdown("### Add a voice")
            new_audio = gr.Audio(label="Clip", type="filepath", sources=["upload", "microphone"])
            new_name = gr.Textbox(label="Name", placeholder="e.g. narrator")
            new_text = gr.Textbox(label="Transcript (optional)", lines=3)
            auto = gr.Checkbox(label="Transcribe automatically if empty", value=True)
            whisper = gr.Dropdown(WHISPER_MODELS, value=DEFAULT_MODEL, label="Whisper model")
            add_btn = gr.Button("Add voice", variant="primary")

    voice.change(_select, voice, [preview, transcript, info])
    save_btn.click(save_transcript, [voice, transcript])
    retranscribe_btn.click(transcribe, [preview, whisper], transcript)
    delete_btn.click(delete_voice, voice, [voice, preview, transcript, info])
    add_btn.click(
        add_voice,
        [new_audio, new_name, new_text, auto, whisper],
        [voice, preview, transcript, info, new_audio, new_name, new_text],
    )
    demo.load(lambda v: _refresh(v), voice, [voice, preview, transcript, info])
