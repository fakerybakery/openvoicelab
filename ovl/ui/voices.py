import os

import gradio as gr

from ovl.models.vibevoice.voices import VoiceMapper

# Initialize voice mapper
voice_mapper = VoiceMapper()


def refresh_voices():
    """Refresh the voice list"""
    voice_mapper.refresh()
    voices = voice_mapper.list_voices()
    first_voice = voices[0] if voices else None
    preview = preview_voice(first_voice) if first_voice else None
    return gr.update(choices=voices, value=first_voice), preview


def upload_voice(audio_file, voice_name):
    """Upload a new voice"""
    if not audio_file:
        raise gr.Error("Please upload an audio file")

    if not voice_name or voice_name.strip() == "":
        raise gr.Error("Please provide a voice name")

    try:
        voice_mapper.add_voice(voice_name.strip(), audio_file)
        gr.Success(f"Successfully added voice: {voice_name}")
        return refresh_voices()
    except Exception as e:
        raise gr.Error(f"Error adding voice: {str(e)}")


def delete_voice(voice_name):
    """Delete a voice"""
    if not voice_name:
        raise gr.Error("Please select a voice to delete")

    try:
        voice_mapper.delete_voice(voice_name)
        gr.Success(f"Successfully deleted voice: {voice_name}")
        return refresh_voices()
    except Exception as e:
        raise gr.Error(f"Error deleting voice: {str(e)}")


def preview_voice(voice_name):
    """Preview a voice"""
    if not voice_name:
        return None

    try:
        voice_path = voice_mapper.get_voice_path(voice_name)
        return voice_path
    except Exception as e:
        print(f"Error previewing voice: {e}")
        return None


with gr.Blocks() as voices_tab:
    gr.Markdown("# Voices")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Upload New Voice")
            voice_name_input = gr.Textbox(label="Voice Name", placeholder="e.g., john_doe")
            audio_upload = gr.Audio(label="Upload Audio File", type="filepath")
            upload_btn = gr.Button("Upload Voice", variant="primary")

        with gr.Column():
            gr.Markdown("### Manage Voices")
            voice_dropdown = gr.Dropdown(
                label="Select Voice",
                choices=voice_mapper.list_voices(),
                value=(voice_mapper.list_voices()[0] if voice_mapper.list_voices() else None),
            )
            refresh_btn = gr.Button("Refresh List")
            preview_audio = gr.Audio(label="Preview", interactive=False)
            delete_btn = gr.Button("Delete Selected Voice", variant="stop")

    # Event handlers
    upload_btn.click(
        fn=upload_voice,
        inputs=[audio_upload, voice_name_input],
        outputs=[voice_dropdown, preview_audio],
    )

    refresh_btn.click(fn=refresh_voices, inputs=[], outputs=[voice_dropdown, preview_audio])

    voice_dropdown.change(fn=preview_voice, inputs=[voice_dropdown], outputs=[preview_audio])

    delete_btn.click(
        fn=delete_voice,
        inputs=[voice_dropdown],
        outputs=[voice_dropdown, preview_audio],
    )

    # Load first voice on page load
    voices_tab.load(
        fn=lambda: (preview_voice(voice_mapper.list_voices()[0]) if voice_mapper.list_voices() else None),
        outputs=[preview_audio],
    )
