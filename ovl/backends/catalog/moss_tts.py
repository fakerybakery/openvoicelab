from ovl.backends.spec import BackendSpec, Capabilities, ModelVariant, Param

SPEC = BackendSpec(
    id="moss_tts",
    name="MOSS-TTS",
    description="OpenMOSS TTS family: stable zero-shot cloning in 31 languages (no transcript needed), "
    "multi-speaker dialogue (TTSD) and voice design from a description (VoiceGenerator).",
    module="ovl.backends.impl.moss_tts",
    # Upstream pins (pyproject torch-runtime extra). The remote model code imports
    # transformers.initialization, which only exists in transformers 5.
    requirements=[
        "torch==2.9.1",
        "torchaudio==2.9.1",
        "transformers==5.0.0",
        "accelerate>=1.10.1",
        "safetensors",
        "sentencepiece",
        "tqdm",
    ],
    python="3.12",
    variants=[
        ModelVariant(
            "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5", "TTS Local v1.5 (4B, 48 kHz stereo)", "17.6 GB",
            "Cloning + plain TTS, 31 languages",
        ),
        ModelVariant("OpenMOSS-Team/MOSS-TTS-v1.5", "TTS v1.5 (8B, flagship)", "24 GB", "Best cloning quality, 31 languages"),
        ModelVariant(
            "OpenMOSS-Team/MOSS-TTSD-v1.0", "TTSD v1.0 (8B, dialogue)", "24 GB", "Up to 5 speakers; clips need transcripts"
        ),
        ModelVariant(
            "OpenMOSS-Team/MOSS-VoiceGenerator", "VoiceGenerator (1.7B, voice design)", "11 GB", "Describe the voice"
        ),
        ModelVariant("OpenMOSS-Team/MOSS-TTS-Nano", "TTS Nano (0.1B, 48 kHz)", "0.3 GB", "Tiny and fast, cloning"),
    ],
    # Defaults describe the TTS checkpoints; load() overrides them for TTSD / VoiceGenerator / Nano.
    capabilities=Capabilities(
        clone=True,
        no_voice=True,
        languages=(
            "Auto",
            "English",
            "Chinese",
            "Cantonese",
            "Arabic",
            "Czech",
            "Danish",
            "Dutch",
            "Finnish",
            "French",
            "German",
            "Greek",
            "Hebrew",
            "Hindi",
            "Hungarian",
            "Italian",
            "Japanese",
            "Korean",
            "Macedonian",
            "Malay",
            "Persian (Farsi)",
            "Polish",
            "Portuguese",
            "Romanian",
            "Russian",
            "Spanish",
            "Swahili",
            "Swedish",
            "Tagalog",
            "Thai",
            "Turkish",
            "Vietnamese",
        ),
    ),
    params=[
        Param(
            "temperature", "Temperature", 0.0, minimum=0.0, maximum=2.5, step=0.05,
            info="Audio sampling temperature. 0 = the checkpoint's recommended value",
        ),
        Param("top_p", "Top-p", 0.0, minimum=0.0, maximum=1.0, step=0.05, info="0 = recommended"),
        Param("top_k", "Top-k", 0, kind="int", minimum=0, maximum=200, step=1, info="0 = recommended"),
        Param(
            "repetition_penalty", "Repetition penalty", 0.0, minimum=0.0, maximum=1.5, step=0.01, info="0 = recommended"
        ),
        Param(
            "use_transcript", "Continue from the clip", False, kind="bool",
            info="Use the reference transcript to continue the clip (closer match, less stable). Plain TTS and Nano only",
        ),
        Param(
            "max_seconds", "Max length (seconds)", 120, kind="int", minimum=5, maximum=600, step=5,
            info="Upper bound on generated audio length",
        ),
    ],
    license="Apache-2.0",
    homepage="https://github.com/OpenMOSS/MOSS-TTS",
    sample_rate=24000,
    tips="Cloning only needs a clip (no transcript), except TTSD, which needs each clip's transcript. "
    "Set the language for best results. TTSD: write 'Speaker 1: ...' lines; speakers without a clip get a "
    "random voice. VoiceGenerator: describe the voice (e.g. 'warm, slow elderly male narrator'). "
    "Insert pauses with [pause 1.5s] (v1.5).",
)
