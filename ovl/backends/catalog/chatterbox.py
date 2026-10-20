from ovl.backends.spec import BackendSpec, Capabilities, ModelVariant, Param

# PyPI chatterbox-tts 0.1.7 predates Multilingual V3 and Nano, so install from git.
CHATTERBOX_COMMIT = "5de7a54aa4e5e2baadb0182dde554908b48b85c2"
PERTH_COMMIT = "ff1c8ac55a976971245cdd53c18d6131ca00d993"

LANGUAGES = (
    "English",
    "Arabic",
    "Chinese",
    "Danish",
    "Dutch",
    "Finnish",
    "French",
    "German",
    "Greek",
    "Hebrew",
    "Hindi",
    "Italian",
    "Japanese",
    "Korean",
    "Malay",
    "Norwegian",
    "Polish",
    "Portuguese",
    "Russian",
    "Spanish",
    "Swahili",
    "Swedish",
    "Turkish",
)

SPEC = BackendSpec(
    id="chatterbox",
    name="Chatterbox",
    description="Resemble AI's zero-shot cloning TTS: emotion exaggeration control, fast Turbo/Nano with [laugh]-style tags, and a 23-language model.",
    module="ovl.backends.impl.chatterbox",
    requirements=[
        # Upstream pins torch 2.6, but the code only uses torchaudio's Resample and kaldi fbank,
        # which work on 2.11 (the last torchaudio release).
        "torch==2.11.0",
        "torchaudio==2.11.0",
        "transformers==5.2.0",
        "diffusers==0.29.0",
        "safetensors>=0.5.3",
        "librosa==0.11.0",
        "s3tokenizer==0.3.0",
        "conformer==0.3.2",
        "einops",
        "omegaconf",
        "pyloudnorm",
        "spacy-pkuseg",
        "pykakasi==2.3.0",
        f"resemble-perth @ git+https://github.com/resemble-ai/Perth.git@{PERTH_COMMIT}",
    ],
    # Without deps so the gradio demo dependency and the exact torch pin stay out.
    no_deps_requirements=[f"chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox@{CHATTERBOX_COMMIT}"],
    python="3.12",
    variants=[
        ModelVariant("ResembleAI/chatterbox-turbo", "Chatterbox Turbo", "3.0 GB", "English, fast, paralinguistic tags"),
        ModelVariant("ResembleAI/chatterbox:multilingual-v3", "Chatterbox Multilingual V3", "3.2 GB", "23 languages"),
        ModelVariant("ResembleAI/chatterbox", "Chatterbox (English)", "3.2 GB", "Original, exaggeration + CFG control"),
        ModelVariant("ResembleAI/chatterbox-nano", "Chatterbox Nano", "1.9 GB", "English, smallest and fastest"),
        ModelVariant("ResembleAI/chatterbox:multilingual-v2", "Chatterbox Multilingual V2", "3.2 GB", "Legacy 23-language model"),
    ],
    capabilities=Capabilities(clone=True, no_voice=True, max_speakers=4, languages=LANGUAGES),
    params=[
        Param(
            "exaggeration",
            "Exaggeration",
            0.5,
            minimum=0.25,
            maximum=2.0,
            step=0.05,
            info="Emotion intensity; 0.5 is neutral. Not used by Turbo/Nano",
        ),
        Param(
            "cfg_weight",
            "CFG weight",
            0.5,
            minimum=0.0,
            maximum=1.0,
            step=0.05,
            info="Lower (~0.3) slows pacing; 0 avoids accent transfer across languages. Not used by Turbo/Nano",
        ),
        Param("temperature", "Temperature", 0.8, minimum=0.05, maximum=2.0, step=0.05),
        Param("repetition_penalty", "Repetition penalty", 1.2, minimum=1.0, maximum=2.0, step=0.05),
    ],
    license="MIT",
    homepage="https://github.com/resemble-ai/chatterbox",
    sample_rate=24000,
    tips=(
        "Use a clean 5-10 s reference clip (Turbo/Nano need at least 5 s). Turbo/Nano understand tags like "
        "[laugh], [chuckle], [cough], [sigh]. For dramatic speech try exaggeration ~0.7 with CFG ~0.3. "
        "Output is watermarked (Resemble Perth)."
    ),
)
