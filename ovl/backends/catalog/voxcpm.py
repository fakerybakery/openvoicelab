from ovl.backends.spec import BackendSpec, Capabilities, ModelVariant, Param

SPEC = BackendSpec(
    id="voxcpm",
    name="VoxCPM",
    description="Tokenizer-free 48 kHz TTS (VoxCPM2): 30 languages, voice design from a description, and expressive cloning.",
    module="ovl.backends.impl.voxcpm",
    # The voxcpm package hard-depends on gradio, modelscope, funasr, datasets, etc. for its
    # web demo, denoiser and training code; install it without those and list what inference needs.
    requirements=[
        "torch>=2.5.0",
        "torchaudio>=2.5.0",
        "transformers>=4.36.2,<5",
        "einops",
        "pydantic",
        "tqdm",
        "librosa",
        "safetensors",
        "huggingface-hub",
        "inflect",
        "wetext",
        "regex",
    ],
    no_deps_requirements=["voxcpm==2.0.3"],
    python="3.12",
    variants=[
        ModelVariant("openbmb/VoxCPM2", "VoxCPM2", "2B", "48 kHz, 30 languages, voice design"),
        ModelVariant("openbmb/VoxCPM1.5", "VoxCPM 1.5", "0.8B", "44.1 kHz, English/Chinese, needs a transcript"),
        ModelVariant("openbmb/VoxCPM-0.5B", "VoxCPM 0.5B", "0.6B", "16 kHz, English/Chinese, needs a transcript"),
    ],
    capabilities=Capabilities(
        clone=True,
        voice_design=True,
        instruct=True,
        no_voice=True,
        languages=(
            "Arabic", "Burmese", "Chinese", "Danish", "Dutch", "English", "Finnish", "French", "German",
            "Greek", "Hebrew", "Hindi", "Indonesian", "Italian", "Japanese", "Khmer", "Korean", "Lao",
            "Malay", "Norwegian", "Polish", "Portuguese", "Russian", "Spanish", "Swahili", "Swedish",
            "Tagalog", "Thai", "Turkish", "Vietnamese",
        ),
    ),
    params=[
        Param("cfg_value", "Guidance (CFG)", 2.0, minimum=1.0, maximum=5.0, step=0.1,
              info="Higher follows the prompt/voice more closely; too high sounds strained."),
        Param("inference_timesteps", "Diffusion steps", 10, kind="int", minimum=4, maximum=50, step=1,
              info="More steps = slightly better quality, slower."),
        Param("normalize", "Normalize text", False, kind="bool",
              info="Spell out numbers, dates and symbols before synthesis."),
        Param("retry_badcase", "Retry bad generations", True, kind="bool",
              info="Regenerate when the output length looks wrong for the text (up to 3 times)."),
    ],
    license="Apache-2.0",
    homepage="https://github.com/OpenBMB/VoxCPM",
    sample_rate=48000,
    tips=(
        "No language setting is needed: just type text in any supported language. "
        "VoxCPM2 clones from the reference clip alone; adding its transcript enables higher-fidelity "
        "continuation cloning. Voice design and style instructions are VoxCPM2-only and can vary "
        "between runs, so try a few seeds. Runs in float32 on Apple Silicon (half precision glitches)."
    ),
)
