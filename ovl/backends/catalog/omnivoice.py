from ovl.backends.spec import BackendSpec, Capabilities, ModelVariant, Param

SPEC = BackendSpec(
    id="omnivoice",
    name="OmniVoice",
    description="Fast diffusion-LM TTS for 600+ languages: zero-shot cloning and attribute-based voice design.",
    module="ovl.backends.impl.omnivoice",
    # omnivoice also depends on gradio/tensorboardX/webdataset for its demo and training code.
    requirements=[
        "torch>=2.4",
        "torchaudio>=2.4",
        "transformers>=5.3.0,<6",
        "accelerate",
        "pydub",
        "librosa",
    ],
    no_deps_requirements=["omnivoice==0.2.1"],
    python="3.12",
    variants=[
        ModelVariant("k2-fsa/OmniVoice", "OmniVoice", "0.6B + 0.2B codec"),
    ],
    capabilities=Capabilities(
        clone=True,
        voice_design=True,
        no_voice=True,
        # The full 600+ list comes from the model once it's loaded.
        languages=(
            "Auto", "English", "Chinese", "Japanese", "Korean", "German", "French", "Spanish", "Portuguese",
            "Italian", "Russian", "Arabic", "Hindi",
        ),
    ),
    params=[
        Param("num_step", "Diffusion steps", 32, kind="int", minimum=4, maximum=64, step=1,
              info="16 is about twice as fast with a small quality cost."),
        Param("guidance_scale", "Guidance (CFG)", 2.0, minimum=0.0, maximum=5.0, step=0.1),
        Param("speed", "Speed", 1.0, minimum=0.5, maximum=1.5, step=0.05,
              info="Ignored when a duration is set."),
        Param("duration", "Duration (s)", 0.0, minimum=0.0, maximum=60.0, step=0.5,
              info="Force the output length. 0 = automatic."),
        Param("denoise", "Denoise", True, kind="bool", info="Ask the model for clean audio even if the reference is noisy."),
        Param("postprocess_output", "Trim silences", True, kind="bool"),
    ],
    license="CC-BY-NC-4.0 (non-commercial; codec under Boson Higgs Audio 2 Community License)",
    homepage="https://github.com/k2-fsa/OmniVoice",
    sample_rate=24000,
    tips=(
        "Use a 3-10 s reference clip in the same language as the text. Without a transcript the clip is "
        "transcribed automatically with Whisper large-v3-turbo. Voice design accepts only comma-separated "
        "attributes: male/female; child, teenager, young adult, middle-aged, elderly; very low ... very high "
        "pitch; whisper; american/british/australian/indian/... accent (e.g. \"female, low pitch, british accent\"). "
        "Inline tags like [laughter] and [sigh] are supported."
    ),
)
