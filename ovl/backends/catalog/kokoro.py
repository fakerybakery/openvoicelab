from ovl.backends.spec import BackendSpec, Capabilities, ModelVariant, Param

SPEC = BackendSpec(
    id="kokoro",
    name="Kokoro",
    description="Tiny (82M), fast preset-voice TTS. Runs comfortably on CPU. No voice cloning.",
    module="ovl.backends.impl.kokoro",
    requirements=["kokoro>=0.9.4", "misaki[en]>=0.9.4", "torch"],
    python="3.12",
    variants=[ModelVariant("hexgrad/Kokoro-82M", "Kokoro 82M", "82M")],
    capabilities=Capabilities(
        presets=True,
        languages=(
            "English (US)",
            "English (UK)",
            "Spanish",
            "French",
            "Hindi",
            "Italian",
            "Portuguese (BR)",
        ),
    ),
    params=[
        Param("speed", "Speed", 1.0, minimum=0.5, maximum=2.0, step=0.05),
    ],
    license="Apache-2.0",
    homepage="https://huggingface.co/hexgrad/Kokoro-82M",
    tips="Voice names start with a language/gender code: af_ = American female, bm_ = British male, ff_ = French female, ...",
)
