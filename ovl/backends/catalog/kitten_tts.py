from ovl.backends.spec import BackendSpec, Capabilities, ModelVariant, Param

SPEC = BackendSpec(
    id="kitten_tts",
    name="Kitten TTS",
    description="Ultra-light (15-80M) ONNX TTS with 8 English preset voices. Very fast on CPU, no GPU or PyTorch needed.",
    module="ovl.backends.impl.kitten_tts",
    # kittenml also ships KittenTTS 2 and depends on torch/transformers/diffusers/vllm;
    # the ONNX models only need the packages below.
    requirements=[
        "onnxruntime>=1.20",
        "phonemizer>=3.3,<4",
        "espeakng-loader>=0.2.4",
        "huggingface_hub>=0.24",
    ],
    no_deps_requirements=["kittenml==0.9.4"],
    python="3.12",
    variants=[
        ModelVariant("KittenML/kitten-tts-mini-0.8", "Mini 0.8", "80M", "Best quality"),
        ModelVariant("KittenML/kitten-tts-micro-0.8", "Micro 0.8", "40M"),
        ModelVariant("KittenML/kitten-tts-nano-0.8-fp32", "Nano 0.8", "15M"),
        ModelVariant("KittenML/kitten-tts-nano-0.8-int8", "Nano 0.8 (int8)", "15M", "25 MB download"),
    ],
    capabilities=Capabilities(presets=True, languages=("English",)),
    params=[
        Param("speed", "Speed", 1.0, minimum=0.5, maximum=2.0, step=0.05),
        Param("clean_text", "Normalize text", True, kind="bool", info="Expand numbers, currencies, dates and units"),
    ],
    license="Apache-2.0",
    homepage="https://github.com/KittenML/KittenTTS",
    sample_rate=24000,
    tips="Runs on CPU (ONNX), so the device setting is ignored. Female voices: Bella, Luna, Rosie, Kiki. Male: Jasper, Bruno, Hugo, Leo.",
)
