from ovl.backends.spec import BackendSpec, Capabilities, ModelVariant, Param

_LANGUAGES = ["English", "French", "German", "Spanish", "Italian", "Portuguese", "Dutch"]


def _variants():
    out = [ModelVariant("english", "English", "110M", "Fast on CPU")]
    out.append(ModelVariant("english_2026-09_24l", "English 24L", "340M", "Higher quality, slower"))
    for lang in _LANGUAGES[1:]:
        out.append(ModelVariant(lang.lower(), lang, "110M"))
        out.append(ModelVariant(f"{lang.lower()}_24l", f"{lang} 24L", "340M"))
    return out


SPEC = BackendSpec(
    id="pocket_tts",
    name="Pocket TTS",
    description="Kyutai's small streaming TTS that runs faster than real time on CPU. Preset voices and zero-shot cloning.",
    module="ovl.backends.impl.pocket_tts",
    # pocket-tts also depends on fastapi/uvicorn/typer for its CLI and web server; the
    # model itself only needs the packages below.
    requirements=[
        "torch>=2.5",
        "einops>=0.4",
        "huggingface_hub>=0.24",
        "pydantic>=2",
        "pyyaml",
        "requests>=2.20",
        "safetensors>=0.4",
        "scipy>=1.5",
        "sentencepiece>=0.2.1",
        "tokenizers>=0.21",
        "typing-extensions>=4.10",
        "numpy>=2",
    ],
    no_deps_requirements=["pocket-tts==3.3.0"],
    python="3.12",
    variants=_variants(),
    capabilities=Capabilities(clone=True, presets=True, max_speakers=4, languages=tuple(_LANGUAGES)),
    params=[
        Param("temperature", "Temperature", 0.3, minimum=0.0, maximum=1.5, step=0.05,
              info="Higher is more expressive but less stable"),
        Param("decode_steps", "Decode steps", 1, kind="int", minimum=1, maximum=8, step=1,
              info="Flow sampler steps per frame. More can improve quality but is slower"),
        Param("eos_threshold", "End-of-speech threshold", -4.0, minimum=-8.0, maximum=0.0, step=0.5,
              info="Higher stops earlier; lower lets the model keep talking"),
    ],
    license="CC-BY-4.0",
    homepage="https://huggingface.co/kyutai/pocket-tts",
    sample_rate=24000,
    supports_mps=False,
    tips=(
        "Each checkpoint is one language. Voice cloning needs the gated kyutai/pocket-tts weights: accept the "
        "terms at huggingface.co/kyutai/pocket-tts and log in with `hf auth login`. Preset voices work without it."
    ),
)
