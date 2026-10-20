from ovl.backends.spec import BackendSpec, Capabilities, ModelVariant, Param

SPEC = BackendSpec(
    id="qwen3_tts",
    name="Qwen3-TTS",
    description="Expressive multilingual TTS from Qwen: 9 premium preset voices with style instructions, "
    "voice design from a text description, or 3-second voice cloning (depends on the checkpoint).",
    module="ovl.backends.impl.qwen3_tts",
    # qwen-tts hard-pins transformers/accelerate (its model code relies on 4.57 internals) and pulls in
    # gradio for its demo, so install it without deps and list the real runtime deps here.
    requirements=[
        "torch",
        "torchaudio",
        "transformers==4.57.3",
        "accelerate==1.12.0",
        "librosa",
        "soundfile",
        "sox",
        "onnxruntime",
        "einops",
    ],
    no_deps_requirements=["qwen-tts==0.1.1"],
    python="3.12",
    variants=[
        ModelVariant("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "0.6B CustomVoice (preset voices)", "0.6B"),
        ModelVariant("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "1.7B CustomVoice (preset voices + instruct)", "1.7B"),
        ModelVariant("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "1.7B VoiceDesign (describe a voice)", "1.7B"),
        ModelVariant("Qwen/Qwen3-TTS-12Hz-0.6B-Base", "0.6B Base (voice cloning)", "0.6B"),
        ModelVariant("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "1.7B Base (voice cloning)", "1.7B"),
    ],
    # Defaults describe the CustomVoice checkpoints; load() overrides them per checkpoint type.
    capabilities=Capabilities(
        presets=True,
        instruct=True,
        max_speakers=4,
        languages=(
            "Auto",
            "Chinese",
            "English",
            "Japanese",
            "Korean",
            "German",
            "French",
            "Russian",
            "Portuguese",
            "Spanish",
            "Italian",
        ),
    ),
    params=[
        Param("temperature", "Temperature", 0.9, minimum=0.1, maximum=1.5, step=0.05),
        Param("top_p", "Top-p", 1.0, minimum=0.1, maximum=1.0, step=0.05),
        Param("top_k", "Top-k", 50, kind="int", minimum=1, maximum=200, step=1),
        Param("repetition_penalty", "Repetition penalty", 1.05, minimum=1.0, maximum=1.5, step=0.01),
        Param(
            "subtalker_temperature",
            "Acoustic temperature",
            0.9,
            minimum=0.1,
            maximum=1.5,
            step=0.05,
            info="Sampling temperature of the sub-talker that predicts the fine acoustic codes",
        ),
        Param(
            "max_new_tokens",
            "Max length (tokens)",
            4096,
            kind="int",
            minimum=256,
            maximum=8192,
            step=256,
            info="12 tokens = 1 second of audio",
        ),
    ],
    license="Apache-2.0",
    homepage="https://github.com/QwenLM/Qwen3-TTS",
    sample_rate=24000,
    tips="CustomVoice: pick a preset (each speaks all languages; Dylan/Eric speak Beijing/Sichuan dialect in "
    "Chinese) and, on 1.7B, add an instruction like 'Very happy'. VoiceDesign: describe the voice. Base: "
    "upload a clip plus its transcript (without a transcript only the speaker timbre is used).",
)
