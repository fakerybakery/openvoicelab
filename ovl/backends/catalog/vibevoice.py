from ovl.backends.spec import BackendSpec, Capabilities, ModelVariant, Param

VIBEVOICE_COMMIT = "952326ddb264062466a888cf32a5b2f4e803e16e"

SPEC = BackendSpec(
    id="vibevoice",
    name="VibeVoice",
    description="Long-form, expressive multi-speaker (up to 4) podcast-style TTS with zero-shot cloning. Fine-tunable with LoRA.",
    module="ovl.backends.impl.vibevoice",
    requirements=[
        # Upstream pins; the fork breaks on transformers 5.
        "transformers==4.51.3",
        "accelerate==1.6.0",
        "datasets==3.5.0",
        "torch>=2.8,<2.15",
        "peft>=0.17,<0.18",
        # Last release before diffusers required huggingface_hub 1.x (transformers 4.51 needs <1.0).
        "diffusers==0.35.2",
        "numba>=0.57",
        "llvmlite>=0.40",
        "librosa>=0.10",
        "scipy",
        "tqdm",
        "ml-collections",
        "absl-py",
        "tensorboard",
    ],
    # Installed without deps so the fork's gradio/aiortc/av demo dependencies stay out.
    no_deps_requirements=[f"vibevoice @ git+https://github.com/vibevoice-community/VibeVoice@{VIBEVOICE_COMMIT}"],
    python="3.12",
    variants=[
        ModelVariant("vibevoice/VibeVoice-1.5B", "VibeVoice 1.5B", "5.4 GB", "Up to 90 min, 4 speakers"),
        ModelVariant("vibevoice/VibeVoice-7B", "VibeVoice 7B", "18.7 GB", "Best quality, needs ~20 GB memory"),
        ModelVariant(
            "microsoft/VibeVoice-Realtime-0.5B", "VibeVoice Realtime 0.5B", "2.0 GB", "Fast, single speaker, preset voices"
        ),
    ],
    capabilities=Capabilities(clone=True, max_speakers=4, no_voice=True, languages=("English", "Chinese"), lora=True, finetune=True),
    params=[
        Param("cfg_scale", "CFG scale", 1.3, minimum=1.0, maximum=3.0, step=0.05, info="Higher follows the voice/text more closely"),
        Param("ddpm_steps", "Diffusion steps", 10, kind="int", minimum=1, maximum=50, step=1, info="Lower = faster, higher = better quality"),
    ],
    license="MIT",
    homepage="https://github.com/vibevoice-community/VibeVoice",
    tips=(
        "Write scripts as 'Speaker 1: ...' lines, one voice per speaker. Clean 10-30 s reference clips work best; "
        "no transcript needed. Without a voice the model picks a random one. Realtime 0.5B is single-speaker with preset voices."
    ),
)
