from ovl.backends.spec import BackendSpec, Capabilities, ModelVariant, Param

SPEC = BackendSpec(
    id="echo_tts",
    name="Echo-TTS",
    description="Expressive 44.1 kHz diffusion TTS with strong zero-shot cloning from a short clip (no transcript needed).",
    module="ovl.backends.impl.echo_tts",
    # Upstream code is flat modules (no package); the worker fetches a commit-pinned copy at load time.
    # torchcodec is skipped on purpose (only used by upstream's file loader, needs system FFmpeg).
    requirements=[
        "torch>=2.9.1,<2.11",
        "torchaudio>=2.9.1,<2.11",
        "safetensors>=0.4",
        "einops>=0.8",
        "huggingface-hub>=0.34",
    ],
    python="3.12",
    variants=[ModelVariant("jordand/echo-tts-base", "Echo-TTS Base", "2.4B + 0.5B codec")],
    capabilities=Capabilities(clone=True, no_voice=True, languages=("English",)),
    params=[
        Param("num_steps", "Steps", 40, kind="int", minimum=10, maximum=80, step=1, info="Diffusion steps"),
        Param("cfg_scale_text", "Text guidance", 3.0, minimum=0.0, maximum=10.0, step=0.5),
        Param("cfg_scale_speaker", "Speaker guidance", 8.0, minimum=0.0, maximum=12.0, step=0.5,
              info="How closely to follow the reference voice"),
        Param("truncation", "Truncation", 1.0, minimum=0.5, maximum=1.0, step=0.05,
              info="Initial noise scale. 0.8 is flatter but more stable"),
        Param("rescale_k", "Temporal rescale k", 1.0, minimum=0.8, maximum=1.5, step=0.05,
              info="1 = off, >1 flattens, <1 sharpens (try 1.2 with truncation 0.8)"),
        Param("force_speaker", "Force speaker", 1.0, minimum=1.0, maximum=2.0, step=0.1,
              info="Speaker KV scale. 1 = off; try 1.2-1.5 if the voice drifts from the reference"),
        Param("chunk_chars", "Max characters per chunk", 400, kind="int", minimum=100, maximum=700, step=50,
              info="Long text is split at sentences into chunks of ~30 s max and joined"),
    ],
    license="CC-BY-NC-SA-4.0 (non-commercial; outputs too)",
    homepage="https://github.com/jordandare/echo-tts",
    sample_rate=44100,
    tips="English only. Reference clips up to 5 min are used, but ~10 s works well. Without a reference you get a "
    "random voice. Commas act as pauses; [S1]/[S2] tags and (laughs) style cues follow the WhisperD format.",
)
