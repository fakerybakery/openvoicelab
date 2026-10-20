from ovl.backends import BackendSpec, all_backends
from ovl.runtime import envs

STATUS_BADGES = {"ready": "✓ installed", "outdated": "↻ update available", "missing": "not installed"}


def status_badge(spec: BackendSpec) -> str:
    return STATUS_BADGES[envs.status(spec)]


def backend_choices():
    return [(f"{spec.name}  ·  {status_badge(spec)}", spec.id) for spec in all_backends()]


def variant_choices(spec: BackendSpec):
    return [(f"{v.label} ({v.size})" if v.size else v.label, v.repo_id) for v in spec.variants]


def capability_summary(caps) -> str:
    """One line listing what a model family can do, from a Capabilities-like mapping."""
    get = caps.get if isinstance(caps, dict) else lambda k, d=None: getattr(caps, k, d)
    parts = []
    if get("clone"):
        parts.append("voice cloning" + (" (needs transcript)" if get("clone_needs_transcript") else ""))
    if get("presets"):
        parts.append("preset voices")
    if get("voice_design"):
        parts.append("voice design")
    if get("instruct"):
        parts.append("style instructions")
    if (get("max_speakers") or 1) > 1:
        parts.append(f"up to {get('max_speakers')} speakers")
    if get("lora"):
        parts.append("LoRA")
    langs = list(get("languages") or [])
    if len(langs) > 1:
        parts.append(f"{len(langs)} languages")
    return " · ".join(parts)


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"
