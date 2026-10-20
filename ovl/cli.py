import json
import sys

import click


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """OpenVoiceLab: run and finetune open TTS models.

    Run without a command to launch the web UI.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(ui)


@main.command()
@click.option("--host", default="127.0.0.1", help="Host to run the server on")
@click.option("--port", default=7860, help="Port to run the server on")
@click.option("--share", is_flag=True, help="Create a public Gradio link")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def ui(host, port, share, debug):
    """Launch the web UI."""
    from ovl.app import launch

    launch(server_name=host, server_port=port, share=share, debug=debug)


@main.command("models")
def list_models():
    """List supported model backends and whether they're installed."""
    from ovl.backends import all_backends
    from ovl.runtime import envs

    for spec in all_backends():
        click.echo(f"{spec.id:<12} {envs.status(spec):<9} {spec.name}  [{spec.license}]")
        for v in spec.variants:
            click.echo(f"{'':<23}- {v.repo_id}")


@main.command()
@click.argument("backend_ids", nargs=-1, required=True)
@click.option("--reinstall", is_flag=True, help="Delete and recreate the environment")
def install(backend_ids, reinstall):
    """Install the environment for one or more backends (or 'all')."""
    from ovl.backends import all_backends, get
    from ovl.runtime import envs

    specs = all_backends() if backend_ids == ("all",) else [get(b) for b in backend_ids]
    failed = []
    for spec in specs:
        click.secho(f"\n=== Installing {spec.name} ===", bold=True)
        try:
            envs.install(spec, log=click.echo, reinstall=reinstall)
        except Exception as e:
            click.secho(f"✗ {spec.name}: {e}", fg="red")
            failed.append(spec.id)
    if failed:
        sys.exit(f"Failed: {', '.join(failed)}")


@main.command()
@click.argument("backend_id")
@click.argument("text")
@click.option("--model", "model_id", help="HF repo id or local path (defaults to the backend's first variant)")
@click.option("--voice", "presets", multiple=True, help="Preset voice name (repeat for multiple speakers)")
@click.option("--ref", "refs", multiple=True, type=click.Path(exists=True), help="Reference audio for cloning (repeatable)")
@click.option("--ref-text", "ref_texts", multiple=True, help="Transcript of each --ref clip")
@click.option("--design", help="Voice description for voice-design models")
@click.option("--instruct", help="Style / emotion instruction")
@click.option("--language")
@click.option("--seed", type=int)
@click.option("--param", "params", multiple=True, help="Backend parameter as key=value (value parsed as JSON)")
@click.option("--lora", help="LoRA adapter path (VibeVoice)")
@click.option("--device", default="auto")
@click.option("-o", "--output", help="Output WAV path")
def tts(backend_id, text, model_id, presets, refs, ref_texts, design, instruct, language, seed, params, lora, device, output):
    """Synthesize TEXT with a backend, e.g.

    openvoicelab tts kokoro "Hello there" --voice af_heart -o hello.wav
    """
    from ovl.backends import get
    from ovl.runtime.session import session

    spec = get(backend_id)
    voices = [{"preset": p} for p in presets]
    for i, ref in enumerate(refs):
        voices.append({"audio": ref, "transcript": ref_texts[i] if i < len(ref_texts) else None})
    if design:
        voices = voices or [{}]
        voices[0]["description"] = design

    parsed = dict(spec.default_params())
    for p in params:
        key, _, value = p.partition("=")
        try:
            parsed[key] = json.loads(value)
        except json.JSONDecodeError:
            parsed[key] = value

    def progress(value, desc):
        if desc:
            click.echo(f"  {desc}" + (f" ({value:.0%})" if isinstance(value, (int, float)) else ""), err=True)

    from ovl.runtime.client import WorkerError

    try:
        session.load(spec, model_id or spec.default_variant.repo_id, device, {"lora_path": lora} if lora else {}, progress)
        result = session.generate(text, voices, language, instruct, seed, parsed, output, progress)
    except WorkerError as e:
        if e.details:
            click.echo(e.details, err=True)
        sys.exit(f"✗ {e}")
    finally:
        session.unload()
    click.echo(
        f"{result.audio_path}  ({result.duration:.2f}s audio in {result.generation_time:.2f}s, RTF {result.rtf:.2f})"
    )


if __name__ == "__main__":
    main()
