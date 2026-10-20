"""Models tab: browse model families and install/remove their environments."""

import queue
import shutil
import threading

import gradio as gr

from ovl import paths
from ovl.backends import all_backends, get
from ovl.runtime import envs
from ovl.runtime.session import session
from ovl.ui.common import backend_choices, capability_summary, status_badge

_install_lock = threading.Lock()


def overview_rows():
    return [
        [spec.name, status_badge(spec), capability_summary(spec.capabilities), spec.license, len(spec.variants)]
        for spec in all_backends()
    ]


def details(backend_id: str) -> str:
    spec = get(backend_id)
    lines = [
        f"## {spec.name}",
        spec.description,
        "",
        f"**Status:** {status_badge(spec)}  ·  **License:** {spec.license or 'unknown'}"
        + (f"  ·  [Homepage]({spec.homepage})" if spec.homepage else ""),
        f"**Can do:** {capability_summary(spec.capabilities)}",
    ]
    if not spec.supports_mps:
        lines.append("**Note:** runs on CPU on Apple Silicon (no MPS support).")
    lines += ["", "**Checkpoints:**"]
    for v in spec.variants:
        extra = " · ".join(x for x in (v.size, v.notes) if x)
        lines.append(f"- [`{v.repo_id}`](https://huggingface.co/{v.repo_id})" + (f" · {extra}" if extra else ""))
    if spec.tips:
        lines += ["", f"**Tips:** {spec.tips}"]
    lines += ["", f"<small>Environment: `{envs.env_dir(spec)}` (Python {spec.python})</small>"]
    return "\n".join(lines)


def _buttons(backend_id: str):
    status = envs.status(get(backend_id))
    return (
        gr.update(value={"missing": "Install", "outdated": "Update"}.get(status, "Reinstall")),
        gr.update(interactive=status != "missing"),
    )


def select(backend_id: str):
    return (details(backend_id), *_buttons(backend_id))


def run_install(backend_id: str):
    """Stream the install log while uv builds the environment."""
    spec = get(backend_id)
    if not _install_lock.acquire(blocking=False):
        raise gr.Error("Another installation is already running")
    try:
        if session.current and session.current.spec.id == backend_id:
            session.unload()
        reinstall = envs.status(spec) == "ready"
        lines, q, outcome = [], queue.Queue(), {}

        def work():
            try:
                envs.install(spec, log=q.put, reinstall=reinstall)
            except Exception as e:
                outcome["error"] = str(e)
            finally:
                q.put(None)

        threading.Thread(target=work, daemon=True).start()
        done = False
        while not done:
            batch = [q.get()]
            while not q.empty():
                batch.append(q.get_nowait())
            done = None in batch
            lines.extend(line for line in batch if line is not None)
            yield "\n".join(lines[-400:]), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        if "error" in outcome:
            lines.append(f"✗ {outcome['error']}")
            gr.Warning(f"{spec.name} installation failed. See the log.")
        else:
            gr.Success(f"{spec.name} is ready")
        yield ("\n".join(lines[-400:]), details(backend_id), *_buttons(backend_id), overview_rows(),
               gr.update(choices=backend_choices(), value=backend_id))
    finally:
        _install_lock.release()


def run_remove(backend_id: str):
    spec = get(backend_id)
    if session.current and session.current.spec.id == backend_id:
        session.unload()
    envs.remove(spec)
    gr.Info(f"Removed the {spec.name} environment")
    return details(backend_id), *_buttons(backend_id), overview_rows(), gr.update(choices=backend_choices())


def disk_usage_note() -> str:
    try:
        free = shutil.disk_usage(paths.envs_dir()).free / 1e9
        return f"<small>Each model family installs into its own environment under `{paths.envs_dir()}` " \
               f"(shared packages are hard-linked, so torch is only stored once). Weights download to the " \
               f"Hugging Face cache on first load. {free:.0f} GB free.</small>"
    except OSError:
        return ""


def build(demo: gr.Blocks):
    gr.Markdown("Install a model family here, then load it on the **Generate** tab.")
    overview = gr.Dataframe(
        value=overview_rows,
        headers=["Model", "Status", "Features", "License", "Checkpoints"],
        interactive=False,
        wrap=True,
    )
    with gr.Row():
        with gr.Column(scale=2):
            backend = gr.Dropdown(choices=backend_choices(), value=all_backends()[0].id, label="Model family")
            info = gr.Markdown(details(all_backends()[0].id))
            with gr.Row():
                install_btn = gr.Button("Install", variant="primary")
                remove_btn = gr.Button("Remove", variant="stop", interactive=False)
            gr.Markdown(disk_usage_note())
        with gr.Column(scale=3):
            log = gr.Textbox(label="Install log", lines=22, max_lines=22, autoscroll=True, interactive=False)

    backend.change(select, backend, [info, install_btn, remove_btn])
    install_btn.click(
        run_install, backend, [log, info, install_btn, remove_btn, overview, backend], concurrency_id="install"
    )
    remove_btn.click(run_remove, backend, [info, install_btn, remove_btn, overview, backend])
    demo.load(select, backend, [info, install_btn, remove_btn])
    return backend
