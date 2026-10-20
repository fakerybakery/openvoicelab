"""Generate tab: load a model and synthesize speech.

Everything shown here is driven by the backend's capabilities: the spec says what a
model family can do, and the loaded checkpoint can narrow that down (LoadInfo).
"""

import logging
from dataclasses import asdict
from datetime import datetime

import gradio as gr

from ovl import paths, voice_library
from ovl.backends import all_backends, get
from ovl.runtime import envs
from ovl.runtime.client import WorkerError
from ovl.runtime.session import session
from ovl.ui.common import backend_choices, capability_summary, variant_choices

logger = logging.getLogger(__name__)

MAX_SPEAKERS = 4
PRESET, LIBRARY, UPLOAD, DESIGN, DEFAULT = "Preset voice", "Voice library", "Upload clip", "Describe a voice", "Default voice"
SPEAKER_FIELDS = ("group", "source", "preset", "library", "upload", "transcript", "design")

_last_log = ""


def spec_caps(spec) -> dict:
    caps = asdict(spec.capabilities)
    caps["languages"] = list(caps["languages"])
    caps["preset_list"] = []
    return caps


def current_caps(backend_id: str) -> dict:
    """Capabilities of the selected backend: the loaded checkpoint's if it is loaded."""
    spec = get(backend_id)
    caps = spec_caps(spec)
    model = session.current
    if model and model.spec.id == backend_id:
        caps.update(model.info.get("capabilities") or {})
        caps["preset_list"] = model.info.get("presets") or []
        caps["languages"] = model.info.get("languages") or caps["languages"]
    return caps


def voice_sources(caps: dict):
    sources = []
    if caps["presets"]:
        sources.append(PRESET)
    if caps["clone"]:
        sources += [LIBRARY, UPLOAD]
    if caps["voice_design"]:
        sources.append(DESIGN)
    if caps["no_voice"]:
        sources.append(DEFAULT)
    return sources


def lora_choices():
    return sorted(str(p) for p in paths.runs_dir().glob("**/lora") if p.is_dir())


def model_status() -> str:
    model = session.current
    if model is None:
        return "No model loaded."
    notes = model.info.get("notes")
    return f"**Loaded:** {model.spec.name} · `{model.model_id}`" + (f"\n\n{notes}" if notes else "")


class GenerateTab:
    def build(self, demo: gr.Blocks):
        specs = all_backends()
        first = specs[0]

        with gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=300):
                self.backend = gr.Dropdown(choices=backend_choices(), value=first.id, label="Model family")
                self.backend_info = gr.Markdown(self._backend_info(first.id))
                self.variant = gr.Dropdown(
                    choices=variant_choices(first),
                    value=first.default_variant.repo_id,
                    label="Checkpoint",
                    info="Pick one, or paste a Hugging Face repo id / local path",
                    allow_custom_value=True,
                )
                self.device = gr.Dropdown(["auto", "cuda", "mps", "cpu"], value="auto", label="Device")
                self.lora = gr.Dropdown(
                    choices=lora_choices(),
                    value=None,
                    label="LoRA adapter (optional)",
                    info="A lora/ folder from a training run",
                    allow_custom_value=True,
                    visible=first.capabilities.lora,
                )
                with gr.Row():
                    self.load_btn = gr.Button("Load model", variant="primary")
                    self.unload_btn = gr.Button("Unload")
                self.status = gr.Markdown(model_status())

            with gr.Column(scale=5):
                self.text = gr.Textbox(label="Text", lines=6, placeholder="Type something to say...")
                with gr.Row():
                    self.language = gr.Dropdown(label="Language", visible=False)
                    self.num_speakers = gr.Slider(1, MAX_SPEAKERS, value=1, step=1, label="Speakers", visible=False)
                self.instruct = gr.Textbox(
                    label="Style instruction",
                    placeholder="e.g. Speak warmly and slowly, with a hint of excitement",
                    visible=False,
                )
                self.speakers = []
                with gr.Row():
                    for i in range(MAX_SPEAKERS):
                        with gr.Column(visible=i == 0, min_width=220) as group:
                            spk = {"group": group}
                            spk["source"] = gr.Radio(label=f"Speaker {i + 1} voice" if i else "Voice", choices=[])
                            spk["preset"] = gr.Dropdown(label="Preset", choices=[], visible=False)
                            spk["library"] = gr.Dropdown(label="Library voice", choices=[], visible=False)
                            spk["upload"] = gr.Audio(label="Reference clip", type="filepath", visible=False)
                            spk["transcript"] = gr.Textbox(label="Transcript of the clip", lines=2, visible=False)
                            spk["design"] = gr.Textbox(
                                label="Voice description",
                                lines=2,
                                placeholder="e.g. A deep, gravelly male voice with a slight Scottish accent",
                                visible=False,
                            )
                        self.speakers.append(spk)

                with gr.Accordion("Advanced settings", open=False):
                    self.param_groups, self.param_inputs = {}, []
                    for spec in specs:
                        with gr.Group(visible=spec.id == first.id) as group:
                            for p in spec.params:
                                self.param_inputs.append((spec.id, p.name, self._param_input(p)))
                        self.param_groups[spec.id] = group
                    self.seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)

                with gr.Row():
                    self.generate_btn = gr.Button("Generate", variant="primary", size="lg")
                    self.stop_btn = gr.Button("Stop", variant="stop", size="lg")
                self.output = gr.Audio(label="Output", type="filepath", interactive=False, autoplay=True)
                self.metrics = gr.Markdown()
                with gr.Accordion("History", open=False):
                    self.history = gr.State([])
                    self.history_table = gr.Dataframe(
                        headers=["Time", "Model", "Text", "Length"], interactive=False, wrap=True
                    )
                with gr.Accordion("Worker log", open=False):
                    self.log = gr.Textbox(lines=15, max_lines=15, show_label=False, interactive=False, autoscroll=True)
                    self.refresh_log = gr.Button("Refresh log", size="sm")

        self._wire(demo)

    @staticmethod
    def _param_input(p):
        kw = {"label": p.label, "value": p.default, "info": p.info or None}
        if p.kind == "bool":
            return gr.Checkbox(**kw)
        if p.kind == "choice":
            return gr.Dropdown(choices=list(p.choices), **kw)
        if p.kind == "text":
            return gr.Textbox(**kw)
        if p.minimum is not None and p.maximum is not None:
            step = p.step or (1 if p.kind == "int" else 0.01)
            return gr.Slider(p.minimum, p.maximum, step=step, **kw)
        return gr.Number(precision=0 if p.kind == "int" else None, **kw)

    @staticmethod
    def _backend_info(backend_id: str) -> str:
        spec = get(backend_id)
        status = envs.status(spec)
        line = f"{spec.description}\n\n<small>{capability_summary(spec.capabilities)} · {spec.license}</small>"
        if status != "ready":
            line += "\n\n*Not installed yet. **Load model** installs it first (this can take a few minutes).*"
        return line

    # --- voice panel ---------------------------------------------------------

    @property
    def _speaker_outputs(self):
        return [spk[f] for spk in self.speakers for f in SPEAKER_FIELDS]

    @property
    def _panel_outputs(self):
        return [self.language, self.num_speakers, self.instruct, *self._speaker_outputs]

    def _panel_updates(self, backend_id, data=None):
        """Updates for the language/speakers/instruct/voice controls, keeping user choices when still valid."""
        data = data or {}
        caps = current_caps(backend_id)
        sources = voice_sources(caps)
        library = voice_library.list_voices()
        presets = caps["preset_list"]
        max_spk = min(caps["max_speakers"] or 1, MAX_SPEAKERS)
        n = min(int(data.get(self.num_speakers) or 1), max_spk)
        languages = caps["languages"]
        language = data.get(self.language)

        def keep(value, options, fallback):
            return value if value in options else fallback

        updates = [
            gr.update(choices=languages, value=keep(language, languages, languages[0] if languages else None),
                      visible=len(languages) > 1),
            gr.update(maximum=max(max_spk, 2), value=n, visible=max_spk > 1),
            gr.update(visible=bool(caps["instruct"])),
        ]
        needs_transcript = caps["clone_needs_transcript"]
        for i, spk in enumerate(self.speakers):
            src = keep(data.get(spk["source"]), sources, sources[0] if sources else None)
            updates += [
                gr.update(visible=i < n and bool(sources)),
                gr.update(choices=sources, value=src, visible=len(sources) > 1),
                gr.update(
                    choices=presets,
                    value=keep(data.get(spk["preset"]), presets, presets[i % len(presets)] if presets else None),
                    visible=src == PRESET,
                    info="" if presets else "Load the model to see its voices",
                ),
                gr.update(
                    choices=library,
                    value=keep(data.get(spk["library"]), library, library[i % len(library)] if library else None),
                    visible=src == LIBRARY,
                ),
                gr.update(visible=src == UPLOAD),
                gr.update(
                    visible=src == UPLOAD,
                    label="Transcript of the clip" + (" (auto-transcribed if empty)" if needs_transcript else " (optional)"),
                ),
                gr.update(visible=src == DESIGN),
            ]
        return updates

    def _panel_inputs(self):
        return {self.backend, self.language, self.num_speakers,
                *[spk[f] for spk in self.speakers for f in ("source", "preset", "library")]}

    # --- events --------------------------------------------------------------

    def on_backend_change(self, data):
        backend_id = data[self.backend]
        spec = get(backend_id)
        loaded = session.current and session.current.spec.id == backend_id
        return [
            gr.update(choices=variant_choices(spec),
                      value=session.current.model_id if loaded else spec.default_variant.repo_id),
            self._backend_info(backend_id),
            gr.update(visible=spec.capabilities.lora, choices=lora_choices()),
            *[gr.update(visible=sid == backend_id) for sid in self.param_groups],
            *self._panel_updates(backend_id, data),
        ]

    def on_load(self, data, progress=gr.Progress()):
        global _last_log
        spec = get(data[self.backend])
        model_id = (data[self.variant] or spec.default_variant.repo_id).strip()
        skip = [gr.update()] * (len(self._panel_outputs) + 1)

        if envs.status(spec) != "ready":
            lines = []

            def log(line):
                lines.append(line)
                progress(None, desc=line[-120:])

            try:
                envs.install(spec, log=log)
            except Exception as e:
                _last_log = "\n".join(lines[-200:])
                raise gr.Error(f"Installing {spec.name} failed: {e}. See 'Worker log' for the install output.")
            yield [f"Installed {spec.name}. Loading...", gr.update(choices=backend_choices(), value=spec.id), *skip[1:]]

        options = {"lora_path": data[self.lora]} if data.get(self.lora) and spec.capabilities.lora else {}
        try:
            session.load(spec, model_id, data[self.device], options,
                         lambda v, d: progress(v, desc=d or f"Loading {spec.name}..."))
        except WorkerError as e:
            _last_log = e.details
            raise gr.Error(f"Could not load {model_id}: {e}. See 'Worker log' for details.")
        except Exception as e:
            raise gr.Error(str(e))
        gr.Success(f"{spec.name} loaded")
        yield [model_status(), gr.update(), *self._panel_updates(spec.id, data)]

    def on_unload(self, data):
        session.unload()
        return [model_status(), *self._panel_updates(data[self.backend], data)]

    def on_stop(self):
        if session.current is None:
            return model_status()
        session.cancel()
        session.unload()
        gr.Info("Stopped. The model was unloaded; load it again to continue.")
        return model_status()

    def _voice_for(self, spk, data, caps, index):
        src = data[spk["source"]]
        label = f"speaker {index + 1}" if (data.get(self.num_speakers) or 1) > 1 else "the voice"
        if src == PRESET:
            if not data[spk["preset"]]:
                raise gr.Error(f"Pick a preset for {label}")
            return {"preset": data[spk["preset"]]}
        if src in (LIBRARY, UPLOAD):
            if src == LIBRARY:
                name = data[spk["library"]]
                if not name:
                    raise gr.Error(f"Pick a library voice for {label} (add some on the Voices tab)")
                audio, transcript = str(voice_library.audio_path(name)), voice_library.transcript(name)
            else:
                audio, transcript = data[spk["upload"]], (data[spk["transcript"]] or "").strip()
                if not audio:
                    raise gr.Error(f"Upload a reference clip for {label}")
            if not transcript and caps["clone_needs_transcript"]:
                from ovl import asr

                gr.Info("Transcribing the reference clip with Whisper...")
                transcript = asr.transcribe(audio)
                asr.unload()
                if src == LIBRARY:
                    voice_library.set_transcript(name, transcript)
            return {"audio": audio, "transcript": transcript or None}
        if src == DESIGN:
            if not (data[spk["design"]] or "").strip():
                raise gr.Error(f"Describe the voice for {label}")
            return {"description": data[spk["design"]].strip()}
        return {}

    def on_generate(self, data, progress=gr.Progress()):
        global _last_log
        model = session.current
        backend_id = data[self.backend]
        if model is None or model.spec.id != backend_id:
            raise gr.Error(f"Load {get(backend_id).name} first")
        text = (data[self.text] or "").strip()
        if not text:
            raise gr.Error("Enter some text")

        caps = current_caps(backend_id)
        n = min(int(data[self.num_speakers] or 1), caps["max_speakers"] or 1)
        voices = [self._voice_for(spk, data, caps, i) for i, spk in enumerate(self.speakers[:n])] if voice_sources(caps) else []
        params = {name: data[comp] for sid, name, comp in self.param_inputs if sid == backend_id}
        languages = caps["languages"]
        language = data[self.language] if data[self.language] in languages else (languages[0] if languages else None)
        seed = int(data[self.seed]) if data[self.seed] is not None and data[self.seed] >= 0 else None

        try:
            result = session.generate(
                text,
                voices,
                language=language if len(languages) > 1 else None,
                instruct=((data[self.instruct] or "").strip() or None) if caps["instruct"] else None,
                seed=seed,
                params=params,
                on_progress=lambda v, d: progress(v, desc=d or "Generating..."),
            )
        except WorkerError as e:
            _last_log = e.details
            raise gr.Error(f"{e}. See 'Worker log' for details.")
        except Exception as e:
            raise gr.Error(str(e))

        speed = result.duration / result.generation_time if result.generation_time else 0
        extra = " · ".join(f"{k}: {v}" for k, v in result.metrics.items())
        metrics = f"{result.duration:.1f}s of audio in {result.generation_time:.1f}s ({speed:.1f}× realtime)" + (
            f" · {extra}" if extra else ""
        )
        history = [
            {"time": datetime.now().strftime("%H:%M:%S"), "model": model.spec.name, "text": text,
             "duration": result.duration, "path": result.audio_path},
            *data[self.history],
        ][:50]
        return result.audio_path, metrics, history, self._history_rows(history)

    @staticmethod
    def _history_rows(history):
        return [[h["time"], h["model"], h["text"][:80] + ("…" if len(h["text"]) > 80 else ""),
                 f"{h['duration']:.1f}s"] for h in history]

    @staticmethod
    def on_history_select(history, evt: gr.SelectData):
        row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        return history[row]["path"] if 0 <= row < len(history) else gr.update()

    @staticmethod
    def worker_log():
        if session.current is not None:
            return session.current.client.recent_log(300)
        return _last_log

    def _wire(self, demo):
        panel_in = self._panel_inputs()
        self.backend.change(
            self.on_backend_change,
            panel_in,
            [self.variant, self.backend_info, self.lora, *self.param_groups.values(), *self._panel_outputs],
        )
        # .input/.release fire only on user interaction, so the updates above don't
        # re-trigger these with the previous model's values.
        for spk in self.speakers:
            spk["source"].input(
                lambda data: self._panel_updates(data[self.backend], data), panel_in, self._panel_outputs
            )
        self.num_speakers.release(lambda data: self._panel_updates(data[self.backend], data), panel_in, self._panel_outputs)

        load_inputs = panel_in | {self.variant, self.device, self.lora}
        self.load_btn.click(
            self.on_load, load_inputs, [self.status, self.backend, *self._panel_outputs], concurrency_id="model"
        ).then(self.worker_log, None, self.log)
        self.unload_btn.click(self.on_unload, panel_in, [self.status, *self._panel_outputs], concurrency_id="model")

        gen_inputs = panel_in | {self.text, self.instruct, self.seed, self.history,
                                 *[spk[f] for spk in self.speakers for f in ("upload", "transcript", "design")],
                                 *[comp for _, _, comp in self.param_inputs]}
        gen = self.generate_btn.click(
            self.on_generate, gen_inputs, [self.output, self.metrics, self.history, self.history_table],
            concurrency_id="model",
        )
        gen.then(self.worker_log, None, self.log)
        self.stop_btn.click(self.on_stop, None, self.status, cancels=[gen])
        self.history_table.select(self.on_history_select, self.history, self.output)
        self.refresh_log.click(self.worker_log, None, self.log)
        demo.load(lambda data: [model_status(), *self._panel_updates(data[self.backend], data)], panel_in,
                  [self.status, *self._panel_outputs])


def build(demo: gr.Blocks):
    tab = GenerateTab()
    tab.build(demo)
    return tab
