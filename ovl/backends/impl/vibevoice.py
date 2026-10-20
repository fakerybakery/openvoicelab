import copy
import json
import os

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from vibevoice.modular.lora_loading import load_lora_assets
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

from ovl.audio import load_audio, save_audio
from ovl.backends.base import Backend as BaseBackend
from ovl.backends.base import LoadInfo, Request, resolve_device, seed_everything, split_speaker_script
from ovl.paths import PACKAGE_DIR

SAMPLE_RATE = 24000
STREAMING_VOICES_DIR = PACKAGE_DIR / "voices" / "streaming"
QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def is_streaming(model_id):
    path = os.path.join(model_id, "config.json")
    if not os.path.exists(path):
        path = hf_hub_download(model_id, "config.json")
    with open(path) as f:
        return json.load(f).get("model_type") == "vibevoice_streaming"


def streaming_voices():
    # "en-Emma_woman.pt" -> "Emma"
    return {p.stem.split("_")[0].split("-")[-1]: p for p in sorted(STREAMING_VOICES_DIR.glob("*.pt"))}


def cast_prompt(obj, dtype):
    """Cached voice prompts are stored in bfloat16; match the model dtype."""
    if torch.is_tensor(obj):
        return obj.to(dtype) if obj.is_floating_point() else obj
    if isinstance(obj, dict):
        return {k: cast_prompt(v, dtype) for k, v in obj.items()}
    if hasattr(obj, "key_cache"):
        obj.key_cache = [cast_prompt(t, dtype) for t in obj.key_cache]
        obj.value_cache = [cast_prompt(t, dtype) for t in obj.value_cache]
    return obj


class Backend(BaseBackend):
    def load(self, model_id, device, options):
        self.device = resolve_device(device)
        self.streaming = is_streaming(model_id)
        self.dtype = dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        model_cls = VibeVoiceStreamingForConditionalGenerationInference if self.streaming else VibeVoiceForConditionalGenerationInference
        processor_cls = VibeVoiceStreamingProcessor if self.streaming else VibeVoiceProcessor

        self.progress(None, f"Loading {model_id}...")
        self.processor = processor_cls.from_pretrained(model_id)
        self.model = model_cls.from_pretrained(model_id, torch_dtype=dtype, attn_implementation="sdpa").to(self.device)

        notes = ""
        lora_path = options.get("lora_path")
        if lora_path:
            if self.streaming:
                raise ValueError("LoRA adapters are only supported by the standard VibeVoice models (1.5B / 7B)")
            self.progress(None, "Loading LoRA adapter...")
            if not os.path.exists(lora_path):
                lora_path = snapshot_download(lora_path)
            report = load_lora_assets(self.model, lora_path)
            loaded = [k for k in ("language_model", "diffusion_head_lora", "diffusion_head_full", "acoustic_connector", "semantic_connector") if getattr(report, k)]
            if not loaded:
                raise ValueError(f"No LoRA weights found in {lora_path}")
            notes = "LoRA: " + ", ".join(loaded)
        self.model.eval()
        self.steps = None

        if self.streaming:
            self.voices = streaming_voices()
            caps = dict(presets=True, clone=False, max_speakers=1, no_voice=False, lora=False, finetune=False, languages=["English"])
            return LoadInfo(presets=list(self.voices), languages=["English"], capabilities=caps, sample_rate=SAMPLE_RATE, notes=notes)
        return LoadInfo(languages=["English", "Chinese"], sample_rate=SAMPLE_RATE, notes=notes)

    def generate(self, request: Request):
        steps = int(request.params.get("ddpm_steps", 10))
        if steps != self.steps:
            self.model.set_ddpm_inference_steps(num_steps=steps)
            self.steps = steps
        seed_everything(request.seed)
        text = request.text.translate(QUOTES)
        cfg_scale = float(request.params.get("cfg_scale", 1.3))
        with torch.inference_mode():
            if self.streaming:
                outputs, inputs_len = self._generate_streaming(text, request, cfg_scale)
            else:
                outputs, inputs_len = self._generate_standard(text, request, cfg_scale)
        wav = outputs.speech_outputs[0]
        if wav is None:
            raise ValueError("VibeVoice produced no audio for this text")
        save_audio(request.output_path, wav.float().cpu().numpy().squeeze(), SAMPLE_RATE)
        return {"generated_tokens": int(outputs.sequences.shape[1] - inputs_len)}

    def _generate_standard(self, text, request, cfg_scale):
        voices = [v for v in request.voices if not v.is_empty]
        if any(not v.audio for v in voices):
            raise ValueError("VibeVoice clones voices from reference clips; pick an audio clip for each speaker")
        lines = split_speaker_script(text, len(voices) or 4)
        if not lines:
            raise ValueError("Enter some text to synthesize")
        script = "\n".join(f"Speaker {i}: {line}" for i, line in lines)
        samples = [load_audio(v.audio, sr=SAMPLE_RATE)[0] for v in voices[: max(i for i, _ in lines) + 1]]

        inputs = self.processor(
            text=[script],
            voice_samples=[samples] if samples else None,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        progress = self.progress

        class Progress:
            def __init__(self, iterable, **kwargs):
                self.iterable = iterable

            def __iter__(self):
                total = len(self.iterable)
                for i, step in enumerate(self.iterable):
                    if i % 10 == 0:
                        progress(i / total, f"Generating ({i} tokens)")
                    yield step

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=cfg_scale,
            tokenizer=self.processor.tokenizer,
            generation_config={"do_sample": False},
            is_prefill=bool(samples),
            tqdm_class=Progress,
        )
        return outputs, inputs["input_ids"].shape[1]

    def _generate_streaming(self, text, request, cfg_scale):
        name = request.voice.preset
        if not name:
            raise ValueError("Pick a VibeVoice Realtime voice")
        if name not in self.voices:
            raise ValueError(f"Unknown voice {name!r}. Available: {', '.join(self.voices)}")
        text = " ".join(line for _, line in split_speaker_script(text, 1))
        if not text:
            raise ValueError("Enter some text to synthesize")
        prompt = cast_prompt(torch.load(self.voices[name], map_location=self.device, weights_only=False), self.dtype)
        inputs = self.processor.process_input_with_cached_prompt(
            text=text, cached_prompt=prompt, padding=True, return_tensors="pt", return_attention_mask=True
        )
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        steps = [0]

        def tick():
            steps[0] += 1
            if steps[0] % 10 == 0:
                self.progress(None, f"Generating ({steps[0]} tokens)")
            return False

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=cfg_scale,
            tokenizer=self.processor.tokenizer,
            generation_config={"do_sample": False},
            all_prefilled_outputs=copy.deepcopy(prompt),
            show_progress_bar=False,
            stop_check_fn=tick,
        )
        return outputs, inputs["tts_text_ids"].shape[1] + prompt["tts_lm"]["last_hidden_state"].size(1)

    def unload(self):
        self.model = self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
