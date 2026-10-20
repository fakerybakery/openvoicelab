import json
import re
from pathlib import Path

import numpy as np
import torch
import torchaudio
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor

from ovl.audio import load_audio, save_audio
from ovl.backends.base import Backend as BaseBackend
from ovl.backends.base import LoadInfo, Request, resolve_device, seed_everything, split_speaker_script

# The checkpoints ship their model code (trust_remote_code), so pin the revisions it was tested with.
REVISIONS = {
    "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5": "be7766a6735b98bd793f7c79fb720b4d0f5d13b8",
    "OpenMOSS-Team/MOSS-TTS-v1.5": "cdd3b911b1585e3f2dbc7775ef10f9926f58850a",
    "OpenMOSS-Team/MOSS-TTSD-v1.0": "c7cd852d87aff71cab5bd2b9b05509cedc0ef1ba",
    "OpenMOSS-Team/MOSS-VoiceGenerator": "97521ec2b6f3ec5026ac1f5751f8fc302d82c2d4",
    "OpenMOSS-Team/MOSS-TTS-Nano": "44502f80dbf9743528fa921cc544d662c685ebec",
    "OpenMOSS-Team/MOSS-Audio-Tokenizer": "3cd226ba2947efa357ef453bcad111b6eafba782",
    "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2": "f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169",
    "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano": "6aa02b01e445cc585582cf0ba480bc3ea6c8dd68",
}

# Recommended audio sampling (temperature, top_p, top_k, repetition_penalty) from the model cards.
SAMPLING = {
    "tts": (1.7, 0.8, 25, 1.0),
    "local_v1": (1.0, 0.95, 50, 1.1),
    "ttsd": (1.1, 0.9, 50, 1.1),
    "voicegen": (1.5, 0.6, 50, 1.1),
    "nano": (0.8, 0.95, 25, 1.2),
}
FRAME_RATE = 12.5  # audio tokens per second for every MOSS codec
TTSD_MAX_SPEAKERS = 5


def _load(path, *args, **kwargs):
    wav, sr = load_audio(str(path), mono=False)
    wav = wav[None] if wav.ndim == 1 else wav.T
    return torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)), sr


def _save(path, src, sample_rate, *args, **kwargs):
    save_audio(str(path), src, sample_rate)


# torchaudio>=2.9 routes load/save through torchcodec (needs FFmpeg dylibs); the Nano model code uses them.
torchaudio.load, torchaudio.save = _load, _save


def _kind(repo_id: str) -> str:
    name = repo_id.lower()
    if "nano" in name:
        return "nano"
    if "ttsd" in name:
        return "ttsd"
    if "voicegenerator" in name or "voice-generator" in name:
        return "voicegen"
    if "local-transformer" in name and "v1.5" not in name:
        return "local_v1"
    return "tts"


def _download(repo_id: str) -> str:
    if Path(repo_id).exists():
        return repo_id
    return snapshot_download(repo_id, revision=REVISIONS.get(repo_id))


class Backend(BaseBackend):
    def load(self, model_id, device, options):
        self.device = resolve_device(device)
        self.kind = _kind(model_id)
        if self.kind == "nano" and self.device == "mps" and device in (None, "auto"):
            self.device = "cpu"  # the 0.1B Nano is several times faster on CPU than on MPS
        if self.device == "cuda":
            torch.backends.cuda.enable_cudnn_sdp(False)  # upstream: cuDNN SDPA is broken for these models
        self.progress(None, f"Downloading {model_id}...")
        path = _download(model_id)

        if self.kind == "nano":
            self._load_nano(path)
        else:
            self._load_tts(path)

        caps, languages = {}, ["Auto"]
        if self.kind == "tts":
            from ovl.backends.catalog.moss_tts import SPEC

            languages = list(SPEC.capabilities.languages)
        elif self.kind == "ttsd":
            caps = {"clone_needs_transcript": True, "max_speakers": TTSD_MAX_SPEAKERS}
        elif self.kind == "voicegen":
            caps = {"clone": False, "voice_design": True, "no_voice": False}
            languages = ["Auto", "English", "Chinese"]
        return LoadInfo(languages=languages, capabilities=caps, sample_rate=self.sample_rate)

    def _load_tts(self, path):
        cfg = Path(path) / "processor_config.json"
        codec = json.loads(cfg.read_text()).get("audio_tokenizer_name_or_path") if cfg.exists() else None
        codec = codec or "OpenMOSS-Team/MOSS-Audio-Tokenizer"
        self.progress(None, f"Downloading {codec}...")
        codec_path = _download(codec)

        self.progress(None, "Loading MOSS-TTS...")
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, codec_path=codec_path)
        self.processor.audio_tokenizer = self.processor.audio_tokenizer.to(self.device).eval()
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self.model = AutoModel.from_pretrained(
            path, trust_remote_code=True, attn_implementation="sdpa", dtype=dtype
        ).to(self.device)
        self.model.eval()
        self.sample_rate = int(self.processor.model_config.sampling_rate)

    def _load_nano(self, path):
        self.progress(None, "Downloading MOSS-Audio-Tokenizer-Nano...")
        self.codec_path = _download("OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano")
        self.progress(None, "Loading MOSS-TTS-Nano...")
        self.model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
        # The Nano code targets transformers 4.x; under 5.x its non-persistent RoPE buffer is
        # left as zeros after meta-device loading, which wipes out positions (garbled, truncated audio).
        base = float(getattr(self.model.config, "rope_base", 10000.0))
        for module in self.model.modules():
            if type(module).__name__.endswith("RotaryEmbedding") and hasattr(module, "inv_freq"):
                dim = module.inv_freq.numel() * 2
                module.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.model.to(device=self.device, dtype=torch.float32)
        self.model._set_attention_implementation("sdpa")
        self.model.eval()
        self.sample_rate = int(self.model.config.audio_tokenizer_sample_rate)

    def _sampling(self, params):
        defaults = SAMPLING[self.kind]
        names = ("temperature", "top_p", "top_k", "repetition_penalty")
        values = [params.get(n) or d for n, d in zip(names, defaults)]
        return dict(zip(("audio_temperature", "audio_top_p", "audio_top_k", "audio_repetition_penalty"), values))

    def _encode(self, wavs):
        return self.processor.encode_audios_from_wav(wavs, sampling_rate=self.sample_rate)

    def _clip(self, voice):
        mono = self.sample_rate != 48000
        wav, _ = load_audio(voice.audio, sr=self.sample_rate, mono=mono)
        return torch.from_numpy(wav[None] if wav.ndim == 1 else wav.T.copy())

    def generate(self, request: Request):
        text = request.text.strip()
        if not text:
            raise ValueError("Enter some text to synthesize")
        seed_everything(request.seed)
        params = request.params
        max_tokens = int(float(params.get("max_seconds", 120)) * FRAME_RATE) + 32
        if self.kind == "nano":
            return self._generate_nano(request, text, max_tokens)

        if self.kind == "ttsd":
            conversation, mode = self._ttsd_conversation(request, text)
        elif self.kind == "voicegen":
            description = request.voice.description or request.instruct
            if not description:
                raise ValueError("Describe the voice you want (VoiceGenerator designs a voice from a description)")
            msg = self.processor.build_user_message(text=text, instruction=description, normalize=True)
            conversation, mode = [msg], "generation"
        else:
            conversation, mode = self._tts_conversation(request, text)

        batch = self.processor([conversation], mode=mode)
        self.progress(None, "Generating speech...")
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                max_new_tokens=max_tokens,
                **self._sampling(params),
            )
        messages = self.processor.decode(outputs)
        if not messages or messages[0] is None or not messages[0].audio_codes_list:
            raise RuntimeError("MOSS-TTS produced no audio for this text")
        save_audio(request.output_path, messages[0].audio_codes_list[0], self.sample_rate)
        return {}

    def _tts_conversation(self, request, text):
        kwargs = {"text": text}
        if self.kind == "tts" and request.language and request.language != "Auto":
            kwargs["language"] = request.language
        voice = request.voice
        if not voice.audio:
            return [self.processor.build_user_message(**kwargs)], "generation"
        codes = self._encode([self._clip(voice)])[0]
        if request.params.get("use_transcript") and voice.transcript:
            # Continuation + clone: the clip is the start of the answer, its transcript the start of the text.
            kwargs["text"] = f"{voice.transcript.strip()} {text}"
            user = self.processor.build_user_message(reference=[codes], **kwargs)
            return [user, self.processor.build_assistant_message(audio_codes_list=[codes])], "continuation"
        return [self.processor.build_user_message(reference=[codes], **kwargs)], "generation"

    def _ttsd_conversation(self, request, text):
        n = max(len(request.voices), 1)
        lines = split_speaker_script(text, n)
        if not lines:
            raise ValueError("Enter a dialogue script ('Speaker 1: ...' lines)")
        dialogue = _merge_tags("".join(f"[S{i + 1}] {line} " for i, line in lines))

        cloned = [i for i, v in enumerate(request.voices[:TTSD_MAX_SPEAKERS]) if v.audio]
        if not cloned:
            return [self.processor.build_user_message(text=dialogue)], "generation"
        missing = [i + 1 for i in cloned if not (request.voices[i].transcript or "").strip()]
        if missing:
            raise ValueError(
                f"MOSS-TTSD needs the transcript of each reference clip (missing for speaker {', '.join(map(str, missing))})"
            )
        clips = [self._clip(request.voices[i]) for i in cloned]
        codes = self._encode(clips)
        reference = [None] * (max(cloned) + 1)
        for i, c in zip(cloned, codes):
            reference[i] = c
        prefix = "".join(f"[S{i + 1}] {request.voices[i].transcript.strip()} " for i in cloned)
        prompt_audio = self._encode([torch.cat(clips, dim=-1)])[0]
        user = self.processor.build_user_message(text=_merge_tags(prefix + dialogue), reference=reference)
        return [user, self.processor.build_assistant_message(audio_codes_list=[prompt_audio])], "continuation"

    def _generate_nano(self, request, text, max_frames):
        voice = request.voice
        kwargs = {"mode": "continuation"}
        if voice.audio:
            kwargs["prompt_audio_path"] = voice.audio
            if request.params.get("use_transcript") and voice.transcript:
                kwargs["prompt_text"] = voice.transcript.strip()
            else:
                kwargs["mode"] = "voice_clone"
        self.progress(None, "Generating speech...")
        with torch.inference_mode():
            self.model.inference(
                text=text,
                output_audio_path=request.output_path,
                audio_tokenizer_pretrained_name_or_path=self.codec_path,
                device=self.device,
                max_new_frames=max_frames,
                do_sample=True,
                text_temperature=1.0,
                **self._sampling(request.params),
                **kwargs,
            )
        return {}


def _merge_tags(text: str) -> str:
    """Drop repeated [Sn] tags when the same speaker continues."""
    out, current = [], None
    for seg in re.split(r"(?=\[S\d+\])", text):
        m = re.match(r"^(\[S\d+\])\s*(.*)", seg.strip(), re.DOTALL)
        if not m:
            continue
        tag, content = m.groups()
        out.append(content if tag == current else f"{tag}{content}")
        current = tag
    return " ".join(p.strip() for p in out if p.strip())
