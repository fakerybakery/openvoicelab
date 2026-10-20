import gc

import torch
import torchaudio
from huggingface_hub import snapshot_download
from voxcpm import VoxCPM
from voxcpm.model.voxcpm2 import VoxCPM2Model

from ovl.audio import load_audio, save_audio
from ovl.backends.base import Backend as BaseBackend
from ovl.backends.base import LoadInfo, Request, resolve_device, seed_everything


def _load(path, *args, **kwargs):
    wav, sr = load_audio(path)
    return torch.from_numpy(wav).unsqueeze(0), sr


# VoxCPM 1.x reads prompts with torchaudio.load, which needs torchcodec + FFmpeg dylibs.
torchaudio.load = _load


class Backend(BaseBackend):
    model = None

    def load(self, model_id, device, options):
        self.device = resolve_device(device)
        self.progress(None, f"Downloading {model_id}...")
        path = snapshot_download(model_id)
        self.progress(None, "Loading VoxCPM...")
        # The ZipEnhancer denoiser pulls in modelscope and a second model; torch.compile only helps on CUDA.
        self.model = VoxCPM(
            voxcpm_model_path=path,
            zipenhancer_model_path=None,
            enable_denoiser=False,
            optimize=self.device.startswith("cuda"),
            device=self.device,
        )
        self.v2 = isinstance(self.model.tts_model, VoxCPM2Model)
        self.sample_rate = self.model.tts_model.sample_rate
        info = LoadInfo(sample_rate=self.sample_rate)
        if not self.v2:
            info.capabilities = {"voice_design": False, "instruct": False, "clone_needs_transcript": True}
            info.languages = ["English", "Chinese"]
            info.notes = "VoxCPM 1.x clones by continuing the reference clip, so it needs the clip's transcript."
        return info

    def generate(self, request: Request):
        voice = request.voice
        text = request.text.strip()
        kwargs = {}
        style = request.instruct if voice.audio else (voice.description or request.instruct)
        if style:
            if not self.v2:
                raise ValueError("Voice design and style instructions need VoxCPM2")
            # VoxCPM2 reads a parenthesized description/instruction at the start of the text.
            text = f"({style.strip().strip('()')}){text}"
        if voice.audio:
            if voice.transcript:
                kwargs.update(prompt_wav_path=voice.audio, prompt_text=voice.transcript)
            elif not self.v2:
                raise ValueError("VoxCPM 1.x needs the transcript of the reference clip")
            if self.v2:
                kwargs["reference_wav_path"] = voice.audio

        p = request.params
        seed_everything(request.seed)
        self.progress(None, "Generating...")
        wav = self.model.generate(
            text=text,
            cfg_value=float(p.get("cfg_value", 2.0)),
            inference_timesteps=int(p.get("inference_timesteps", 10)),
            normalize=bool(p.get("normalize", False)),
            retry_badcase=bool(p.get("retry_badcase", True)),
            **kwargs,
        )
        save_audio(request.output_path, wav, self.sample_rate)
        return {}

    def unload(self):
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
