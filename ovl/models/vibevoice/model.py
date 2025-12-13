import copy
import json
import os
import time
from enum import Enum
from typing import List, Optional, Union

import torch
from vibevoice.modular.lora_loading import load_lora_assets
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

from ovl.models.base import GenerationResult, TTSModel


class VibeVoiceModelType(Enum):
    """Enum for VibeVoice model types"""
    STANDARD = "vibevoice"
    STREAMING = "vibevoice_streaming"


def detect_model_type(model_path: str) -> VibeVoiceModelType:
    """
    Detect the VibeVoice model type from config.json.

    Args:
        model_path: Path to the model directory or HuggingFace model ID

    Returns:
        VibeVoiceModelType enum indicating the model architecture
    """
    config_path = os.path.join(model_path, "config.json")

    if os.path.exists(config_path):
        # Local path
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Try to fetch from HuggingFace
        try:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(repo_id=model_path, filename="config.json")
            with open(config_file, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Could not load config.json: {e}")
            # Default to standard model
            return VibeVoiceModelType.STANDARD

    model_type = config.get("model_type", "vibevoice")

    if model_type == "vibevoice_streaming":
        return VibeVoiceModelType.STREAMING
    else:
        return VibeVoiceModelType.STANDARD


class VibeVoiceModel(TTSModel):
    """Wrapper for VibeVoice model with simplified interface"""

    def __init__(
        self,
        model_path: str = "vibevoice/VibeVoice-1.5B",
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        self.model_path = model_path
        self.device = device
        self.checkpoint_path = checkpoint_path

        print(f"Loading VibeVoice model on {device}...")
        self._load_model()

    def _print_lora_report(self, report):
        """Print detailed LoRA loading report"""
        print("\n" + "="*60)
        print("LoRA Loading Report")
        print("="*60)
        print(f"Adapter root: {report.adapter_root}")
        print("\nComponents loaded:")
        print(f"  ✓ Language Model LoRA:      {'YES' if report.language_model else 'NO'}")
        print(f"  ✓ Diffusion Head LoRA:      {'YES' if report.diffusion_head_lora else 'NO'}")
        print(f"  ✓ Diffusion Head Full:      {'YES' if report.diffusion_head_full else 'NO'}")
        print(f"  ✓ Acoustic Connector:       {'YES' if report.acoustic_connector else 'NO'}")
        print(f"  ✓ Semantic Connector:       {'YES' if report.semantic_connector else 'NO'}")

        loaded_count = sum([
            report.language_model,
            report.diffusion_head_lora,
            report.diffusion_head_full,
            report.acoustic_connector,
            report.semantic_connector
        ])
        print(f"\nTotal components loaded: {loaded_count}/5")
        print("="*60 + "\n")

    def _load_model(self):
        """Load processor and model"""
        # Load processor
        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)

        # Determine dtype and attention implementation
        if self.device == "mps":
            load_dtype = torch.float32
            attn_impl = "sdpa"
        elif self.device == "cuda":
            load_dtype = torch.bfloat16
            attn_impl = "flash_attention_2"
        else:  # cpu
            load_dtype = torch.float32
            attn_impl = "sdpa"

        # Load model
        try:
            if self.device == "mps":
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    attn_implementation=attn_impl,
                    device_map=None,
                )
                self.model.to("mps")
            elif self.device == "cuda":
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    attn_implementation=attn_impl,
                )
                self.model.to("cuda")
            else:  # cpu
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map="cpu",
                    attn_implementation=attn_impl,
                )
        except Exception as e:
            if attn_impl == "flash_attention_2":
                print(f"Flash attention failed, falling back to SDPA: {e}")
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    attn_implementation="sdpa",
                )
                if self.device in ("mps", "cuda"):
                    self.model.to(self.device)
            else:
                raise e

        # Load LoRA checkpoint if provided
        if self.checkpoint_path:
            print(f"Loading checkpoint from {self.checkpoint_path}")

            # Check if it's a HuggingFace repo or local path
            if not os.path.exists(self.checkpoint_path):
                # Try to download from HuggingFace
                try:
                    from huggingface_hub import snapshot_download

                    print(f"Downloading LoRA adapter from HuggingFace: {self.checkpoint_path}")
                    local_path = snapshot_download(repo_id=self.checkpoint_path)
                    print(f"Downloaded to: {local_path}")
                    report = load_lora_assets(self.model, local_path)
                    self._print_lora_report(report)
                except Exception as e:
                    print(f"Failed to download from HuggingFace: {e}")
                    raise ValueError(
                        f"LoRA path not found locally and failed to download from HuggingFace: {self.checkpoint_path}"
                    )
            else:
                # Local path
                report = load_lora_assets(self.model, self.checkpoint_path)
                self._print_lora_report(report)

        self.model.eval()
        self.model.set_ddpm_inference_steps(num_steps=10)

    def generate(
        self,
        text: str,
        voice_samples: Optional[List[str]],
        output_path: str,
        cfg_scale: float = 1.3,
        enable_voice_cloning: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> GenerationResult:
        """
        Generate speech from text using voice samples

        Args:
            text: Input text to synthesize
            voice_samples: List of voice sample file paths, or None to disable voice cloning
            output_path: Path to save output audio
            cfg_scale: Classifier-free guidance scale
            enable_voice_cloning: Whether to enable voice cloning (prefill)
            progress_callback: Optional callback for progress updates (e.g., Gradio progress)

        Returns:
            GenerationResult with metrics
        """
        # Prepare processor inputs
        processor_kwargs = {
            "text": [text],
            "padding": True,
            "return_tensors": "pt",
            "return_attention_mask": True,
        }

        # Only pass voice_samples if provided (for voice cloning)
        if voice_samples is not None:
            processor_kwargs["voice_samples"] = [voice_samples]

        inputs = self.processor(**processor_kwargs)

        # Move to device
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device)

        # Generate
        start_time = time.time()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=cfg_scale,
            tokenizer=self.processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=True,
            is_prefill=enable_voice_cloning,
            tqdm_class=progress_callback if progress_callback else None,
        )
        generation_time = time.time() - start_time

        # Calculate metrics
        sample_rate = 24000
        audio_samples = (
            outputs.speech_outputs[0].shape[-1]
            if len(outputs.speech_outputs[0].shape) > 0
            else len(outputs.speech_outputs[0])
        )
        audio_duration = audio_samples / sample_rate
        rtf = generation_time / audio_duration if audio_duration > 0 else float("inf")

        input_tokens = inputs["input_ids"].shape[1]
        output_tokens = outputs.sequences.shape[1]
        generated_tokens = output_tokens - input_tokens

        # Save audio
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.processor.save_audio(outputs.speech_outputs[0], output_path=output_path)

        return GenerationResult(
            audio_path=output_path,
            audio_duration=audio_duration,
            generation_time=generation_time,
            rtf=rtf,
            input_tokens=input_tokens,
            generated_tokens=generated_tokens,
        )


class VibeVoiceStreamingModel(TTSModel):
    """Wrapper for VibeVoice Realtime model (0.5B) with simplified interface.

    The streaming model uses pre-computed voice embeddings (.pt files) instead of
    audio samples for low-latency generation.
    """

    def __init__(
        self,
        model_path: str = "vibevoice/VibeVoice-Realtime-0.5B",
        device: Optional[str] = None,
        ddpm_steps: int = 5,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        self.model_path = model_path
        self.device = device
        self.ddpm_steps = ddpm_steps

        print(f"Loading VibeVoice Realtime model on {device}...")
        self._load_model()

    def _load_model(self):
        """Load processor and streaming model"""
        # Load processor
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

        # Determine dtype and attention implementation
        if self.device == "mps":
            load_dtype = torch.float32
            attn_impl = "sdpa"
        elif self.device == "cuda":
            load_dtype = torch.bfloat16
            attn_impl = "flash_attention_2"
        else:  # cpu
            load_dtype = torch.float32
            attn_impl = "sdpa"

        # Load model
        try:
            if self.device == "mps":
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    attn_implementation=attn_impl,
                    device_map=None,
                )
                self.model.to("mps")
            elif self.device == "cuda":
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    attn_implementation=attn_impl,
                    device_map="cuda",
                )
            else:  # cpu
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map="cpu",
                    attn_implementation=attn_impl,
                )
        except Exception as e:
            if attn_impl == "flash_attention_2":
                print(f"Flash attention failed, falling back to SDPA: {e}")
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    attn_implementation="sdpa",
                    device_map=(self.device if self.device in ("cuda", "cpu") else None),
                )
                if self.device == "mps":
                    self.model.to("mps")
            else:
                raise e

        self.model.eval()
        self.model.set_ddpm_inference_steps(num_steps=self.ddpm_steps)

    def set_ddpm_steps(self, num_steps: int):
        """Set the number of DDPM inference steps"""
        self.ddpm_steps = num_steps
        self.model.set_ddpm_inference_steps(num_steps=num_steps)

    def generate(
        self,
        text: str,
        voice_prompt_path: str,
        output_path: str,
        cfg_scale: float = 1.5,
        progress_callback: Optional[callable] = None,
    ) -> GenerationResult:
        """
        Generate speech from text using a pre-computed voice prompt.

        Args:
            text: Input text to synthesize
            voice_prompt_path: Path to the .pt file containing pre-computed voice embeddings
            output_path: Path to save output audio
            cfg_scale: Classifier-free guidance scale
            progress_callback: Optional callback for progress updates

        Returns:
            GenerationResult with metrics
        """
        # Load the pre-computed voice prompt
        all_prefilled_outputs = torch.load(
            voice_prompt_path,
            map_location=self.device,
            weights_only=False
        )

        # Prepare inputs using the streaming processor
        inputs = self.processor.process_input_with_cached_prompt(
            text=text,
            cached_prompt=all_prefilled_outputs,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Move to device
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device)

        # Generate
        start_time = time.time()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=cfg_scale,
            tokenizer=self.processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=True,
            all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs),
            show_progress_bar=progress_callback is not None,
        )
        generation_time = time.time() - start_time

        # Calculate metrics
        sample_rate = 24000
        audio_samples = (
            outputs.speech_outputs[0].shape[-1]
            if len(outputs.speech_outputs[0].shape) > 0
            else len(outputs.speech_outputs[0])
        )
        audio_duration = audio_samples / sample_rate
        rtf = generation_time / audio_duration if audio_duration > 0 else float("inf")

        input_tokens = inputs["tts_text_ids"].shape[1]
        output_tokens = outputs.sequences.shape[1]
        prefill_tokens = all_prefilled_outputs['tts_lm']['last_hidden_state'].size(1)
        generated_tokens = output_tokens - input_tokens - prefill_tokens

        # Save audio
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.processor.save_audio(outputs.speech_outputs[0], output_path=output_path)

        return GenerationResult(
            audio_path=output_path,
            audio_duration=audio_duration,
            generation_time=generation_time,
            rtf=rtf,
            input_tokens=input_tokens,
            generated_tokens=generated_tokens,
        )


def load_vibevoice_model(
    model_path: str,
    device: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    ddpm_steps: int = 5,
) -> Union[VibeVoiceModel, VibeVoiceStreamingModel]:
    """
    Factory function to load the appropriate VibeVoice model based on model type.

    Automatically detects whether the model is a standard VibeVoice or VibeVoice-Realtime
    model and returns the appropriate wrapper.

    Args:
        model_path: Path to the model directory or HuggingFace model ID
        device: Device for inference (cuda, mps, cpu). Auto-detected if None.
        checkpoint_path: Path to LoRA checkpoint (only for standard model)
        ddpm_steps: Number of DDPM inference steps (only for streaming model)

    Returns:
        VibeVoiceModel or VibeVoiceStreamingModel instance
    """
    model_type = detect_model_type(model_path)

    if model_type == VibeVoiceModelType.STREAMING:
        print(f"Detected VibeVoice-Realtime model at {model_path}")
        return VibeVoiceStreamingModel(
            model_path=model_path,
            device=device,
            ddpm_steps=ddpm_steps,
        )
    else:
        print(f"Detected standard VibeVoice model at {model_path}")
        return VibeVoiceModel(
            model_path=model_path,
            device=device,
            checkpoint_path=checkpoint_path,
        )
