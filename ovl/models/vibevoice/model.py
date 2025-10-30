import os
import time
from typing import List, Optional

import torch
from vibevoice.modular.lora_loading import load_lora_assets
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

from ovl.models.base import GenerationResult, TTSModel


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
                    device_map="cuda",
                    attn_implementation=attn_impl,
                )
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
                    device_map=(self.device if self.device in ("cuda", "cpu") else None),
                    attn_implementation="sdpa",
                )
                if self.device == "mps":
                    self.model.to("mps")
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
