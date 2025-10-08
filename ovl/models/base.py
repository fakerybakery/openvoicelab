"""Base classes and shared components for TTS models"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GenerationResult:
    """Result from TTS model generation"""

    audio_path: str
    audio_duration: float
    generation_time: float
    rtf: float  # Real-time factor
    input_tokens: int
    generated_tokens: int


class TTSModel(ABC):
    """Abstract base class for TTS models"""

    @abstractmethod
    def generate(self, text: str, voice_samples: List[str], output_path: str, **kwargs) -> GenerationResult:
        """
        Generate speech from text using voice samples

        Args:
            text: Input text to synthesize
            voice_samples: List of voice sample file paths
            output_path: Path to save output audio
            **kwargs: Model-specific generation parameters

        Returns:
            GenerationResult with metrics
        """
        pass
