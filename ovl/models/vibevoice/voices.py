import os
from enum import Enum
from typing import Dict, List, Optional


# Get package directory for default voice paths
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_VOICES_DIR = os.path.join(_PACKAGE_DIR, "voices")
_DEFAULT_STREAMING_VOICES_DIR = os.path.join(_PACKAGE_DIR, "voices", "streaming")


class VoiceType(Enum):
    """Type of voice file"""
    AUDIO = "audio"  # .wav files for standard model
    CACHED_PROMPT = "cached_prompt"  # .pt files for streaming model


class VoiceMapper:
    """Maps speaker names to voice file paths from voices/ folder"""

    def __init__(self, voices_dir: Optional[str] = None):
        if voices_dir is None:
            voices_dir = _DEFAULT_VOICES_DIR
        self.voices_dir = os.path.abspath(voices_dir)
        self.voice_presets: Dict[str, str] = {}
        self.refresh()

    def refresh(self):
        """Scan voices directory and update available voices"""
        if not os.path.exists(self.voices_dir):
            os.makedirs(self.voices_dir, exist_ok=True)
            print(f"Created voices directory at {self.voices_dir}")
            return

        # Get all .wav files
        wav_files = [
            f
            for f in os.listdir(self.voices_dir)
            if f.lower().endswith(".wav") and os.path.isfile(os.path.join(self.voices_dir, f))
        ]

        self.voice_presets = {}
        for wav_file in wav_files:
            name = os.path.splitext(wav_file)[0]
            full_path = os.path.join(self.voices_dir, wav_file)
            self.voice_presets[name] = full_path

        # Sort alphabetically
        self.voice_presets = dict(sorted(self.voice_presets.items()))

    def get_voice_path(self, speaker_name: str) -> str:
        """Get voice file path for a given speaker name"""
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]

        # Try case-insensitive match
        speaker_lower = speaker_name.lower()
        for preset_name, path in self.voice_presets.items():
            if preset_name.lower() == speaker_lower:
                return path

        raise ValueError(f"Voice '{speaker_name}' not found in {self.voices_dir}")

    def list_voices(self) -> List[str]:
        """Return list of available voice names"""
        return list(self.voice_presets.keys())

    def add_voice(self, name: str, audio_file_path: str) -> str:
        """
        Add a new voice to the voices directory

        Args:
            name: Name for the voice (without extension)
            audio_file_path: Path to the audio file to copy

        Returns:
            Path to the new voice file
        """
        import shutil

        # Clean the name
        name = name.replace(" ", "_")
        dest_path = os.path.join(self.voices_dir, f"{name}.wav")

        # Copy the file
        shutil.copy(audio_file_path, dest_path)

        # Refresh the voice list
        self.refresh()

        return dest_path

    def delete_voice(self, name: str):
        """Delete a voice from the voices directory"""
        if name in self.voice_presets:
            os.remove(self.voice_presets[name])
            self.refresh()
        else:
            raise ValueError(f"Voice '{name}' not found")


class StreamingVoiceMapper:
    """Maps speaker names to pre-computed voice prompt files (.pt) for the streaming model.

    The streaming model uses pre-computed voice embeddings instead of audio samples
    for low-latency generation.
    """

    def __init__(self, voices_dir: Optional[str] = None):
        if voices_dir is None:
            voices_dir = _DEFAULT_STREAMING_VOICES_DIR
        self.voices_dir = os.path.abspath(voices_dir)
        self.voice_presets: Dict[str, str] = {}
        self.refresh()

    def refresh(self):
        """Scan voices directory and update available streaming voices"""
        if not os.path.exists(self.voices_dir):
            os.makedirs(self.voices_dir, exist_ok=True)
            print(f"Created streaming voices directory at {self.voices_dir}")
            return

        # Get all .pt files
        pt_files = [
            f
            for f in os.listdir(self.voices_dir)
            if f.lower().endswith(".pt") and os.path.isfile(os.path.join(self.voices_dir, f))
        ]

        self.voice_presets = {}
        for pt_file in pt_files:
            name = os.path.splitext(pt_file)[0]
            full_path = os.path.join(self.voices_dir, pt_file)
            self.voice_presets[name] = full_path

        # Create aliases without language prefix (e.g., "en-Emma" -> "Emma")
        aliases = {}
        for name, path in self.voice_presets.items():
            # Remove gender suffix (e.g., "en-Emma_woman" -> "en-Emma")
            if '_' in name:
                base_name = name.split('_')[0]
                aliases[base_name] = path

            # Remove language prefix (e.g., "en-Emma" -> "Emma")
            if '-' in name:
                short_name = name.split('-')[-1]
                aliases[short_name] = path

        self.voice_presets.update(aliases)

        # Sort alphabetically
        self.voice_presets = dict(sorted(self.voice_presets.items()))

    def get_voice_path(self, speaker_name: str) -> str:
        """Get voice prompt file path for a given speaker name"""
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]

        # Try case-insensitive match
        speaker_lower = speaker_name.lower()
        for preset_name, path in self.voice_presets.items():
            if preset_name.lower() == speaker_lower:
                return path

        # Try partial match
        for preset_name, path in self.voice_presets.items():
            if preset_name.lower() in speaker_lower or speaker_lower in preset_name.lower():
                return path

        raise ValueError(f"Streaming voice '{speaker_name}' not found in {self.voices_dir}")

    def list_voices(self) -> List[str]:
        """Return list of available streaming voice names"""
        # Filter to return only the shortest/cleanest names (avoid duplicates)
        unique_paths = {}
        for name, path in self.voice_presets.items():
            if path not in unique_paths or len(name) < len(unique_paths[path]):
                unique_paths[path] = name
        return sorted(unique_paths.values())

    def delete_voice(self, name: str):
        """Delete a streaming voice from the voices directory"""
        if name in self.voice_presets:
            os.remove(self.voice_presets[name])
            self.refresh()
        else:
            raise ValueError(f"Streaming voice '{name}' not found")


class UnifiedVoiceMapper:
    """Unified voice mapper that supports both standard and streaming voices.

    Automatically detects voice type based on file extension.
    """

    def __init__(
        self,
        standard_voices_dir: Optional[str] = None,
        streaming_voices_dir: Optional[str] = None,
    ):
        self.standard_mapper = VoiceMapper(standard_voices_dir)
        self.streaming_mapper = StreamingVoiceMapper(streaming_voices_dir)

    def refresh(self):
        """Refresh both voice mappers"""
        self.standard_mapper.refresh()
        self.streaming_mapper.refresh()

    def list_standard_voices(self) -> List[str]:
        """Return list of standard voice names (audio files)"""
        return self.standard_mapper.list_voices()

    def list_streaming_voices(self) -> List[str]:
        """Return list of streaming voice names (cached prompts)"""
        return self.streaming_mapper.list_voices()

    def get_standard_voice_path(self, speaker_name: str) -> str:
        """Get audio file path for standard model"""
        return self.standard_mapper.get_voice_path(speaker_name)

    def get_streaming_voice_path(self, speaker_name: str) -> str:
        """Get cached prompt path for streaming model"""
        return self.streaming_mapper.get_voice_path(speaker_name)
