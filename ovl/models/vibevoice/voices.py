import os
from typing import Dict, List


class VoiceMapper:
    """Maps speaker names to voice file paths from voices/ folder"""

    def __init__(self, voices_dir: str = "voices"):
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
