"""Core audio segmentation, transcription, and dataset processing."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torchaudio
from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
from transformers import pipeline as hf_pipeline

_SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}


class DatasetBuilder:
    """Handles audio dataset creation with segmentation and transcription"""

    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processing = False
        self.current_job = None
        self.progress_callback = None

    def _iter_audio_files(self, input_dir: Path) -> List[Path]:
        """Get all audio files from input directory"""
        files = []
        for path in sorted(input_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in _SUPPORTED_EXTENSIONS:
                files.append(path)
        return files

    def _resolve_device(self) -> str:
        """Auto-detect best available device"""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def segment_audio(self, input_dir: Path, segments_dir: Path, vad_sampling_rate: int = 16_000) -> List[Path]:
        """Split audio files into voice segments using Silero VAD"""
        segments_dir.mkdir(parents=True, exist_ok=True)
        model = load_silero_vad()
        created_segments: List[Path] = []

        audio_files = self._iter_audio_files(input_dir)
        total_files = len(audio_files)

        for idx, audio_path in enumerate(audio_files):
            if self.progress_callback:
                progress = idx / total_files if total_files > 0 else 0
                self.progress_callback(progress, f"Segmenting audio ({idx+1}/{total_files})")

            wav = read_audio(str(audio_path), sampling_rate=vad_sampling_rate)
            timestamps = get_speech_timestamps(wav, model, sampling_rate=vad_sampling_rate)

            if not timestamps:
                continue

            audio_tensor, sample_rate = torchaudio.load(str(audio_path))
            for seg_idx, segment in enumerate(timestamps, start=1):
                start = int(segment["start"] * sample_rate / vad_sampling_rate)
                end = int(segment["end"] * sample_rate / vad_sampling_rate)
                if end <= start:
                    continue

                segment_tensor = audio_tensor[:, start:end]
                output_path = segments_dir / f"{audio_path.stem}_{seg_idx:03d}.wav"
                torchaudio.save(str(output_path), segment_tensor, sample_rate)
                created_segments.append(output_path)

        return created_segments

    def transcribe_audio(
        self,
        paths: List[Path],
        model_id: str = "openai/whisper-base",
        device: Optional[str] = None,
    ) -> Dict[Path, str]:
        """Transcribe audio files using Whisper"""
        if not paths:
            return {}

        resolved_device = device or self._resolve_device()
        pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=resolved_device,
            chunk_length_s=30,  # Process in 30-second chunks
            return_timestamps=True,  # Enable long-form transcription
        )
        transcripts: Dict[Path, str] = {}

        total = len(paths)
        for idx, path in enumerate(paths):
            if self.progress_callback:
                progress = idx / total if total > 0 else 0
                self.progress_callback(progress, f"Transcribing ({idx+1}/{total})")

            result = pipe(str(path))
            # Extract just the text, not timestamps
            if isinstance(result, dict) and "text" in result:
                transcripts[path] = result["text"].strip()
            elif isinstance(result, dict) and "chunks" in result:
                # Concatenate all chunks
                text = " ".join(chunk["text"].strip() for chunk in result["chunks"])
                transcripts[path] = text.strip()
            else:
                transcripts[path] = str(result).strip()

        return transcripts

    def save_ljspeech_format(
        self,
        transcripts: Dict[Path, str],
        dataset_name: str,
        wavs_dir: Path,
        metadata_path: Path,
    ):
        """Save dataset in LJSpeech format"""
        # Create wavs directory
        wavs_dir.mkdir(parents=True, exist_ok=True)

        # Build metadata
        metadata_lines = []
        for idx, (audio_path, text) in enumerate(sorted(transcripts.items(), key=lambda x: str(x[0]))):
            # Copy wav file to wavs directory
            new_name = f"{dataset_name}_{idx:06d}.wav"
            dest_path = wavs_dir / new_name

            # Copy the file
            import shutil

            shutil.copy(audio_path, dest_path)

            # Add metadata line: filename|text|text
            metadata_lines.append(f"{new_name}|{text}|{text}")

        # Write metadata.csv
        metadata_path.write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")

    def process_dataset(
        self,
        input_dir: str,
        dataset_name: str,
        whisper_model: str = "openai/whisper-base",
        device: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ):
        """Process audio files into LJSpeech format dataset"""
        self.processing = True
        self.progress_callback = progress_callback

        try:
            input_path = Path(input_dir)
            if not input_path.exists():
                raise ValueError(f"Input directory not found: {input_dir}")

            # Create dataset directory
            dataset_dir = self.output_dir / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # Create work directory for segments
            work_dir = dataset_dir / "_work"
            segments_dir = work_dir / "segments"

            # Step 1: Segment audio
            if progress_callback:
                progress_callback(0, "Starting segmentation...")

            segments = self.segment_audio(input_path, segments_dir)

            if not segments:
                raise ValueError("No audio segments found after VAD processing")

            # Step 2: Transcribe
            if progress_callback:
                progress_callback(0.5, "Starting transcription...")

            transcripts = self.transcribe_audio(segments, model_id=whisper_model, device=device)

            # Step 3: Save in LJSpeech format
            if progress_callback:
                progress_callback(0.9, "Saving dataset...")

            wavs_dir = dataset_dir / "wavs"
            metadata_path = dataset_dir / "metadata.csv"

            self.save_ljspeech_format(transcripts, dataset_name, wavs_dir, metadata_path)

            # Save dataset info
            info = {
                "name": dataset_name,
                "created_at": datetime.now().isoformat(),
                "num_samples": len(transcripts),
                "whisper_model": whisper_model,
                "input_dir": str(input_dir),
            }
            (dataset_dir / "info.json").write_text(json.dumps(info, indent=2))

            if progress_callback:
                progress_callback(1.0, f"Dataset created: {len(transcripts)} samples")

        finally:
            self.processing = False
            self.progress_callback = None

    def process_dataset_async(self, *args, **kwargs):
        """Process dataset in background thread"""
        thread = threading.Thread(target=self.process_dataset, args=args, kwargs=kwargs)
        thread.daemon = True
        thread.start()
        return thread

    def list_datasets(self) -> List[Dict]:
        """List all created datasets"""
        datasets = []
        for dataset_dir in self.output_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            info_path = dataset_dir / "info.json"
            if info_path.exists():
                info = json.loads(info_path.read_text())
                datasets.append(info)

        return sorted(datasets, key=lambda x: x.get("created_at", ""), reverse=True)
