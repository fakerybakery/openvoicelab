"""Core audio segmentation, transcription, and dataset processing."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
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

    def transcribe_and_segment_audio(
        self,
        paths: List[Path],
        model_id: str = "openai/whisper-base",
        device: Optional[str] = None,
        min_segment_duration: float = 3.0,
        max_segment_duration: float = 15.0,
    ) -> List[Dict]:
        """Transcribe audio files using Whisper and segment based on timestamps

        Returns list of segments with audio data and transcripts that can be merged if needed
        """
        if not paths:
            return []

        resolved_device = device or self._resolve_device()
        pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=resolved_device,
            chunk_length_s=30,  # Process in 30-second chunks
            return_timestamps=True,  # Enable long-form transcription with word-level timestamps
        )

        all_segments = []

        total = len(paths)
        for idx, path in enumerate(paths):
            if self.progress_callback:
                progress = idx / total if total > 0 else 0
                self.progress_callback(progress, f"Transcribing and segmenting ({idx+1}/{total})")

            result = pipe(str(path))

            # Load the full audio for this VAD segment
            audio_tensor, sample_rate = torchaudio.load(str(path))

            # Process chunks with timestamps
            if isinstance(result, dict) and "chunks" in result:
                chunks = result["chunks"]

                for chunk in chunks:
                    if "timestamp" in chunk and chunk["timestamp"]:
                        start_time, end_time = chunk["timestamp"]

                        # Handle None timestamps (usually at the end)
                        if start_time is None:
                            start_time = 0.0
                        if end_time is None:
                            end_time = audio_tensor.shape[1] / sample_rate

                        duration = end_time - start_time

                        # Only include segments within our duration range
                        # Very short segments will be merged later
                        if duration > 0.5:  # Filter out very short noise
                            start_sample = int(start_time * sample_rate)
                            end_sample = int(end_time * sample_rate)

                            segment_audio = audio_tensor[:, start_sample:end_sample]

                            all_segments.append({
                                "audio": segment_audio,
                                "sample_rate": sample_rate,
                                "text": chunk["text"].strip(),
                                "duration": duration,
                                "source_file": path.stem,
                            })
            elif isinstance(result, dict) and "text" in result:
                # Fallback: no chunks, use whole file
                duration = audio_tensor.shape[1] / sample_rate
                all_segments.append({
                    "audio": audio_tensor,
                    "sample_rate": sample_rate,
                    "text": result["text"].strip(),
                    "duration": duration,
                    "source_file": path.stem,
                })

        # Now merge segments that are too short
        merged_segments = self._merge_short_segments(all_segments, min_segment_duration, max_segment_duration)

        return merged_segments

    def _merge_short_segments(
        self,
        segments: List[Dict],
        min_duration: float,
        max_duration: float
    ) -> List[Dict]:
        """Merge segments that are too short, respecting max duration"""
        if not segments:
            return []

        merged = []
        current_batch = []
        current_duration = 0.0

        for segment in segments:
            duration = segment["duration"]

            # If this segment alone is already good, add any pending batch first
            if duration >= min_duration:
                # Flush current batch if exists
                if current_batch:
                    merged.append(self._combine_segments(current_batch))
                    current_batch = []
                    current_duration = 0.0

                # Add this segment directly (might need to split if too long)
                if duration <= max_duration:
                    merged.append(segment)
                else:
                    # Split long segment into chunks
                    merged.extend(self._split_long_segment(segment, max_duration))
            else:
                # Segment is too short, add to batch
                if current_duration + duration <= max_duration:
                    current_batch.append(segment)
                    current_duration += duration

                    # If batch is now long enough, flush it
                    if current_duration >= min_duration:
                        merged.append(self._combine_segments(current_batch))
                        current_batch = []
                        current_duration = 0.0
                else:
                    # Adding this would exceed max, flush current batch first
                    if current_batch:
                        merged.append(self._combine_segments(current_batch))
                    current_batch = [segment]
                    current_duration = duration

        # Flush remaining batch
        if current_batch:
            merged.append(self._combine_segments(current_batch))

        return merged

    def _combine_segments(self, segments: List[Dict]) -> Dict:
        """Combine multiple segments into one"""
        if len(segments) == 1:
            return segments[0]

        # Combine audio with small silence between
        sample_rate = segments[0]["sample_rate"]
        silence_samples = int(0.2 * sample_rate)  # 0.2s silence
        silence = torch.zeros(1, silence_samples)

        combined_audio_parts = []
        combined_texts = []
        total_duration = 0.0

        for i, seg in enumerate(segments):
            combined_audio_parts.append(seg["audio"])
            combined_texts.append(seg["text"])
            total_duration += seg["duration"]

            # Add silence between segments (but not after the last one)
            if i < len(segments) - 1:
                combined_audio_parts.append(silence)
                total_duration += 0.2

        combined_audio = torch.cat(combined_audio_parts, dim=1)

        return {
            "audio": combined_audio,
            "sample_rate": sample_rate,
            "text": " ".join(combined_texts),
            "duration": total_duration,
            "source_file": f"{segments[0]['source_file']}_merged",
        }

    def _split_long_segment(self, segment: Dict, max_duration: float) -> List[Dict]:
        """Split a segment that's too long into smaller chunks"""
        duration = segment["duration"]
        if duration <= max_duration:
            return [segment]

        # Calculate how many chunks we need
        num_chunks = int(np.ceil(duration / max_duration))
        sample_rate = segment["sample_rate"]
        audio = segment["audio"]
        total_samples = audio.shape[1]
        samples_per_chunk = total_samples // num_chunks

        chunks = []
        for i in range(num_chunks):
            start_sample = i * samples_per_chunk
            end_sample = (i + 1) * samples_per_chunk if i < num_chunks - 1 else total_samples

            chunk_audio = audio[:, start_sample:end_sample]
            chunk_duration = chunk_audio.shape[1] / sample_rate

            chunks.append({
                "audio": chunk_audio,
                "sample_rate": sample_rate,
                "text": segment["text"],  # Same text for all chunks (not ideal but simple)
                "duration": chunk_duration,
                "source_file": f"{segment['source_file']}_split{i}",
            })

        return chunks

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
        min_segment_duration: float = 7.0,
        max_segment_duration: float = 20.0,
    ):
        """Process audio files into LJSpeech format dataset using Whisper timestamp-based segmentation"""
        self.processing = True
        self.progress_callback = progress_callback

        try:
            input_path = Path(input_dir)
            if not input_path.exists():
                raise ValueError(f"Input directory not found: {input_dir}")

            # Create dataset directory
            dataset_dir = self.output_dir / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # Create work directory for VAD segments
            work_dir = dataset_dir / "_work"
            vad_segments_dir = work_dir / "vad_segments"

            # Step 1: VAD Segmentation (coarse speech detection)
            if progress_callback:
                progress_callback(0, "Running VAD segmentation...")

            vad_segments = self.segment_audio(input_path, vad_segments_dir)

            if not vad_segments:
                raise ValueError("No audio segments found after VAD processing")

            # Step 2: Transcribe with Whisper and create fine-grained segments
            if progress_callback:
                progress_callback(0.3, "Transcribing with Whisper...")

            segments = self.transcribe_and_segment_audio(
                vad_segments,
                model_id=whisper_model,
                device=device,
                min_segment_duration=min_segment_duration,
                max_segment_duration=max_segment_duration,
            )

            if not segments:
                raise ValueError("No segments created after transcription")

            # Step 3: Save segments to wavs/ directory
            if progress_callback:
                progress_callback(0.9, "Saving dataset...")

            wavs_dir = dataset_dir / "wavs"
            wavs_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = dataset_dir / "metadata.csv"

            metadata_lines = []
            total_duration = 0.0

            for idx, segment in enumerate(segments):
                # Save audio file
                filename = f"{dataset_name}_{idx:06d}.wav"
                wav_path = wavs_dir / filename

                torchaudio.save(
                    str(wav_path),
                    segment["audio"],
                    segment["sample_rate"]
                )

                # Add metadata line: filename|text|text
                metadata_lines.append(f"{filename}|{segment['text']}|{segment['text']}")
                total_duration += segment["duration"]

            # Write metadata.csv
            metadata_path.write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")

            # Save dataset info
            info = {
                "name": dataset_name,
                "created_at": datetime.now().isoformat(),
                "num_samples": len(segments),
                "total_duration": total_duration,
                "whisper_model": whisper_model,
                "input_dir": str(input_dir),
                "min_segment_duration": min_segment_duration,
                "max_segment_duration": max_segment_duration,
            }
            (dataset_dir / "info.json").write_text(json.dumps(info, indent=2))

            if progress_callback:
                progress_callback(1.0, f"Dataset created: {len(segments)} samples")

        finally:
            self.processing = False
            self.progress_callback = None

    def process_dataset_async(self, *args, **kwargs):
        """Process dataset in background thread"""
        thread = threading.Thread(target=self.process_dataset, args=args, kwargs=kwargs)
        thread.daemon = True
        thread.start()
        return thread

    def _calculate_dataset_duration(self, dataset_dir: Path) -> float:
        """Calculate total duration of all audio files in dataset (in seconds)"""
        wavs_dir = dataset_dir / "wavs"
        if not wavs_dir.exists():
            return 0.0

        total_duration = 0.0
        for wav_file in wavs_dir.glob("*.wav"):
            try:
                metadata = torchaudio.info(str(wav_file))
                duration = metadata.num_frames / metadata.sample_rate
                total_duration += duration
            except Exception:
                # Skip files that can't be read
                continue

        return total_duration

    def list_datasets(self) -> List[Dict]:
        """List all created datasets with duration information"""
        datasets = []
        for dataset_dir in self.output_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            info_path = dataset_dir / "info.json"
            if info_path.exists():
                info = json.loads(info_path.read_text())

                # Calculate total duration if not already cached
                if "total_duration" not in info:
                    info["total_duration"] = self._calculate_dataset_duration(dataset_dir)
                    # Update info.json with duration
                    try:
                        info_path.write_text(json.dumps(info, indent=2))
                    except Exception:
                        pass  # Continue even if we can't write

                datasets.append(info)

        return sorted(datasets, key=lambda x: x.get("created_at", ""), reverse=True)
