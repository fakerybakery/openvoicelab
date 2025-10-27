"""Data utilities for training"""

import json
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds"""
    try:
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        return len(audio) / sr
    except Exception as e:
        logger.warning(f"Failed to get duration for {audio_path}: {e}")
        return 0.0


def merge_audio_files(audio_paths: List[str], output_path: str, target_sr: int = 24000) -> None:
    """Merge multiple audio files into one"""
    merged_audio = []

    for audio_path in audio_paths:
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            merged_audio.append(audio)
            # Add small silence between segments (0.2 seconds)
            silence = np.zeros(int(0.2 * target_sr), dtype=audio.dtype)
            merged_audio.append(silence)
        except Exception as e:
            logger.warning(f"Failed to load {audio_path}: {e}")
            continue

    if merged_audio:
        # Remove last silence
        if len(merged_audio) > 1:
            merged_audio = merged_audio[:-1]

        final_audio = np.concatenate(merged_audio)
        sf.write(output_path, final_audio, target_sr)
        logger.info(f"Merged {len(audio_paths)} segments into {output_path} ({len(final_audio)/target_sr:.2f}s)")


def prepare_training_data(dataset_path: str, min_duration: float = 10.0) -> str:
    """Convert LJSpeech format dataset to JSONL for VibeVoice training

    Args:
        dataset_path: Path to dataset directory (contains wavs/ and metadata.csv)
        min_duration: Minimum duration in seconds for each sample. Shorter samples will be merged.

    Returns:
        Path to generated metadata.jsonl file
    """
    dataset_dir = Path(dataset_path)
    metadata_csv = dataset_dir / "metadata.csv"
    wavs_dir = dataset_dir / "wavs"
    merged_wavs_dir = dataset_dir / "wavs_merged"

    logger.info(f"Preparing training data from {dataset_path}")
    logger.info(f"Minimum duration set to {min_duration} seconds")

    if not metadata_csv.exists():
        logger.error(f"metadata.csv not found in {dataset_path}")
        raise ValueError(f"metadata.csv not found in {dataset_path}")

    if not wavs_dir.exists():
        logger.error(f"wavs/ directory not found in {dataset_path}")
        raise ValueError(f"wavs/ directory not found in {dataset_path}")

    with open(metadata_csv, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    logger.info(f"Found {len(lines)} samples in metadata.csv")

    # Parse all entries
    entries = []
    for line in lines:
        parts = line.split("|")
        if len(parts) >= 2:
            filename = parts[0]
            text = parts[1]

            if not filename.endswith(".wav"):
                filename = f"{filename}.wav"

            audio_path = wavs_dir / filename
            if audio_path.exists():
                duration = get_audio_duration(str(audio_path))
                entries.append({
                    "filename": filename,
                    "text": text,
                    "audio_path": str(audio_path),
                    "duration": duration
                })

    logger.info(f"Parsed {len(entries)} valid entries")

    # Separate short and long segments
    short_segments = [e for e in entries if e["duration"] < min_duration and e["duration"] > 0]
    long_segments = [e for e in entries if e["duration"] >= min_duration]

    logger.info(f"Found {len(short_segments)} short segments (< {min_duration}s)")
    logger.info(f"Found {len(long_segments)} long segments (>= {min_duration}s)")

    jsonl_entries = []

    # Add long segments directly
    for entry in long_segments:
        jsonl_entries.append({
            "text": f"Speaker 0: {entry['text']}",
            "audio": entry["audio_path"]
        })

    # Merge short segments
    if short_segments:
        merged_wavs_dir.mkdir(exist_ok=True)
        logger.info(f"Merging {len(short_segments)} short segments...")

        current_batch = []
        current_duration = 0.0
        current_texts = []
        merged_count = 0

        for entry in short_segments:
            current_batch.append(entry["audio_path"])
            current_texts.append(entry["text"])
            current_duration += entry["duration"]

            # If we've accumulated enough duration, merge this batch
            if current_duration >= min_duration:
                merged_count += 1
                merged_filename = f"merged_{merged_count:04d}.wav"
                merged_path = merged_wavs_dir / merged_filename

                merge_audio_files(current_batch, str(merged_path))

                # Combine texts
                combined_text = " ".join(current_texts)

                jsonl_entries.append({
                    "text": f"Speaker 0: {combined_text}",
                    "audio": str(merged_path)
                })

                # Reset batch
                current_batch = []
                current_texts = []
                current_duration = 0.0

        # Handle remaining segments
        if current_batch:
            merged_count += 1
            merged_filename = f"merged_{merged_count:04d}.wav"
            merged_path = merged_wavs_dir / merged_filename

            merge_audio_files(current_batch, str(merged_path))

            combined_text = " ".join(current_texts)

            jsonl_entries.append({
                "text": f"Speaker 0: {combined_text}",
                "audio": str(merged_path)
            })

        logger.info(f"Created {merged_count} merged segments")

    logger.info(f"Total final samples: {len(jsonl_entries)}")

    # Write JSONL
    jsonl_lines = [json.dumps(entry) for entry in jsonl_entries]

    output_jsonl = dataset_dir / "metadata.jsonl"
    output_jsonl.write_text("\n".join(jsonl_lines) + "\n")

    logger.info(f"Training data written to {output_jsonl}")
    return str(output_jsonl)
