"""Data utilities for training"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def prepare_training_data(dataset_path: str) -> str:
    """Convert LJSpeech format dataset to JSONL for VibeVoice training

    Args:
        dataset_path: Path to dataset directory (contains wavs/ and metadata.csv)

    Returns:
        Path to generated metadata.jsonl file
    """
    dataset_dir = Path(dataset_path)
    metadata_csv = dataset_dir / "metadata.csv"
    wavs_dir = dataset_dir / "wavs"

    logger.info(f"Preparing training data from {dataset_path}")

    if not metadata_csv.exists():
        logger.error(f"metadata.csv not found in {dataset_path}")
        raise ValueError(f"metadata.csv not found in {dataset_path}")

    if not wavs_dir.exists():
        logger.error(f"wavs/ directory not found in {dataset_path}")
        raise ValueError(f"wavs/ directory not found in {dataset_path}")

    with open(metadata_csv, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    logger.info(f"Found {len(lines)} samples in metadata.csv")

    def process_line(line: str) -> str:
        parts = line.split("|")
        if len(parts) >= 2:
            filename = parts[0]
            text = parts[1]

            if not filename.endswith(".wav"):
                filename = f"{filename}.wav"

            entry = {"text": f"Speaker 0: {text}", "audio": str(wavs_dir / filename)}
            return json.dumps(entry)
        return None

    num_workers = min(os.cpu_count() or 4, 8)
    logger.info(f"Processing data with {num_workers} workers")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        jsonl_lines = list(filter(None, executor.map(process_line, lines)))

    logger.info(f"Successfully processed {len(jsonl_lines)} samples")

    output_jsonl = dataset_dir / "metadata.jsonl"
    output_jsonl.write_text("\n".join(jsonl_lines) + "\n")

    logger.info(f"Training data written to {output_jsonl}")
    return str(output_jsonl)
