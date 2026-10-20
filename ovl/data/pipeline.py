"""Turn a folder of recordings into an LJSpeech-style training dataset.

1. Silero VAD finds the speech regions of each file.
2. Whisper transcribes each region with timestamps, giving sentence-level chunks.
3. Chunks are merged/split to land within [min, max] seconds and saved as wavs/ + metadata.csv.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from ovl import paths
from ovl.audio import duration as audio_duration
from ovl.audio import load_audio, resample, save_audio

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}
VAD_SR = 16_000
GAP_SECONDS = 0.2


class DatasetBuilder:
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else paths.data_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _audio_files(input_dir: Path) -> List[Path]:
        return sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)

    def _segments(self, path: Path, vad, pipe, language: Optional[str] = None) -> List[Dict]:
        """Sentence-level segments of one file: [{"audio", "sample_rate", "text", "duration"}]."""
        import torch
        from silero_vad import get_speech_timestamps

        wav, sr = load_audio(str(path))
        wav16 = resample(wav, sr, VAD_SR)
        regions = get_speech_timestamps(torch.from_numpy(wav16), vad, sampling_rate=VAD_SR)

        segments = []
        for region in regions:
            start16, end16 = region["start"], region["end"]
            result = pipe(
                {"raw": wav16[start16:end16], "sampling_rate": VAD_SR},
                chunk_length_s=30,
                return_timestamps=True,
                **({"generate_kwargs": {"language": language}} if language else {}),
            )
            offset = start16 / VAD_SR
            region_end = end16 / VAD_SR
            chunks = result.get("chunks") or [{"timestamp": (0.0, None), "text": result.get("text", "")}]
            for chunk in chunks:
                text = (chunk.get("text") or "").strip()
                t0, t1 = chunk.get("timestamp") or (0.0, None)
                t0 = offset + (t0 or 0.0)
                t1 = min(region_end, offset + t1) if t1 is not None else region_end
                if not text or t1 - t0 <= 0.5:
                    continue
                audio = wav[int(t0 * sr) : int(t1 * sr)]
                segments.append({"audio": audio, "sample_rate": sr, "text": text, "duration": len(audio) / sr})
        return segments

    @staticmethod
    def _combine(batch: List[Dict]) -> Dict:
        if len(batch) == 1:
            return batch[0]
        sr = batch[0]["sample_rate"]
        gap = np.zeros(int(GAP_SECONDS * sr), dtype=np.float32)
        parts = []
        for i, seg in enumerate(batch):
            parts.append(seg["audio"])
            if i < len(batch) - 1:
                parts.append(gap)
        audio = np.concatenate(parts)
        return {"audio": audio, "sample_rate": sr, "text": " ".join(s["text"] for s in batch), "duration": len(audio) / sr}

    @staticmethod
    def _split(segment: Dict, max_duration: float) -> List[Dict]:
        """Split an over-long segment evenly, dividing the words proportionally."""
        n = int(np.ceil(segment["duration"] / max_duration))
        audio, sr = segment["audio"], segment["sample_rate"]
        words = segment["text"].split()
        out = []
        for i in range(n):
            piece = audio[i * len(audio) // n : (i + 1) * len(audio) // n]
            text = " ".join(words[i * len(words) // n : (i + 1) * len(words) // n])
            if text:
                out.append({"audio": piece, "sample_rate": sr, "text": text, "duration": len(piece) / sr})
        return out

    def _merge(self, segments: List[Dict], min_duration: float, max_duration: float) -> List[Dict]:
        merged, batch, batch_duration = [], [], 0.0
        for seg in segments:
            if seg["duration"] > max_duration:
                if batch:
                    merged.append(self._combine(batch))
                    batch, batch_duration = [], 0.0
                merged.extend(self._split(seg, max_duration))
                continue
            if batch and batch_duration + GAP_SECONDS + seg["duration"] > max_duration:
                merged.append(self._combine(batch))
                batch, batch_duration = [], 0.0
            batch.append(seg)
            batch_duration += seg["duration"] + (GAP_SECONDS if len(batch) > 1 else 0)
            if batch_duration >= min_duration:
                merged.append(self._combine(batch))
                batch, batch_duration = [], 0.0
        if batch:
            merged.append(self._combine(batch))
        return merged

    def process_dataset(
        self,
        input_dir: str,
        dataset_name: str,
        whisper_model: str = "openai/whisper-large-v3-turbo",
        progress_callback: Optional[Callable[[float, str], None]] = None,
        min_segment_duration: float = 7.0,
        max_segment_duration: float = 20.0,
        language: Optional[str] = None,
    ) -> Dict:
        from silero_vad import load_silero_vad

        from ovl import asr

        progress = progress_callback or (lambda *_: None)
        input_path = Path(input_dir).expanduser()
        if not input_path.is_dir():
            raise ValueError(f"Input directory not found: {input_dir}")
        files = self._audio_files(input_path)
        if not files:
            raise ValueError(f"No audio files ({', '.join(sorted(SUPPORTED_EXTENSIONS))}) found in {input_dir}")

        progress(0, "Loading VAD and Whisper...")
        vad = load_silero_vad()
        pipe = asr.get_pipeline(whisper_model)

        segments = []
        try:
            for i, path in enumerate(files):
                progress(0.9 * i / len(files), f"Transcribing {path.name} ({i + 1}/{len(files)})")
                segments.extend(self._merge(self._segments(path, vad, pipe, language), min_segment_duration, max_segment_duration))
        finally:
            asr.unload()
        if not segments:
            raise ValueError("No speech found in the input files")

        progress(0.9, "Saving dataset...")
        dataset_dir = self.output_dir / dataset_name
        wavs_dir = dataset_dir / "wavs"
        wavs_dir.mkdir(parents=True, exist_ok=True)
        lines, total = [], 0.0
        for idx, seg in enumerate(segments):
            filename = f"{dataset_name}_{idx:06d}.wav"
            save_audio(wavs_dir / filename, seg["audio"], seg["sample_rate"])
            text = seg["text"].replace("|", " ").replace("\n", " ")
            lines.append(f"{filename}|{text}|{text}")
            total += seg["duration"]
        (dataset_dir / "metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

        info = {
            "name": dataset_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "num_samples": len(segments),
            "total_duration": total,
            "whisper_model": whisper_model,
            "input_dir": str(input_path),
            "min_segment_duration": min_segment_duration,
            "max_segment_duration": max_segment_duration,
        }
        (dataset_dir / "info.json").write_text(json.dumps(info, indent=2))
        progress(1.0, f"Dataset created: {len(segments)} samples")
        return info

    def list_datasets(self) -> List[Dict]:
        datasets = []
        for dataset_dir in self.output_dir.iterdir():
            info_path = dataset_dir / "info.json"
            if not info_path.exists():
                continue
            info = json.loads(info_path.read_text())
            if not info.get("total_duration"):
                info["total_duration"] = sum(audio_duration(str(p)) for p in (dataset_dir / "wavs").glob("*.wav"))
                info_path.write_text(json.dumps(info, indent=2))
            info["path"] = str(dataset_dir)
            datasets.append(info)
        return sorted(datasets, key=lambda x: x.get("created_at", ""), reverse=True)
