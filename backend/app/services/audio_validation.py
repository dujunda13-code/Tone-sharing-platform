import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf

from backend.app.schemas.dataset import AudioInspection, DatasetManifest, DatasetManifestRow


DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_SNR_WARNING_DB = 20.0
DEFAULT_MAX_CLIPPING_RATIO = 0.01
SNR_BELOW_RECOMMENDED = "SNR_BELOW_RECOMMENDED"
DEFAULT_SEGMENT_MIN_SECONDS = 3.0
DEFAULT_SEGMENT_MAX_SECONDS = 12.0
DEFAULT_MAX_SPEECH_MERGE_GAP_SECONDS = 0.75
DATASET_DURATION_MIN = 480.0
DATASET_DURATION_MAX = 720.0
REFERENCE_SECONDS_MIN = 3.0
REFERENCE_SECONDS_MAX = 10.0
REFERENCE_SPLIT = "reference"
SPLIT_SEED = 20260902


class UnsupportedAudioFormat(ValueError):
    """Raised when an upload is not a supported audio format."""


class DatasetDurationOutOfRange(ValueError):
    """Raised when effective speech is not within the training contract."""


class ReferenceDurationOutOfRange(ValueError):
    """Raised when effective speech is not within the 3-10s reference contract."""


class LocalAudioPathError(ValueError):
    """Raised when a manifest audio path is not a local storage file."""


def resolve_local_audio_path(path: str | Path, storage_root: Path | str) -> Path:
    """Resolve a manifest path and keep it inside the local storage root."""
    raw_path = str(path)
    candidate_path = Path(raw_path)
    if candidate_path.is_absolute() or candidate_path.drive:
        raise LocalAudioPathError("manifest audio path must be relative to local storage")
    root = Path(storage_root).expanduser().resolve()
    resolved = (root / candidate_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise LocalAudioPathError("manifest audio path escapes local storage") from exc
    if not resolved.is_file():
        raise LocalAudioPathError(f"manifest audio file is missing from local storage: {raw_path}")
    return resolved


def _load_audio(path: Path, target_sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    try:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        samples = audio.T
    except RuntimeError:
        try:
            samples, sample_rate = librosa.load(path, sr=None, mono=False)
            samples = np.atleast_2d(samples).astype(np.float32, copy=False)
        except Exception as load_exc:
            raise UnsupportedAudioFormat(path.suffix.lower()) from load_exc

    if target_sample_rate is not None and sample_rate != target_sample_rate:
        samples = np.vstack(
            [
                librosa.resample(
                    channel,
                    orig_sr=sample_rate,
                    target_sr=target_sample_rate,
                )
                for channel in samples
            ]
        ).astype(np.float32, copy=False)
        sample_rate = target_sample_rate
    return np.clip(samples, -1.0, 1.0), int(sample_rate)


def _estimate_snr(samples: np.ndarray) -> float:
    mono = np.mean(samples, axis=0)
    signal_rms = float(np.sqrt(np.mean(np.square(mono))))
    if signal_rms <= 1e-12:
        return 0.0
    difference = np.diff(mono)
    noise_rms = float(np.std(difference) / math.sqrt(2.0))
    if noise_rms <= 1e-12:
        return 100.0
    return float(20.0 * np.log10(signal_rms / noise_rms))


def snr_warning_codes(
    snr_db: float | None,
    warning_db: float = DEFAULT_SNR_WARNING_DB,
) -> tuple[str, ...]:
    if snr_db is not None and snr_db < warning_db:
        return (SNR_BELOW_RECOMMENDED,)
    return ()


def inspect_audio(
    path: Path | str,
    *,
    snr_warning_db: float = DEFAULT_SNR_WARNING_DB,
    max_clipping_ratio: float = DEFAULT_MAX_CLIPPING_RATIO,
) -> AudioInspection:
    audio_path = Path(path)
    if audio_path.suffix.lower() not in {".wav", ".flac", ".mp3"}:
        raise UnsupportedAudioFormat(audio_path.suffix.lower())
    samples, sample_rate = _load_audio(audio_path)
    clipping_ratio = float(np.mean(np.abs(samples) >= 0.999))
    snr_db = _estimate_snr(samples)
    rejection_codes: list[str] = []
    if clipping_ratio > max_clipping_ratio:
        rejection_codes.append("CLIPPING_EXCEEDED")
    warning_codes = snr_warning_codes(snr_db, snr_warning_db)
    return AudioInspection(
        path=audio_path,
        sample_rate=sample_rate,
        channels=int(samples.shape[0]),
        duration_seconds=round(samples.shape[1] / sample_rate, 6),
        snr_db=round(snr_db, 3),
        clipping_ratio=round(clipping_ratio, 6),
        rejection_codes=tuple(rejection_codes),
        warning_codes=warning_codes,
    )


def _speech_intervals(mono: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
    frame_length = max(256, int(sample_rate * 0.02))
    hop_length = max(128, int(sample_rate * 0.01))
    frame_rms = librosa.feature.rms(
        y=mono,
        frame_length=frame_length,
        hop_length=hop_length,
        center=False,
    )[0]
    if len(frame_rms) == 0:
        return []
    percentile_threshold = float(np.percentile(frame_rms, 20)) * 1.5
    threshold = max(1e-4, min(percentile_threshold, float(np.max(frame_rms)) * 0.5))
    active = frame_rms >= threshold
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_active in enumerate(active):
        if is_active and start is None:
            start = index
        if start is not None and (not is_active or index == len(active) - 1):
            end_index = index + 1 if is_active and index == len(active) - 1 else index
            intervals.append((start * hop_length, min(len(mono), end_index * hop_length + frame_length)))
            start = None
    return intervals


def _merge_nearby_speech_intervals(
    intervals: list[tuple[int, int]],
    sample_rate: int,
    *,
    max_silence_seconds: float = DEFAULT_MAX_SPEECH_MERGE_GAP_SECONDS,
) -> list[tuple[int, int]]:
    max_gap = int(max_silence_seconds * sample_rate)
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if merged and start - merged[-1][1] <= max_gap:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(end, previous_end))
        else:
            merged.append((start, end))
    return merged


def _chunk_interval(start: int, end: int, sample_rate: int) -> list[tuple[int, int]]:
    minimum = int(DEFAULT_SEGMENT_MIN_SECONDS * sample_rate)
    maximum = int(DEFAULT_SEGMENT_MAX_SECONDS * sample_rate)
    chunks: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        remaining = end - cursor
        if remaining < minimum and chunks:
            previous_start, _ = chunks[-1]
            chunks[-1] = (previous_start, end)
            break
        chunk_end = min(end, cursor + maximum)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return [chunk for chunk in chunks if chunk[1] - chunk[0] >= minimum]


def _split_for_source(source: Path, seed: int = SPLIT_SEED) -> str:
    value = int(hashlib.sha256(f"{seed}:{source.name}".encode()).hexdigest()[:8], 16) % 10
    if value == 0:
        return "test"
    if value == 1:
        return "validation"
    return "train"


def validate_training_duration(effective_seconds: float) -> None:
    """Offline-training-only gate; the client never reads this 480-720 window."""
    if not DATASET_DURATION_MIN <= effective_seconds <= DATASET_DURATION_MAX:
        raise DatasetDurationOutOfRange(
            f"Effective speech must be between {DATASET_DURATION_MIN:.0f} and {DATASET_DURATION_MAX:.0f} seconds"
        )


def validate_reference_duration(effective_seconds: float) -> None:
    """Client reference gate: inclusive 3.00-10.00 seconds of effective speech."""
    if not REFERENCE_SECONDS_MIN <= effective_seconds <= REFERENCE_SECONDS_MAX:
        raise ReferenceDurationOutOfRange(
            f"参考音频有效人声时长需在 {REFERENCE_SECONDS_MIN:.2f}–{REFERENCE_SECONDS_MAX:.2f} 秒内"
        )


def prepare_reference_dataset(
    dataset_id: str,
    sources: Iterable[Path | str] | None = None,
    *,
    output_dir: Path | str | None = None,
    speaker: str | None = None,
    language: str = "zh",
) -> DatasetManifest:
    """Normalize an authorized short upload into ONE local reference segment.

    Zero-shot references have no train/validation/test purpose, so the single
    persisted row always carries the fixed ``reference`` split and the raw
    duration window is re-checked by :func:`validate_reference_duration`.
    """
    source_paths = (
        [Path(source) for source in sources]
        if sources is not None
        else sorted(Path("data/uploads").joinpath(dataset_id).glob("*"))
    )
    target_dir = Path(output_dir or Path("data/datasets") / dataset_id)
    segment_dir = target_dir / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    rows: list[DatasetManifestRow] = []
    for source in source_paths:
        inspection = inspect_audio(source)
        if inspection.rejection_codes:
            continue
        samples, sample_rate = _load_audio(source, DEFAULT_SAMPLE_RATE)
        mono = np.mean(samples, axis=0)
        speech_intervals = _merge_nearby_speech_intervals(
            _speech_intervals(mono, sample_rate),
            sample_rate,
        )
        if not speech_intervals:
            continue
        start = speech_intervals[0][0]
        end = speech_intervals[-1][1]
        segment = mono[start:end]
        peak = float(np.max(np.abs(segment))) if len(segment) else 0.0
        if peak > 0:
            segment = segment * (10 ** (-1.0 / 20.0)) / peak
        segment_path = segment_dir / "seg_0001.wav"
        sf.write(segment_path, segment.astype(np.float32), sample_rate, subtype="PCM_16")
        rows.append(
            DatasetManifestRow(
                segment_id="seg_0001",
                path=f"datasets/{dataset_id}/segments/seg_0001.wav",
                speaker=speaker or f"voice_{dataset_id}",
                language=language,
                duration_seconds=round(len(segment) / sample_rate, 6),
                snr_db=inspection.snr_db,
                clipping_ratio=0.0,
                warning_codes=inspection.warning_codes,
                split=REFERENCE_SPLIT,
            )
        )
        break
    return DatasetManifest(
        dataset_id=dataset_id,
        effective_seconds=round(sum(row.duration_seconds for row in rows), 6),
        rows=rows,
        manifest_path=target_dir / "manifest.jsonl",
    )


def prepare_dataset(
    dataset_id: str,
    sources: Iterable[Path | str] | None = None,
    *,
    output_dir: Path | str | None = None,
    speaker: str | None = None,
    language: str = "zh",
) -> DatasetManifest:
    source_paths = (
        [Path(source) for source in sources]
        if sources is not None
        else sorted(Path("data/uploads").joinpath(dataset_id).glob("*"))
    )
    target_dir = Path(output_dir or Path("data/datasets") / dataset_id)
    segment_dir = target_dir / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    rows: list[DatasetManifestRow] = []
    segment_number = 1
    for source in source_paths:
        inspection = inspect_audio(source)
        if inspection.rejection_codes:
            continue
        samples, sample_rate = _load_audio(source, DEFAULT_SAMPLE_RATE)
        mono = np.mean(samples, axis=0)
        split = _split_for_source(source)
        speech_intervals = _merge_nearby_speech_intervals(
            _speech_intervals(mono, sample_rate),
            sample_rate,
        )
        for start, end in (
            interval
            for speech_start, speech_end in speech_intervals
            for interval in _chunk_interval(speech_start, speech_end, sample_rate)
        ):
            segment = mono[start:end]
            peak = float(np.max(np.abs(segment))) if len(segment) else 0.0
            if peak > 0:
                segment = segment * (10 ** (-1.0 / 20.0)) / peak
            segment_id = f"seg_{segment_number:04d}"
            segment_number += 1
            segment_path = segment_dir / f"{segment_id}.wav"
            sf.write(segment_path, segment.astype(np.float32), sample_rate, subtype="PCM_16")
            rows.append(
                DatasetManifestRow(
                    segment_id=segment_id,
                    path=f"datasets/{dataset_id}/segments/{segment_id}.wav",
                    speaker=speaker or f"voice_{dataset_id}",
                    language=language,
                    duration_seconds=round(len(segment) / sample_rate, 6),
                    snr_db=inspection.snr_db,
                    clipping_ratio=0.0,
                    warning_codes=inspection.warning_codes,
                    split=split,
                )
            )
    manifest_path = target_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row.model_dump(mode="json"), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return DatasetManifest(
        dataset_id=dataset_id,
        effective_seconds=round(sum(row.duration_seconds for row in rows), 6),
        rows=rows,
        manifest_path=manifest_path,
    )
