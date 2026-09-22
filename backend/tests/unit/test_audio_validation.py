from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from backend.app.services.audio_validation import (
    _chunk_interval,
    _merge_nearby_speech_intervals,
    inspect_audio,
    prepare_dataset,
    prepare_reference_dataset,
    validate_reference_duration,
    ReferenceDurationOutOfRange,
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _write_low_snr_reference(path: Path) -> None:
    sample_rate = 16_000
    seconds = 6
    time = np.arange(sample_rate * seconds, dtype=np.float32) / sample_rate
    rng = np.random.default_rng(20260912)
    voice = 0.08 * np.sin(2 * np.pi * 220 * time)
    noise = 0.05 * rng.standard_normal(time.shape[0])
    sf.write(path, np.clip(voice + noise, -0.8, 0.8), sample_rate, subtype="PCM_16")


def test_low_snr_is_advisory_instead_of_rejected(tmp_path: Path):
    source = tmp_path / "low-snr.wav"
    _write_low_snr_reference(source)

    report = inspect_audio(source)

    assert report.snr_db < 20.0
    assert report.rejection_codes == ()
    assert report.warning_codes == ("SNR_BELOW_RECOMMENDED",)


def test_prepare_reference_dataset_keeps_low_snr_with_warning(tmp_path: Path):
    source = tmp_path / "low-snr.wav"
    _write_low_snr_reference(source)

    manifest = prepare_reference_dataset(
        "low-snr", [source], output_dir=tmp_path / "out"
    )

    assert len(manifest.rows) == 1
    assert 3.0 <= manifest.effective_seconds <= 10.0
    assert manifest.rows[0].warning_codes == ("SNR_BELOW_RECOMMENDED",)


def test_clipped_audio_is_rejected():
    report = inspect_audio(FIXTURES / "clipped_3s.wav")

    assert report.clipping_ratio > 0.01
    assert "CLIPPING_EXCEEDED" in report.rejection_codes
    assert isinstance(report.warning_codes, tuple)


def test_clean_audio_is_16khz_compatible_and_accepted():
    report = inspect_audio(FIXTURES / "clean_10s.wav")

    assert report.sample_rate == 16_000
    assert report.channels == 1
    assert report.duration_seconds == 10.0
    assert report.rejection_codes == ()


def test_prepare_dataset_keeps_constant_energy_speech(tmp_path: Path):
    manifest = prepare_dataset(
        "d-clean",
        [FIXTURES / "clean_10s.wav"],
        output_dir=tmp_path,
    )

    assert manifest.rows
    assert manifest.effective_seconds >= 3.0
    assert manifest.rows[0].path.startswith("datasets/d-clean/segments/")


@pytest.mark.parametrize("seconds", [3.0, 5.0, 10.0])
def test_reference_duration_accepts_inclusive_range(seconds):
    validate_reference_duration(seconds)


@pytest.mark.parametrize("seconds", [0.0, 2.99, 10.01, 600.0])
def test_reference_duration_rejects_outside_range(seconds):
    with pytest.raises(ReferenceDurationOutOfRange):
        validate_reference_duration(seconds)


def test_prepare_reference_dataset_persists_one_reference_segment(tmp_path: Path):
    manifest = prepare_reference_dataset(
        "ref-clean",
        [FIXTURES / "clean_10s.wav"],
        output_dir=tmp_path,
    )

    assert len(manifest.rows) == 1
    assert manifest.rows[0].split == "reference"
    assert 3.0 <= manifest.effective_seconds <= 10.0
    assert manifest.rows[0].path.startswith("datasets/ref-clean/segments/")
    assert (tmp_path / manifest.rows[0].path.removeprefix("datasets/ref-clean/")).is_file()


def test_prepare_reference_dataset_yields_no_rows_for_clipped_audio(tmp_path: Path):
    manifest = prepare_reference_dataset(
        "ref-clipped",
        [FIXTURES / "clipped_3s.wav"],
        output_dir=tmp_path,
    )

    assert manifest.rows == []
    with pytest.raises(ReferenceDurationOutOfRange):
        validate_reference_duration(manifest.effective_seconds)


def test_merge_nearby_speech_intervals_keeps_long_silence_as_boundary():
    sample_rate = 16_000

    merged = _merge_nearby_speech_intervals(
        [(0, 16_000), (20_000, 48_000), (60_001, 76_001)],
        sample_rate,
    )

    assert merged == [(0, 48_000), (60_001, 76_001)]


def test_merged_short_gap_produces_a_minimum_training_chunk():
    sample_rate = 16_000
    merged = _merge_nearby_speech_intervals(
        [(0, 16_000), (24_000, 48_000)],
        sample_rate,
    )

    chunks = [
        chunk
        for start, end in merged
        for chunk in _chunk_interval(start, end, sample_rate)
    ]

    assert chunks == [(0, 48_000)]
