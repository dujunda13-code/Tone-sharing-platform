from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from backend.app.services.fingerprint import FingerprintBaseline, FingerprintService


def test_fingerprint_uses_relative_high_band_and_flags_large_profile_deviation(tmp_path: Path):
    sample_rate = 16_000
    seconds = 0.25
    samples = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    audio = 0.4 * np.sin(2 * np.pi * 7_000 * samples)
    wav = tmp_path / "high-band.wav"
    sf.write(wav, audio, sample_rate)

    fingerprint = FingerprintService().analyze(
        wav,
        baseline=FingerprintBaseline(high_band_energy_mean=0.0, high_band_energy_std=0.01),
    )

    assert fingerprint.sample_rate == 16_000
    assert fingerprint.high_band_hz == (6_000.0, 7_600.0)
    assert 0.0 <= fingerprint.high_band_energy_ratio <= 1.0
    assert fingerprint.zscore_vs_profile >= 4.0
    assert fingerprint.anomaly is True
