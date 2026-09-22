from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class FingerprintBaseline:
    high_band_energy_mean: float
    high_band_energy_std: float


@dataclass(frozen=True)
class SpectralFingerprint:
    sample_rate: int
    high_band_hz: tuple[float, float]
    high_band_energy_ratio: float
    spectral_flatness: float
    zscore_vs_profile: float
    anomaly: bool


class FingerprintService:
    """Measure the fixed relative high-frequency band without binding to a sample rate."""

    LOW_RATIO = 0.75
    HIGH_RATIO = 0.95
    ANOMALY_ZSCORE = 4.0

    def analyze(
        self,
        wav: Path | str,
        *,
        baseline: FingerprintBaseline | None = None,
    ) -> SpectralFingerprint:
        wav_path = Path(wav).resolve()
        samples, sample_rate = sf.read(wav_path, dtype="float64", always_2d=True)
        if samples.size == 0 or sample_rate <= 0:
            raise ValueError("cannot fingerprint an empty local WAV")
        mono = samples.mean(axis=1)
        spectrum = np.fft.rfft(mono)
        power = np.abs(spectrum) ** 2
        frequencies = np.fft.rfftfreq(mono.size, d=1.0 / sample_rate)
        nyquist = sample_rate / 2.0
        low_hz = self.LOW_RATIO * nyquist
        high_hz = self.HIGH_RATIO * nyquist
        mask = (frequencies >= low_hz) & (frequencies <= high_hz)
        if not mask.any():
            raise ValueError("audio is too short to measure the configured high-frequency band")
        total_energy = float(power.sum())
        high_band_energy = float(power[mask].sum())
        ratio = high_band_energy / total_energy if total_energy > 0 else 0.0
        band_power = power[mask] + np.finfo(np.float64).eps
        spectral_flatness = float(np.exp(np.mean(np.log(band_power))) / np.mean(band_power))
        zscore = self._zscore(ratio, baseline)
        return SpectralFingerprint(
            sample_rate=int(sample_rate),
            high_band_hz=(low_hz, high_hz),
            high_band_energy_ratio=ratio,
            spectral_flatness=spectral_flatness,
            zscore_vs_profile=zscore,
            anomaly=abs(zscore) >= self.ANOMALY_ZSCORE,
        )

    @staticmethod
    def _zscore(value: float, baseline: FingerprintBaseline | None) -> float:
        if baseline is None:
            return 0.0
        if baseline.high_band_energy_std < 0:
            raise ValueError("fingerprint baseline standard deviation must not be negative")
        if baseline.high_band_energy_std == 0:
            return 0.0 if value == baseline.high_band_energy_mean else float("inf")
        return (value - baseline.high_band_energy_mean) / baseline.high_band_energy_std
