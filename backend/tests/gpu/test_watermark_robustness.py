from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from uuid import uuid4

import librosa
import imageio_ffmpeg
import numpy as np
import pytest
import soundfile as sf
import torch

from backend.app.services.watermark import WatermarkService


ROOT = Path(__file__).resolve().parents[3]
TEMP_ROOT = ROOT / "data" / "temp"
SOURCE_WAV = ROOT / "backend" / "tests" / "fixtures" / "clean_10s.wav"
PAYLOAD = 0x1234


def _run_ffmpeg(ffmpeg: str, *args: str) -> None:
    completed = subprocess.run(
        [ffmpeg, "-nostdin", "-y", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _ffmpeg_executable() -> str:
    return shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()


@pytest.mark.gpu
def test_audioseal_watermark_survives_supported_attacks_and_rejects_severe_noise():
    ffmpeg = _ffmpeg_executable()
    assert Path(ffmpeg).is_file(), "ffmpeg is required for the fixed MP3 192 kbps acceptance attack"
    assert torch.cuda.is_available(), "real AudioSeal acceptance requires cuda:0"
    work_dir = (TEMP_ROOT / f"task9-watermark-robustness-{uuid4().hex}").resolve()
    work_dir.relative_to(TEMP_ROOT.resolve())
    work_dir.mkdir(parents=True)
    embedded = work_dir / "embedded.wav"
    mp3 = work_dir / "embedded-192k.mp3"
    mp3_decoded = work_dir / "embedded-192k.wav"
    resampled = work_dir / "embedded-32k.wav"
    noisy = work_dir / "embedded-30db.wav"

    try:
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        watermark = WatermarkService()
        watermark.embed(SOURCE_WAV, embedded, PAYLOAD)
        samples, sample_rate = sf.read(embedded, dtype="float32", always_2d=True)
        _run_ffmpeg(ffmpeg, "-i", str(embedded), "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3))
        _run_ffmpeg(ffmpeg, "-i", str(mp3), str(mp3_decoded))
        samples_48k = librosa.resample(
            samples.mean(axis=1), orig_sr=sample_rate, target_sr=48_000
        )
        resampled_samples = librosa.resample(
            samples_48k, orig_sr=48_000, target_sr=32_000
        )
        sf.write(resampled, resampled_samples, 32_000)
        signal_rms = float(np.sqrt(np.mean(samples.mean(axis=1) ** 2)))
        noise_rms = signal_rms / (10 ** (30 / 20))
        noise = np.random.default_rng(20260902).normal(0.0, noise_rms, samples.shape[0])
        sf.write(noisy, samples.mean(axis=1) + noise, sample_rate)

        for attack_name, attacked in (
            ("mp3_192kbps", mp3_decoded),
            ("resample_48khz_to_32khz", resampled),
        ):
            detection = watermark.detect(attacked)
            assert detection.probability >= 0.80, f"{attack_name}: {detection}"
            assert detection.payload == PAYLOAD, f"{attack_name}: {detection}"
        severe_noise_detection = watermark.detect(noisy)
        assert (
            severe_noise_detection.probability < 0.80
            or severe_noise_detection.payload != PAYLOAD
        ), f"severe external noise must remain unverified: {severe_noise_detection}"
        assert torch.cuda.max_memory_allocated(0) <= 8000 * 1024**2
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir)


@pytest.mark.gpu
def test_audioseal_embeds_at_model_rate_for_high_sample_rate_sources():
    """v2Pro synthesis outputs 32 kHz WAVs, and the mandatory watermark
    recheck runs after embedding: the generator must operate at AudioSeal's
    16 kHz model rate (like the detector already does) or the watermark is
    undetectable and every real synthesis fails the publication gate."""
    assert torch.cuda.is_available(), "real AudioSeal acceptance requires cuda:0"
    work_dir = (TEMP_ROOT / f"watermark-rate-{uuid4().hex}").resolve()
    work_dir.relative_to(TEMP_ROOT.resolve())
    work_dir.mkdir(parents=True)
    try:
        source_samples, source_rate = sf.read(
            SOURCE_WAV, dtype="float32", always_2d=True
        )
        mono = source_samples.mean(axis=1)
        source_32k = work_dir / "source-32k.wav"
        sf.write(
            source_32k,
            librosa.resample(mono, orig_sr=source_rate, target_sr=32_000),
            32_000,
        )

        watermark = WatermarkService()
        embedded = work_dir / "embedded-32k.wav"
        watermark.embed(source_32k, embedded, PAYLOAD)
        detection = watermark.detect(embedded)
        assert detection.probability >= 0.80, str(detection)
        assert detection.payload == PAYLOAD, str(detection)
    finally:
        shutil.rmtree(work_dir)
