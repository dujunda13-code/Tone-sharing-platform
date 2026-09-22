from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import torch

from backend.app.services.watermark import (
    WatermarkDetection,
    WatermarkService,
    mono_samples_to_bct,
    normalize_detector_sample_rate,
)


class CopyingWatermarkBackend:
    def __init__(self) -> None:
        self.payloads: list[int] = []

    def embed(self, input_wav: Path, output_wav: Path, payload: int) -> None:
        self.payloads.append(payload)
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_wav, output_wav)

    def detect(self, wav: Path) -> WatermarkDetection:
        return WatermarkDetection(probability=0.83, payload=513)


def test_watermark_service_embeds_and_returns_detector_probability(tmp_path: Path):
    source = tmp_path / "raw.wav"
    destination = tmp_path / "watermarked.wav"
    source.write_bytes(b"local authorized audio")
    backend = CopyingWatermarkBackend()
    service = WatermarkService(backend=backend)

    embedded = service.embed(source, destination, payload=513)
    detection = service.detect(destination)

    assert embedded.output_wav == destination
    assert embedded.payload == 513
    assert backend.payloads == [513]
    assert destination.read_bytes() == b"local authorized audio"
    assert detection == WatermarkDetection(probability=0.83, payload=513)


def test_local_audioseal_input_is_batched_mono_channel_time():
    samples = np.arange(4, dtype=np.float32)

    audio = mono_samples_to_bct(samples, torch.device("cpu"))

    assert tuple(audio.shape) == (1, 1, 4)
    assert audio.dtype is torch.float32


def test_detector_audio_is_normalized_to_the_fixed_sixteen_kilohertz_model_rate():
    samples = np.arange(32_000, dtype=np.float32)

    normalized, sample_rate = normalize_detector_sample_rate(samples, 32_000)

    assert sample_rate == 16_000
    assert normalized.dtype == np.dtype(np.float32)
    assert normalized.shape == (16_000,)
