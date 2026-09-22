from __future__ import annotations

from dataclasses import dataclass
from math import gcd
import os
from pathlib import Path
from typing import Protocol


class WatermarkRuntimeError(RuntimeError):
    """Raised when the local AudioSeal runtime or its pinned local models are unavailable."""


@dataclass(frozen=True)
class WatermarkResult:
    output_wav: Path
    payload: int


@dataclass(frozen=True)
class WatermarkDetection:
    probability: float
    payload: int | None


class WatermarkBackend(Protocol):
    def embed(self, input_wav: Path, output_wav: Path, payload: int) -> None: ...

    def detect(self, wav: Path) -> WatermarkDetection: ...


def mono_samples_to_bct(samples, device):
    """Convert local mono PCM into AudioSeal's batch/channel/time tensor shape."""
    import torch

    return torch.from_numpy(samples).reshape(1, 1, -1).to(device=device, dtype=torch.float32)


def normalize_detector_sample_rate(samples, sample_rate: int):
    """Resample detector input to AudioSeal's fixed 16 kHz model rate."""
    import numpy as np
    from scipy.signal import resample_poly

    if sample_rate <= 0:
        raise WatermarkRuntimeError("local WAV has an invalid sample rate")
    mono = np.asarray(samples, dtype=np.float32)
    if sample_rate == 16_000:
        return mono, sample_rate
    divisor = gcd(sample_rate, 16_000)
    normalized = resample_poly(mono, 16_000 // divisor, sample_rate // divisor)
    return np.asarray(normalized, dtype=np.float32), 16_000


class LocalAudioSealBackend:
    """Lazy, local-only AudioSeal adapter. Model paths are never remote model-card names."""

    def __init__(
        self,
        generator_model: Path | str = Path("models/audioseal/generator_base.pth"),
        detector_model: Path | str = Path("models/audioseal/detector_base.pth"),
    ) -> None:
        self.generator_model = Path(generator_model).resolve()
        self.detector_model = Path(detector_model).resolve()
        self._generator = None
        self._detector = None

    def _load_models(self):
        if not self.generator_model.exists() or not self.detector_model.exists():
            raise WatermarkRuntimeError("local AudioSeal model files are unavailable")
        try:
            # AudioSeal 0.2's optional Inductor path is incompatible with the pinned
            # PyTorch 2.5.1 runtime on this Windows profile. This only disables an
            # optimization; the local generator and detector stay unchanged.
            os.environ.setdefault("NO_TORCH_COMPILE", "1")
            import audioseal
            import torch
        except ImportError as exc:
            raise WatermarkRuntimeError("AudioSeal local runtime is unavailable") from exc
        if not torch.cuda.is_available():
            raise WatermarkRuntimeError("AudioSeal requires local cuda:0")
        device = torch.device("cuda:0")
        if self._generator is None:
            self._generator = audioseal.AudioSeal.load_generator(
                str(self.generator_model), nbits=16, device=device, dtype=torch.float16
            )
            self._generator.eval()
        if self._detector is None:
            self._detector = audioseal.AudioSeal.load_detector(
                str(self.detector_model), nbits=16, device=device, dtype=torch.float16
            )
            self._detector.eval()
        return torch, device

    @staticmethod
    def _read_mono(wav: Path):
        try:
            import soundfile as sf
        except ImportError as exc:
            raise WatermarkRuntimeError("soundfile local runtime is unavailable") from exc
        samples, sample_rate = sf.read(wav, dtype="float32", always_2d=True)
        if samples.size == 0:
            raise WatermarkRuntimeError("cannot watermark an empty local WAV")
        return samples.mean(axis=1), int(sample_rate), sf

    def embed(self, input_wav: Path, output_wav: Path, payload: int) -> None:
        torch, device = self._load_models()
        samples, sample_rate, soundfile = self._read_mono(input_wav)
        # AudioSeal's generator is 16 kHz-native just like the detector:
        # embedding at the source rate (e.g. the 32 kHz v2Pro output) writes a
        # watermark the recheck can never verify. Normalize both sides.
        samples, sample_rate = normalize_detector_sample_rate(samples, sample_rate)
        message = torch.tensor(
            [[(payload >> shift) & 1 for shift in range(15, -1, -1)]],
            device=device,
            dtype=torch.float32,
        )
        audio = mono_samples_to_bct(samples, device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            watermarked = self._generator(audio, sample_rate=sample_rate, message=message)
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(output_wav, watermarked[0, 0].float().cpu().numpy(), sample_rate)

    def detect(self, wav: Path) -> WatermarkDetection:
        torch, device = self._load_models()
        samples, sample_rate, _ = self._read_mono(wav)
        samples, sample_rate = normalize_detector_sample_rate(samples, sample_rate)
        audio = mono_samples_to_bct(samples, device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            probability, message = self._detector.detect_watermark(audio, sample_rate=sample_rate)
        bits = message[0].to(torch.int64).cpu().tolist()
        payload = sum(int(bit) << (15 - index) for index, bit in enumerate(bits))
        return WatermarkDetection(probability=float(probability[0].cpu()), payload=payload)


class WatermarkService:
    """Embed and inspect the fixed 16-bit watermark through a local backend."""

    def __init__(self, backend: WatermarkBackend | None = None) -> None:
        self.backend = backend or LocalAudioSealBackend()

    def embed(self, input_wav: Path | str, output_wav: Path | str, payload: int) -> WatermarkResult:
        if not 0 <= payload <= 0xFFFF:
            raise ValueError("watermark payload must fit in 16 bits")
        input_path = Path(input_wav).resolve()
        output_path = Path(output_wav).resolve()
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        self.backend.embed(input_path, output_path, payload)
        if not output_path.is_file():
            raise WatermarkRuntimeError("local watermark backend did not create an output WAV")
        return WatermarkResult(output_wav=output_path, payload=payload)

    def detect(self, wav: Path | str) -> WatermarkDetection:
        wav_path = Path(wav).resolve()
        if not wav_path.is_file():
            raise FileNotFoundError(wav_path)
        detection = self.backend.detect(wav_path)
        if not 0.0 <= detection.probability <= 1.0:
            raise WatermarkRuntimeError("local watermark backend returned an invalid probability")
        if detection.payload is not None and not 0 <= detection.payload <= 0xFFFF:
            raise WatermarkRuntimeError("local watermark backend returned an invalid payload")
        return detection
