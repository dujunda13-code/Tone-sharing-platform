from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RuntimeSettings:
    device: str
    gpu_workers: int
    minimum_vram_gib: int
    cuda_allocator_conf: str


@dataclass(frozen=True)
class TrainingSettings:
    batch_size: int
    fp16: bool
    gpu_oom_retries: int
    gpu_phases_serial: bool
    s2_grad_ckpt: bool


@dataclass(frozen=True)
class VoiceBaseSettings:
    """One offline-deployed GPT-SoVITS base model entry.

    Weights are project-relative paths under ``models/`` and must be
    registered in ``models/checksums.sha256`` before use.
    """

    family: str
    source_tag: str
    gpt_weight: Path
    sovits_weight: Path


@dataclass(frozen=True)
class ModelSettings:
    gpt_sovits_tag: str
    emotion_model: str
    speaker_model: str
    watermark_generator: str
    watermark_detector: str
    active_voice_base: str
    voice_bases: dict[str, VoiceBaseSettings]


@dataclass(frozen=True)
class AudioSettings:
    reference_seconds_min: float
    reference_seconds_max: float
    offline_training_effective_seconds_min: int
    offline_training_effective_seconds_max: int
    segment_seconds_min: int
    segment_seconds_max: int
    snr_warning_db: float
    max_clipping_ratio: float


@dataclass(frozen=True)
class QualitySettings:
    speaker_similarity_median: float
    speaker_similarity_p10: float
    watermark_probability: float
    emotion_confidence: float


@dataclass(frozen=True)
class StorageSettings:
    root: Path


@dataclass(frozen=True)
class AppSettings:
    runtime: RuntimeSettings
    training: TrainingSettings
    models: ModelSettings
    audio: AudioSettings
    quality: QualitySettings
    storage: StorageSettings


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing configuration section: {name}")
    return value


def get_settings(path: Path | str = Path("config/app.yaml")) -> AppSettings:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Application configuration must be a mapping")

    runtime = _section(data, "runtime")
    training = _section(data, "training")
    models = _section(data, "models")
    audio = _section(data, "audio")
    quality = _section(data, "quality")
    storage = _section(data, "storage")

    voice_bases_data = models.get("voice_bases")
    if not isinstance(voice_bases_data, dict) or not voice_bases_data:
        raise ValueError("Missing configuration section: models.voice_bases")
    voice_bases: dict[str, VoiceBaseSettings] = {}
    for base_id, base_data in voice_bases_data.items():
        if not isinstance(base_data, dict):
            raise ValueError(f"Invalid voice base entry: {base_id}")
        voice_bases[str(base_id)] = VoiceBaseSettings(
            family=str(base_data["family"]),
            source_tag=str(base_data["source_tag"]),
            gpt_weight=Path(base_data["gpt_weight"]),
            sovits_weight=Path(base_data["sovits_weight"]),
        )

    return AppSettings(
        runtime=RuntimeSettings(**runtime),
        training=TrainingSettings(**training),
        models=ModelSettings(
            gpt_sovits_tag=str(models["gpt_sovits_tag"]),
            emotion_model=str(models["emotion_model"]),
            speaker_model=str(models["speaker_model"]),
            watermark_generator=str(models["watermark_generator"]),
            watermark_detector=str(models["watermark_detector"]),
            active_voice_base=str(models["active_voice_base"]),
            voice_bases=voice_bases,
        ),
        audio=AudioSettings(**audio),
        quality=QualitySettings(**quality),
        storage=StorageSettings(root=Path(storage["root"])),
    )
