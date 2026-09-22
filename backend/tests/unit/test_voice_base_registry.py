import hashlib
from pathlib import Path

import pytest

from backend.app.core.config import (
    AppSettings,
    AudioSettings,
    ModelSettings,
    QualitySettings,
    RuntimeSettings,
    StorageSettings,
    TrainingSettings,
    VoiceBaseSettings,
)
from backend.app.services.voice_base_registry import BaseModelUnavailable, VoiceBaseRegistry


ACTIVE_BASE_ID = "gpt-sovits-v2proplus-official"


def _make_settings(active_base: str = ACTIVE_BASE_ID, **weight_overrides) -> AppSettings:
    base = {
        "family": "GPT-SoVITS",
        "source_tag": "20250606v2pro",
        "gpt_weight": Path("gpt-sovits/v2Pro/s1v3.ckpt"),
        "sovits_weight": Path("gpt-sovits/v2Pro/s2Gv2ProPlus.pth"),
    }
    base.update(weight_overrides)
    return AppSettings(
        runtime=RuntimeSettings(
            device="cuda:0",
            gpu_workers=1,
            minimum_vram_gib=8,
            cuda_allocator_conf="expandable_segments:True",
        ),
        training=TrainingSettings(
            batch_size=1,
            fp16=True,
            gpu_oom_retries=1,
            gpu_phases_serial=True,
            s2_grad_ckpt=True,
        ),
        models=ModelSettings(
            gpt_sovits_tag="20250606v2pro",
            emotion_model="iic/emotion2vec_plus_large",
            speaker_model="iic/speech_campplus_sv_zh-cn_16k-common",
            watermark_generator="audioseal_wm_16bits",
            watermark_detector="audioseal_detector_16bits",
            active_voice_base=active_base,
            voice_bases={ACTIVE_BASE_ID: VoiceBaseSettings(**base)},
        ),
        audio=AudioSettings(
            reference_seconds_min=3.0,
            reference_seconds_max=10.0,
            offline_training_effective_seconds_min=480,
            offline_training_effective_seconds_max=720,
            segment_seconds_min=3,
            segment_seconds_max=12,
            snr_warning_db=20.0,
            max_clipping_ratio=0.01,
        ),
        quality=QualitySettings(
            speaker_similarity_median=0.90,
            speaker_similarity_p10=0.85,
            watermark_probability=0.80,
            emotion_confidence=0.55,
        ),
        storage=StorageSettings(root=Path("data")),
    )


def _deploy_weight(models_root: Path, relative: str, payload: bytes) -> None:
    weight_path = models_root / relative
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    weight_path.write_bytes(payload)


def _write_manifest(models_root: Path, entries: dict[str, bytes]) -> Path:
    lines = [
        f"{hashlib.sha256(payload).hexdigest()}  models/{relative}"
        for relative, payload in entries.items()
    ]
    manifest = models_root / "checksums.sha256"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _make_registry(tmp_path: Path, settings: AppSettings) -> VoiceBaseRegistry:
    models_root = tmp_path / "models"
    gpt_payload = b"gpt-weight-bytes"
    sovits_payload = b"sovits-weight-bytes"
    _deploy_weight(models_root, "gpt-sovits/v2Pro/s1v3.ckpt", gpt_payload)
    _deploy_weight(models_root, "gpt-sovits/v2Pro/s2Gv2ProPlus.pth", sovits_payload)
    manifest = _write_manifest(
        models_root,
        {
            "gpt-sovits/v2Pro/s1v3.ckpt": gpt_payload,
            "gpt-sovits/v2Pro/s2Gv2ProPlus.pth": sovits_payload,
        },
    )
    return VoiceBaseRegistry(
        settings=settings,
        models_root=models_root,
        checksum_path=manifest,
    )


def test_resolve_returns_checksum_verified_paths_under_models_root(tmp_path):
    registry = _make_registry(tmp_path, _make_settings())

    resolved = registry.resolve()

    assert resolved.id == ACTIVE_BASE_ID
    assert resolved.source_tag == "20250606v2pro"
    assert resolved.gpt_weight == (tmp_path / "models/gpt-sovits/v2Pro/s1v3.ckpt").resolve()
    assert resolved.sovits_weight == (
        tmp_path / "models/gpt-sovits/v2Pro/s2Gv2ProPlus.pth"
    ).resolve()


def test_resolve_defaults_to_active_base_and_accepts_explicit_id(tmp_path):
    registry = _make_registry(tmp_path, _make_settings())

    assert registry.resolve(ACTIVE_BASE_ID).id == registry.resolve().id


def test_resolve_rejects_unknown_base_model(tmp_path):
    registry = _make_registry(tmp_path, _make_settings())

    with pytest.raises(BaseModelUnavailable, match="unknown-base"):
        registry.resolve("unknown-base")


def test_resolve_rejects_weight_path_escaping_models_root(tmp_path):
    registry = _make_registry(
        tmp_path,
        _make_settings(gpt_weight=Path("../outside/s1v3.ckpt")),
    )

    with pytest.raises(BaseModelUnavailable):
        registry.resolve()


def test_resolve_rejects_weight_missing_from_checksum_manifest(tmp_path):
    models_root = tmp_path / "models"
    gpt_payload = b"gpt-weight-bytes"
    sovits_payload = b"sovits-weight-bytes"
    _deploy_weight(models_root, "gpt-sovits/v2Pro/s1v3.ckpt", gpt_payload)
    _deploy_weight(models_root, "gpt-sovits/v2Pro/s2Gv2ProPlus.pth", sovits_payload)
    manifest = _write_manifest(
        models_root,
        {"gpt-sovits/v2Pro/s2Gv2ProPlus.pth": sovits_payload},
    )
    registry = VoiceBaseRegistry(
        settings=_make_settings(), models_root=models_root, checksum_path=manifest
    )

    with pytest.raises(BaseModelUnavailable):
        registry.resolve()


def test_resolve_rejects_sha256_mismatch(tmp_path):
    models_root = tmp_path / "models"
    gpt_payload = b"gpt-weight-bytes"
    sovits_payload = b"sovits-weight-bytes"
    _deploy_weight(models_root, "gpt-sovits/v2Pro/s1v3.ckpt", gpt_payload)
    _deploy_weight(models_root, "gpt-sovits/v2Pro/s2Gv2ProPlus.pth", sovits_payload)
    manifest = _write_manifest(
        models_root,
        {
            "gpt-sovits/v2Pro/s1v3.ckpt": b"tampered-payload",
            "gpt-sovits/v2Pro/s2Gv2ProPlus.pth": sovits_payload,
        },
    )
    registry = VoiceBaseRegistry(
        settings=_make_settings(), models_root=models_root, checksum_path=manifest
    )

    with pytest.raises(BaseModelUnavailable):
        registry.resolve()


def test_resolve_rejects_missing_weight_file(tmp_path):
    models_root = tmp_path / "models"
    sovits_payload = b"sovits-weight-bytes"
    _deploy_weight(models_root, "gpt-sovits/v2Pro/s2Gv2ProPlus.pth", sovits_payload)
    manifest = _write_manifest(
        models_root,
        {
            "gpt-sovits/v2Pro/s1v3.ckpt": b"gpt-weight-bytes",
            "gpt-sovits/v2Pro/s2Gv2ProPlus.pth": sovits_payload,
        },
    )
    registry = VoiceBaseRegistry(
        settings=_make_settings(), models_root=models_root, checksum_path=manifest
    )

    with pytest.raises(BaseModelUnavailable):
        registry.resolve()
