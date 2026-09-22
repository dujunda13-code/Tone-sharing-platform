from pathlib import Path

from backend.app.core.config import AppSettings, get_settings


ROOT = Path(__file__).resolve().parents[3]


def test_get_settings_loads_fixed_runtime_contract():
    settings = get_settings(Path("config/app.yaml"))

    assert isinstance(settings, AppSettings)
    assert settings.runtime.device == "cuda:0"
    assert settings.runtime.gpu_workers == 1
    assert settings.audio.offline_training_effective_seconds_min == 480
    assert settings.audio.offline_training_effective_seconds_max == 720
    assert settings.audio.snr_warning_db == 20.0
    assert not hasattr(settings.audio, "min_snr_db")
    assert settings.quality.watermark_probability == 0.80
    assert settings.storage.root == Path("data")


def test_get_settings_exposes_authorized_eight_gib_runtime_profile():
    settings = get_settings(Path("config/app.yaml"))

    assert settings.runtime.minimum_vram_gib == 8
    assert settings.runtime.cuda_allocator_conf == "expandable_segments:True"
    assert settings.training.batch_size == 1
    assert settings.training.fp16 is True
    assert settings.training.gpu_oom_retries == 1
    assert settings.training.gpu_phases_serial is True
    assert settings.training.s2_grad_ckpt is True


def test_zero_shot_contract_uses_short_reference_and_registered_base():
    settings = get_settings(ROOT / "config/app.yaml")

    assert settings.audio.reference_seconds_min == 3.0
    assert settings.audio.reference_seconds_max == 10.0
    assert settings.models.active_voice_base == "gpt-sovits-v2proplus-official"
    assert set(settings.models.voice_bases) == {"gpt-sovits-v2proplus-official"}
    base = settings.models.voice_bases[settings.models.active_voice_base]
    assert base.family == "GPT-SoVITS"
    assert base.source_tag == "20250606v2pro"
    assert base.gpt_weight.as_posix() == "gpt-sovits/v2Pro/s1v3.ckpt"
    assert base.sovits_weight.as_posix() == "gpt-sovits/v2Pro/s2Gv2ProPlus.pth"
