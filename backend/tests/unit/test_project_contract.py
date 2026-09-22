import importlib
import os
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[3]


def test_gpt_sovits_ffmpeg_python_runtime_module_is_importable():
    """The pinned vendor's tools.my_utils dependency must be available at runtime."""
    ffmpeg = importlib.import_module("ffmpeg")

    assert callable(ffmpeg.input)


def test_pinned_gpt_sovits_vendor_tools_import_without_network_downloads():
    """The deployed v2Pro tools module must import from declared local packages."""
    vendor_dir = ROOT / "vendor" / "GPT-SoVITS"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(vendor_dir / "GPT_SoVITS"), str(vendor_dir)]
    )
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", "from tools.my_utils import clean_path; print(clean_path('ok'))"],
        cwd=vendor_dir,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_required_directories_and_pins_exist():
    cfg = yaml.safe_load((ROOT / "config/app.yaml").read_text(encoding="utf-8"))

    assert cfg["runtime"] == {
        "device": "cuda:0",
        "gpu_workers": 1,
        "minimum_vram_gib": 8,
        "cuda_allocator_conf": "expandable_segments:True",
    }
    assert cfg["training"] == {
        "batch_size": 1,
        "fp16": True,
        "gpu_oom_retries": 1,
        "gpu_phases_serial": True,
        "s2_grad_ckpt": True,
    }
    assert cfg["models"]["gpt_sovits_tag"] == "20250606v2pro"
    assert cfg["quality"]["speaker_similarity_median"] == 0.90
    assert cfg["quality"]["speaker_similarity_p10"] == 0.85


def test_zero_shot_client_contract_in_fixed_configuration():
    cfg = yaml.safe_load((ROOT / "config/app.yaml").read_text(encoding="utf-8"))

    assert cfg["audio"]["reference_seconds_min"] == 3.0
    assert cfg["audio"]["reference_seconds_max"] == 10.0
    assert cfg["audio"]["offline_training_effective_seconds_min"] == 480
    assert cfg["audio"]["offline_training_effective_seconds_max"] == 720
    assert cfg["audio"]["snr_warning_db"] == 20
    assert "min_snr_db" not in cfg["audio"]

    assert cfg["models"]["active_voice_base"] == "gpt-sovits-v2proplus-official"
    assert set(cfg["models"]["voice_bases"]) == {"gpt-sovits-v2proplus-official"}
    base = cfg["models"]["voice_bases"]["gpt-sovits-v2proplus-official"]
    assert base["family"] == "GPT-SoVITS"
    assert base["source_tag"] == "20250606v2pro"
    assert base["gpt_weight"] == "gpt-sovits/v2Pro/s1v3.ckpt"
    assert base["sovits_weight"] == "gpt-sovits/v2Pro/s2Gv2ProPlus.pth"

    for name in (
        "uploads",
        "datasets",
        "features",
        "profiles",
        "outputs",
        "temp",
        "logs",
    ):
        assert (ROOT / "data" / name).is_dir()


def test_v2proplus_weight_has_reproducible_verified_installer():
    script = (ROOT / "scripts/download_models.ps1").read_text(encoding="utf-8-sig")
    manifest = (ROOT / "models/checksums.sha256").read_text(encoding="utf-8")
    digest = "d42a22bbbf65fb2bbdd45ad6a66841156977db45c7aabe0a6992ff378d9c7d3b"

    assert "InstallV2ProPlus" in script
    assert "s2Gv2ProPlus.pth" in script
    assert digest in script
    assert "Remove-Item -LiteralPath $v2ProPlusPart -Force" in script
    assert (
        f"{digest}  models/gpt-sovits/v2Pro/s2Gv2ProPlus.pth" in manifest
    )
