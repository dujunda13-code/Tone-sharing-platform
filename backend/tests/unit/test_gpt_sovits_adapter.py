import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

from backend.app.services.gpt_sovits import (
    GPTSoVITSAdapter,
    CudaRuntimeInfo,
    ModelArtifactError,
    ModelRuntimeError,
    ModelRuntimeInfo,
    RawSynthesis,
    SynthesisSpec,
    TrainSpec,
)


@pytest.fixture
def adapter(tmp_path: Path):
    vendor_dir = tmp_path / "GPT-SoVITS"
    config_path = vendor_dir / "GPT_SoVITS" / "configs" / "s2v2Pro.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "train": {
                    "batch_size": 32,
                    "fp16_run": False,
                    "grad_ckpt": False,
                    "seed": 1234,
                },
                "data": {"exp_dir": ""},
                "model": {},
                "s2_ckpt_dir": "logs/s2/big2k1",
            }
        ),
        encoding="utf-8",
    )
    (vendor_dir / "GPT_SoVITS" / "s2_train.py").write_text("# test entrypoint\n", encoding="utf-8")
    return GPTSoVITSAdapter(vendor_dir=vendor_dir)


@pytest.fixture
def train_spec(tmp_path: Path):
    return TrainSpec(
        profile_id="v1",
        dataset_list=tmp_path / "dataset.list",
        output_dir=tmp_path / "profile" / "weights",
    )


def test_runtime_info_default_reports_eight_gb_class_floor(adapter):
    info = ModelRuntimeInfo(
        available=False,
        tag=adapter.TAG,
        device="cuda:0",
        executable=adapter.python_executable,
        vendor_dir=adapter.vendor_dir,
    )

    assert info.required_vram_bytes == 8000 * 1024**2


def test_train_command_uses_pinned_s2_entrypoint_and_generated_config(adapter, train_spec):
    command = adapter.build_train_command(train_spec)

    assert command == [
        str(adapter.python_executable),
        str(adapter.s2_train_script),
        "--config",
        str(train_spec.output_dir / "s2v2Pro-runtime.json"),
    ]
    assert train_spec.dataset_list.as_posix() not in command

    generated = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
    assert generated["train"] == {
        "batch_size": 1,
        "fp16_run": True,
        "grad_ckpt": True,
        "seed": 20260902,
        "gpu_numbers": "0",
        "pretrained_s2G": (adapter.models_root / "gpt-sovits" / "v2Pro" / "s2Gv2Pro.pth").as_posix(),
        "pretrained_s2D": (adapter.models_root / "gpt-sovits" / "v2Pro" / "s2Dv2Pro.pth").as_posix(),
        "if_save_latest": 1,
        "if_save_every_weights": True,
        "save_every_epoch": 1,
    }
    assert generated["data"]["exp_dir"] == train_spec.dataset_list.parent.as_posix()
    assert generated["s2_ckpt_dir"] == train_spec.output_dir.as_posix()
    assert generated["save_weight_dir"] == (train_spec.output_dir / "weights").as_posix()
    assert generated["name"] == train_spec.profile_id
    assert generated["version"] == "v2Pro"
    assert generated["model"]["version"] == "v2Pro"
    assert (train_spec.output_dir / "logs_s2_v2Pro").is_dir()
    assert (train_spec.dataset_list.parent / "logs_s2_v2Pro").is_dir()


def test_rejects_weight_outside_profile_directory(adapter, tmp_path: Path):
    raw_result = RawSynthesis(
        output_wav=tmp_path / "profile" / "output.wav",
        gpt_weight=tmp_path / "profile" / "gpt.pth",
        sovits_weight=Path("C:/outside/model.pth"),
        profile_dir=tmp_path / "profile",
    )

    with pytest.raises(ModelArtifactError):
        adapter.validate_weights(raw_result)


def test_run_rejects_unavailable_runtime_before_spawning_process(adapter, monkeypatch):
    adapter.vendor_dir.mkdir(exist_ok=True)
    runtime = ModelRuntimeInfo(
        available=False,
        tag=adapter.TAG,
        device="cuda:0",
        executable=adapter.python_executable,
        vendor_dir=adapter.vendor_dir,
        reason="CUDA runtime is unavailable",
    )
    monkeypatch.setattr(adapter, "probe", lambda: runtime)

    with pytest.raises(ModelRuntimeError, match="CUDA runtime is unavailable"):
        adapter._run([sys.executable, "-c", "raise SystemExit(0)"])


def test_run_exposes_cuda12_nvrtc_runtime_to_vendor_process(adapter, monkeypatch):
    runtime = ModelRuntimeInfo(
        available=True,
        tag=adapter.TAG,
        device="cuda:0",
        executable=adapter.python_executable,
        vendor_dir=adapter.vendor_dir,
    )
    monkeypatch.setattr(adapter, "probe", lambda: runtime)
    captured: dict[str, object] = {}

    def fake_run(command, *, cwd, env, **kwargs):
        del command, cwd, kwargs
        captured["env"] = env
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr("backend.app.services.gpt_sovits.subprocess.run", fake_run)
    adapter._run([sys.executable, "-c", "pass"])

    import nvidia.cuda_nvrtc

    nvrtc_bin = Path(nvidia.cuda_nvrtc.__path__[0]).resolve() / "bin"
    path_entries = str(captured["env"]["PATH"]).split(os.pathsep)
    assert str(nvrtc_bin) in path_entries


def test_probe_rejects_cuda_runtime_below_eight_gib(tmp_path: Path, monkeypatch):
    vendor_dir = tmp_path / "GPT-SoVITS"
    vendor_dir.mkdir()
    models_root = tmp_path / "models"
    model_path = models_root / "gpt-sovits" / "base.pth"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"verified-model")
    checksum_path = tmp_path / "checksums.sha256"
    checksum_path.write_text(
        f"{hashlib.sha256(model_path.read_bytes()).hexdigest()} models/gpt-sovits/base.pth\n",
        encoding="utf-8",
    )
    adapter = GPTSoVITSAdapter(
        vendor_dir=vendor_dir,
        models_root=models_root,
        checksum_path=checksum_path,
    )
    monkeypatch.setattr(adapter, "_pinned_checkout_tag", lambda: adapter.TAG)
    monkeypatch.setattr(
        adapter,
        "_cuda_runtime",
        lambda: CudaRuntimeInfo(
            torch_version="2.5.1+cu124",
            cuda_version="12.4",
            total_vram_bytes=8000 * 1024**2 - 1,
        ),
    )

    info = adapter.probe()

    assert info.available is False
    assert info.total_vram_bytes == 8000 * 1024**2 - 1
    assert info.required_vram_bytes == 8000 * 1024**2
    assert info.reason == "CUDA device 0 has less than required 8000 MiB VRAM"


def test_probe_accepts_8188_mib_eight_gib_class_gpu(tmp_path: Path, monkeypatch):
    vendor_dir = tmp_path / "GPT-SoVITS"
    vendor_dir.mkdir()
    models_root = tmp_path / "models"
    model_path = models_root / "gpt-sovits" / "base.pth"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"verified-model")
    checksum_path = tmp_path / "checksums.sha256"
    checksum_path.write_text(
        f"{hashlib.sha256(model_path.read_bytes()).hexdigest()} models/gpt-sovits/base.pth\n",
        encoding="utf-8",
    )
    adapter = GPTSoVITSAdapter(
        vendor_dir=vendor_dir,
        models_root=models_root,
        checksum_path=checksum_path,
    )
    monkeypatch.setattr(adapter, "_pinned_checkout_tag", lambda: adapter.TAG)
    monkeypatch.setattr(
        adapter,
        "_cuda_runtime",
        lambda: CudaRuntimeInfo(
            torch_version="2.5.1+cu124",
            cuda_version="12.4",
            total_vram_bytes=8188 * 1024**2,
        ),
    )

    info = adapter.probe()

    assert info.available is True
    assert info.total_vram_bytes == 8188 * 1024**2
    assert info.required_vram_bytes == 8000 * 1024**2


def test_probe_rejects_unpinned_torch_version_before_spawn(tmp_path: Path, monkeypatch):
    vendor_dir = tmp_path / "GPT-SoVITS"
    vendor_dir.mkdir()
    models_root = tmp_path / "models"
    model_path = models_root / "gpt-sovits" / "base.pth"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"verified-model")
    checksum_path = tmp_path / "checksums.sha256"
    checksum_path.write_text(
        f"{hashlib.sha256(model_path.read_bytes()).hexdigest()} models/gpt-sovits/base.pth\n",
        encoding="utf-8",
    )
    adapter = GPTSoVITSAdapter(
        vendor_dir=vendor_dir,
        models_root=models_root,
        checksum_path=checksum_path,
    )
    monkeypatch.setattr(adapter, "_pinned_checkout_tag", lambda: adapter.TAG)
    monkeypatch.setattr(
        adapter,
        "_cuda_runtime",
        lambda: CudaRuntimeInfo(
            torch_version="2.6.0+cu124",
            cuda_version="12.4",
            total_vram_bytes=8 * 1024**3,
        ),
    )

    info = adapter.probe()

    assert info.available is False
    assert info.reason == "PyTorch version is '2.6.0+cu124', expected 2.5.1"


def test_probe_rejects_empty_manifest_while_reporting_cuda_runtime(tmp_path: Path, monkeypatch):
    vendor_dir = tmp_path / "GPT-SoVITS"
    vendor_dir.mkdir()
    models_root = tmp_path / "models"
    models_root.mkdir()
    checksum_path = tmp_path / "checksums.sha256"
    checksum_path.write_text("# no models yet\n", encoding="utf-8")
    adapter = GPTSoVITSAdapter(
        vendor_dir=vendor_dir,
        models_root=models_root,
        checksum_path=checksum_path,
    )
    monkeypatch.setattr(adapter, "_pinned_checkout_tag", lambda: adapter.TAG)
    monkeypatch.setattr(
        adapter,
        "_cuda_runtime",
        lambda: CudaRuntimeInfo(
            torch_version="2.5.1+cu124",
            cuda_version="12.4",
            total_vram_bytes=8 * 1024**3,
        ),
    )

    info = adapter.probe()

    assert info.available is False
    assert info.verified_model_count == 0
    assert info.reason == "model checksum manifest has no entries"
    assert info.total_vram_bytes == 8 * 1024**3
    assert info.torch_version == "2.5.1+cu124"
    assert info.cuda_version == "12.4"


def test_synthesis_fails_closed_when_pinned_runtime_is_unavailable(adapter, tmp_path: Path):
    profile_dir = tmp_path / "profile"
    weights_dir = profile_dir / "weights"
    weights_dir.mkdir(parents=True)
    gpt_weight = weights_dir / "voice-gpt.ckpt"
    sovits_weight = weights_dir / "voice-sovits.pth"
    reference_audio = profile_dir / "reference.wav"
    gpt_weight.write_bytes(b"gpt")
    sovits_weight.write_bytes(b"sovits")
    reference_audio.write_bytes(b"reference")
    spec = SynthesisSpec(
        text="hello",
        text_lang="en",
        prompt_text="hello",
        prompt_lang="en",
        reference_audio=reference_audio,
        gpt_weight=gpt_weight,
        sovits_weight=sovits_weight,
        output_wav=profile_dir / "output.wav",
    )

    with pytest.raises(ModelRuntimeError, match="runtime is unavailable"):
        adapter.synthesize(spec)


def test_train_fails_closed_when_pinned_runtime_is_unavailable(adapter, train_spec):
    with pytest.raises(ModelRuntimeError, match="runtime is unavailable"):
        adapter.train(train_spec)
