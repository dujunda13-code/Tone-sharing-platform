from __future__ import annotations

import json
from pathlib import Path
import subprocess

import torch
import yaml

from backend.app.services.gpt_sovits import (
    GPTSoVITSAdapter,
    ModelRuntimeInfo,
    TrainSpec,
)


def _adapter(tmp_path: Path, runner):
    vendor = tmp_path / "vendor" / "GPT-SoVITS"
    gpt_dir = vendor / "GPT_SoVITS"
    (gpt_dir / "configs").mkdir(parents=True)
    (gpt_dir / "s1_train.py").write_text("# test entrypoint\n", encoding="utf-8")
    (gpt_dir / "s2_train.py").write_text("# test entrypoint\n", encoding="utf-8")
    (gpt_dir / "configs" / "s1longer-v2.yaml").write_text(
        yaml.safe_dump(
            {
                "train": {"batch_size": 8, "precision": "32", "seed": 1},
                "optimizer": {},
                "data": {"num_workers": 4},
                "model": {},
            }
        ),
        encoding="utf-8",
    )
    (gpt_dir / "configs" / "s2v2Pro.json").write_text(
        json.dumps(
            {
                "train": {"batch_size": 8, "fp16_run": False, "grad_ckpt": False},
                "data": {},
                "model": {},
            }
        ),
        encoding="utf-8",
    )
    models_root = tmp_path / "models"
    model_dir = models_root / "gpt-sovits" / "v2Pro"
    model_dir.mkdir(parents=True)
    for name in ("s1v3.ckpt", "s2Gv2Pro.pth", "s2Dv2Pro.pth"):
        (model_dir / name).write_bytes(b"test-base-model")
    adapter = GPTSoVITSAdapter(
        vendor_dir=vendor,
        models_root=models_root,
        command_runner=runner,
    )
    adapter.probe = lambda: ModelRuntimeInfo(
        available=True,
        tag=adapter.TAG,
        device="cuda:0",
        executable=adapter.python_executable,
        vendor_dir=adapter.vendor_dir,
        total_vram_bytes=8188 * 1024**2,
        torch_version="2.5.1+cu124",
        cuda_version="12.4",
        verified_model_count=4,
    )
    return adapter


def _spec(tmp_path: Path) -> TrainSpec:
    dataset_dir = tmp_path / "features" / "d1"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "6-name2semantic.tsv").write_text("sample.wav\t1 2 3\n", encoding="utf-8")
    (dataset_dir / "2-name2text.txt").write_text("sample.wav\tphone\t[1]\t文本\n", encoding="utf-8")
    return TrainSpec(
        profile_id="profile-1",
        dataset_list=dataset_dir / "6-name2semantic.tsv",
        output_dir=tmp_path / "profile" / "run",
        profile_dir=tmp_path / "profile",
    )


def test_training_adapter_returns_real_weight_paths_only_after_process_success(tmp_path: Path):
    calls: list[str] = []

    def runner(command, *, cwd, env, timeout):
        del cwd, env, timeout
        calls.append(Path(command[1]).name)
        config_path = Path(command[-1])
        if Path(command[1]).name == "s2_train.py":
            config = json.loads(config_path.read_text(encoding="utf-8"))
            weight = Path(config["save_weight_dir"]) / "profile-1_e1_s1.pth"
        else:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            weight = Path(config["train"]["half_weights_save_dir"]) / "profile-1-e1.ckpt"
        weight.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"weight": {"sample": torch.ones(1)}}, weight)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    adapter = _adapter(tmp_path, runner)
    result = adapter.train(_spec(tmp_path))

    assert calls == ["s2_train.py", "s1_train.py"]
    assert result.profile_dir == (tmp_path / "profile").resolve()
    assert result.gpt_weight.is_file()
    assert result.sovits_weight.is_file()
    assert result.gpt_weight.resolve().is_relative_to(tmp_path / "profile" / "run")
    assert result.sovits_weight.resolve().is_relative_to(tmp_path / "profile" / "run")


def test_training_adapter_does_not_return_weights_after_process_failure(tmp_path: Path):
    def runner(command, *, cwd, env, timeout):
        del cwd, env, timeout
        return subprocess.CompletedProcess(command, 7, "", "failed")

    adapter = _adapter(tmp_path, runner)

    try:
        adapter.train(_spec(tmp_path))
    except Exception as exc:
        assert "exited with 7" in str(exc)
    else:
        raise AssertionError("failed GPT-SoVITS process must not return trained weights")
