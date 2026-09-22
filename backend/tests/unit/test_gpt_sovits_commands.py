from pathlib import Path

import yaml
import pytest

from backend.app.services.gpt_sovits import (
    GPTSoVITSAdapter,
    ModelRuntimeError,
    SynthesisSpec,
    TrainSpec,
)


def test_v2pro_commands_point_to_pinned_vendor_scripts(tmp_path: Path):
    adapter = GPTSoVITSAdapter(vendor_dir=Path("vendor/GPT-SoVITS"))
    s1 = adapter.build_s1_train_command(tmp_path / "s1.yaml")
    s2 = adapter.build_s2_train_command(tmp_path / "s2.json")
    spec = SynthesisSpec(
        text="hello",
        text_lang="en",
        prompt_text="authorized reference",
        prompt_lang="zh",
        reference_audio=tmp_path / "reference.wav",
        gpt_weight=tmp_path / "profile" / "gpt.ckpt",
        sovits_weight=tmp_path / "profile" / "sovits.pth",
        output_wav=tmp_path / "output" / "audio.wav",
    )
    infer = adapter.build_inference_command(spec)

    assert s1[1].replace("\\", "/").endswith("GPT_SoVITS/s1_train.py")
    assert s1[s1.index("--config_file") + 1] == str((tmp_path / "s1.yaml").resolve())
    assert s2[1].replace("\\", "/").endswith("GPT_SoVITS/s2_train.py")
    assert "--config" in s2
    assert infer[1].replace("\\", "/").endswith("GPT_SoVITS/inference_cli.py")
    assert "--gpt_model" in infer and "--sovits_model" in infer
    assert infer[infer.index("--ref_language") + 1] == "中文"
    assert infer[infer.index("--target_language") + 1] == "英文"


def test_training_configs_keep_single_gpu_fp16_contract(tmp_path: Path):
    adapter = GPTSoVITSAdapter(vendor_dir=Path("vendor/GPT-SoVITS"))
    dataset_list = tmp_path / "prepared" / "6-name2semantic.tsv"
    dataset_list.parent.mkdir(parents=True)
    dataset_list.write_text("segment\t[1]\n", encoding="utf-8")
    spec = TrainSpec(profile_id="profile-1", dataset_list=dataset_list, output_dir=tmp_path / "run")

    s1_path = adapter.write_s1_training_config(spec)
    s2_path = adapter.write_s2_training_config(spec)
    s1 = yaml.safe_load(s1_path.read_text(encoding="utf-8"))
    s2 = yaml.safe_load(s2_path.read_text(encoding="utf-8")) if s2_path.suffix in {".yaml", ".yml"} else __import__("json").loads(s2_path.read_text(encoding="utf-8"))

    assert s1["train"]["batch_size"] == 1
    assert s1["train"]["precision"] == "16-mixed"
    assert s2["train"]["batch_size"] == 1
    assert s2["train"]["fp16_run"] is True
    assert s2["train"]["grad_ckpt"] is True
    assert s2["train"]["gpu_numbers"] == "0"


def test_command_builders_fail_closed_when_pinned_script_is_missing(tmp_path: Path):
    adapter = GPTSoVITSAdapter(vendor_dir=tmp_path / "missing-vendor")

    with pytest.raises(ModelRuntimeError, match="vendor script is missing"):
        adapter.build_s1_train_command(tmp_path / "s1.yaml")
