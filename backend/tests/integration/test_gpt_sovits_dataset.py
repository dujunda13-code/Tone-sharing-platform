from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf
import pytest

from backend.app.schemas.dataset import DatasetManifest, DatasetManifestRow
from backend.app.services.gpt_sovits_dataset import GPTSoVITSDatasetPreparer
from backend.app.workers.gpu_worker import TrainingPipelineError


def _manifest(tmp_path: Path, *, path: str = "datasets/d1/segments/seg-0001.wav") -> DatasetManifest:
    row = DatasetManifestRow(
        segment_id="seg-0001",
        path=path,
        speaker="voice-d1",
        language="zh",
        text="本地授权训练样本",
        duration_seconds=6.0,
        snr_db=30.0,
        clipping_ratio=0.0,
        split="train",
    )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(row.model_dump_json() + "\n", encoding="utf-8")
    return DatasetManifest(
        dataset_id="d1",
        effective_seconds=600.0,
        rows=[row],
        manifest_path=manifest_path,
    )


def _fake_vendor_models(tmp_path: Path) -> tuple[Path, Path, Path]:
    models = tmp_path / "models"
    (models / "gpt-sovits" / "bert").mkdir(parents=True)
    (models / "gpt-sovits" / "chinese-hubert-base").mkdir(parents=True)
    g2pw = models / "gpt-sovits" / "GPT_SoVITS" / "text" / "G2PWModel"
    g2pw.mkdir(parents=True)
    for name in (
        "g2pW.onnx",
        "config.py",
        "POLYPHONIC_CHARS.txt",
        "MONOPHONIC_CHARS.txt",
        "bopomofo_to_pinyin_wo_tune_dict.json",
        "char_bopomofo_dict.json",
    ):
        (g2pw / name).write_text("test", encoding="utf-8")
    s2g = models / "gpt-sovits" / "v2Pro" / "s2Gv2Pro.pth"
    s2g.parent.mkdir(parents=True)
    s2g.write_bytes(b"test-weight")
    sv_model = models / "gpt-sovits" / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt"
    sv_model.parent.mkdir(parents=True)
    sv_model.write_bytes(b"test-speaker-vector-weight")
    return models, models / "gpt-sovits" / "bert", models / "gpt-sovits" / "chinese-hubert-base"


def _fake_runner(command, *, cwd, env, timeout):
    del cwd, timeout
    script = Path(command[1]).name
    output = Path(env["opt_dir"])
    audio_name = Path(env["inp_text"]).read_text(encoding="utf-8").split("|", 1)[0]
    if script == "1-get-text.py":
        (output / "3-bert").mkdir(parents=True, exist_ok=True)
        (output / f"2-name2text-{env['i_part']}.txt").write_text(
            f"{audio_name}\tphone\t[1]\t本地授权训练样本\n", encoding="utf-8"
        )
        (output / "3-bert" / f"{audio_name}.pt").write_bytes(b"bert")
    elif script == "2-get-hubert-wav32k.py":
        (output / "4-cnhubert").mkdir(parents=True, exist_ok=True)
        (output / "5-wav32k").mkdir(parents=True, exist_ok=True)
        (output / "4-cnhubert" / f"{audio_name}.pt").write_bytes(b"hubert")
        (output / "5-wav32k" / audio_name).write_bytes(b"wav")
    elif script == "2-get-sv.py":
        (output / "7-sv_cn").mkdir(parents=True, exist_ok=True)
        (output / "7-sv_cn" / f"{audio_name}.pt").write_bytes(b"speaker-vector")
    elif script == "3-get-semantic.py":
        (output / f"6-name2semantic-{env['i_part']}.tsv").write_text(
            f"{audio_name}\t1 2 3\n", encoding="utf-8"
        )
    return subprocess.CompletedProcess(command, 0, "", "")


def test_stage_environment_resolves_gpt_sovits_text_and_vendor_tools(tmp_path: Path):
    """The real subprocess must see both import roots used by official scripts."""
    vendor_dir = tmp_path / "vendor"
    script = vendor_dir / "GPT_SoVITS" / "prepare_datasets" / "imports.py"
    package_root = vendor_dir / "GPT_SoVITS"
    text_dir = package_root / "text"
    tools_dir = vendor_dir / "tools"
    script.parent.mkdir(parents=True)
    text_dir.mkdir()
    tools_dir.mkdir()
    model_runtime_root = tmp_path / "models" / "gpt-sovits"
    model_runtime_root.mkdir(parents=True)
    (text_dir / "__init__.py").write_text("", encoding="utf-8")
    (text_dir / "cleaner.py").write_text(
        "def clean_text(value):\n    return value\n", encoding="utf-8"
    )
    (tools_dir / "__init__.py").write_text("", encoding="utf-8")
    (tools_dir / "my_utils.py").write_text(
        "def clean_path(value):\n    return value\n", encoding="utf-8"
    )
    script.write_text(
        "from pathlib import Path\n"
        "from text.cleaner import clean_text\n"
        "from tools.my_utils import clean_path\n"
        "Path('stage-cwd.txt').write_text(str(Path.cwd()), encoding='utf-8')\n",
        encoding="utf-8",
    )
    preparer = GPTSoVITSDatasetPreparer(
        storage_root=tmp_path / "data",
        models_root=tmp_path / "models",
        vendor_dir=vendor_dir,
        python_executable=Path(sys.executable),
    )

    preparer._run_stage("imports", "imports.py", {}, tmp_path / "output")

    assert (tmp_path / "output" / "logs" / "imports.stderr.log").read_text(encoding="utf-8") == ""
    assert (model_runtime_root / "stage-cwd.txt").read_text(encoding="utf-8") == str(
        model_runtime_root
    )


def test_speaker_vector_stage_resolves_vendor_eres2net_import(tmp_path: Path):
    vendor_dir = tmp_path / "vendor"
    script_root = vendor_dir / "GPT_SoVITS" / "prepare_datasets"
    eres2net_dir = vendor_dir / "GPT_SoVITS" / "eres2net"
    script_root.mkdir(parents=True)
    eres2net_dir.mkdir(parents=True)
    (eres2net_dir / "ERes2NetV2.py").write_text(
        "class ERes2NetV2:\n    pass\n", encoding="utf-8"
    )
    (script_root / "2-get-sv.py").write_text(
        "from ERes2NetV2 import ERes2NetV2\n"
        "from pathlib import Path\n"
        "Path('speaker-vector-import.txt').write_text(ERes2NetV2.__name__, encoding='utf-8')\n",
        encoding="utf-8",
    )
    model_runtime_root = tmp_path / "models" / "gpt-sovits"
    model_runtime_root.mkdir(parents=True)
    preparer = GPTSoVITSDatasetPreparer(
        storage_root=tmp_path / "data",
        models_root=tmp_path / "models",
        vendor_dir=vendor_dir,
        python_executable=Path(sys.executable),
    )

    preparer._run_stage(
        "speaker_vector", "2-get-sv.py", {}, tmp_path / "output"
    )

    assert (model_runtime_root / "speaker-vector-import.txt").read_text(
        encoding="utf-8"
    ) == "ERes2NetV2"


def test_stage_environment_exposes_local_ffmpeg_command(tmp_path: Path):
    storage_root = tmp_path / "data"
    bundled_binary = tmp_path / "toolchain" / "ffmpeg-win-x86_64-v7.1.exe"
    bundled_binary.parent.mkdir(parents=True)
    bundled_binary.write_bytes(b"bundled-local-ffmpeg")
    preparer = GPTSoVITSDatasetPreparer(
        storage_root=storage_root,
        models_root=tmp_path / "models",
        vendor_dir=tmp_path / "vendor",
        ffmpeg_executable=bundled_binary,
    )

    environment = preparer._base_environment(
        tmp_path / "input.txt", tmp_path / "wav", tmp_path / "output", "dataset-1"
    )

    alias = storage_root / "runtime" / "bin" / "ffmpeg.exe"
    assert alias.read_bytes() == bundled_binary.read_bytes()
    assert environment["PATH"].split(os.pathsep)[0] == str(alias.parent)
    assert environment["FFMPEG_BINARY"] == str(alias)


def test_stage_environment_exposes_cuda12_nvrtc_runtime(tmp_path: Path):
    preparer = GPTSoVITSDatasetPreparer(
        storage_root=tmp_path / "data",
        models_root=tmp_path / "models",
        vendor_dir=tmp_path / "vendor",
        ffmpeg_executable=tmp_path / "ffmpeg.exe",
    )
    preparer.ffmpeg_executable.write_bytes(b"ffmpeg")

    environment = preparer._base_environment(
        tmp_path / "input.txt", tmp_path / "wav", tmp_path / "output", "dataset-1"
    )

    import nvidia.cuda_nvrtc

    nvrtc_bin = Path(nvidia.cuda_nvrtc.__path__[0]).resolve() / "bin"
    assert str(nvrtc_bin) in environment["PATH"].split(os.pathsep)
    assert (nvrtc_bin / "nvrtc-builtins64_124.dll").is_file()


def test_pinned_vendor_audio_loader_uses_the_deployed_ffmpeg_path():
    root = Path(__file__).resolve().parents[3]
    source = (root / "vendor" / "GPT-SoVITS" / "tools" / "my_utils.py").read_text(
        encoding="utf-8"
    )

    assert 'os.environ.get("FFMPEG_BINARY", "ffmpeg")' in source


def test_fixed_vendor_semantic_frontend_imports_x_transformers_offline():
    root = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()
    vendor_dir = root / "vendor" / "GPT-SoVITS"
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(vendor_dir / "GPT_SoVITS"), str(vendor_dir))
    )
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"

    result = subprocess.run(
        [sys.executable, "-c", "from f5_tts.model.backbones.dit import DiT; print(DiT.__name__)"],
        cwd=root / "models" / "gpt-sovits",
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "DiT"


def test_fixed_gpt_sovits_training_runtime_imports_tensorboard_writer():
    result = subprocess.run(
        [sys.executable, "-c", "from torch.utils.tensorboard import SummaryWriter; print(SummaryWriter.__name__)"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "SummaryWriter"


def test_prepare_requires_deployed_g2pw_model_before_vendor_scripts_run(tmp_path: Path):
    storage_root = tmp_path / "data"
    source = storage_root / "datasets" / "d1" / "segments" / "seg-0001.wav"
    source.parent.mkdir(parents=True)
    sf.write(source, np.zeros(16_000, dtype=np.float32), 16_000)
    models, _, _ = _fake_vendor_models(tmp_path)
    missing_model = models / "gpt-sovits" / "GPT_SoVITS" / "text" / "G2PWModel" / "g2pW.onnx"
    missing_model.unlink()
    vendor_dir = tmp_path / "vendor"
    scripts = vendor_dir / "GPT_SoVITS" / "prepare_datasets"
    scripts.mkdir(parents=True)
    for name in ("1-get-text.py", "2-get-hubert-wav32k.py", "2-get-sv.py", "3-get-semantic.py"):
        (scripts / name).write_text("# test entrypoint\n", encoding="utf-8")
    config = vendor_dir / "GPT_SoVITS" / "configs" / "s2v2Pro.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}", encoding="utf-8")

    def runner(*args, **kwargs):
        pytest.fail("the vendor scripts must not start when the local G2PW model is incomplete")

    preparer = GPTSoVITSDatasetPreparer(
        storage_root=storage_root,
        models_root=models,
        vendor_dir=vendor_dir,
        command_runner=runner,
    )

    with pytest.raises(TrainingPipelineError, match="G2PW"):
        preparer.prepare(
            _manifest(tmp_path),
            storage_root / "features" / "d1" / "job-1",
        )


def test_prepare_rejects_manifest_without_authorized_local_audio(tmp_path: Path):
    models, _, _ = _fake_vendor_models(tmp_path)
    preparer = GPTSoVITSDatasetPreparer(
        storage_root=tmp_path / "data",
        models_root=models,
        vendor_dir=tmp_path / "vendor",
        command_runner=_fake_runner,
    )

    with pytest.raises(TrainingPipelineError, match="local storage"):
        preparer.prepare(
            _manifest(tmp_path, path="C:/outside/audio.wav"),
            tmp_path / "data" / "features" / "d1" / "job-1",
        )


def test_prepare_writes_vendor_dataset_contract(tmp_path: Path):
    storage_root = tmp_path / "data"
    source = storage_root / "datasets" / "d1" / "segments" / "seg-0001.wav"
    source.parent.mkdir(parents=True)
    sf.write(source, np.zeros(16_000, dtype=np.float32), 16_000)
    models, _, _ = _fake_vendor_models(tmp_path)
    vendor_dir = tmp_path / "vendor"
    scripts = vendor_dir / "GPT_SoVITS" / "prepare_datasets"
    scripts.mkdir(parents=True)
    for name in ("1-get-text.py", "2-get-hubert-wav32k.py", "2-get-sv.py", "3-get-semantic.py"):
        (scripts / name).write_text("# test entrypoint\n", encoding="utf-8")
    config = vendor_dir / "GPT_SoVITS" / "configs" / "s2v2Pro.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}", encoding="utf-8")
    preparer = GPTSoVITSDatasetPreparer(
        storage_root=storage_root,
        models_root=models,
        vendor_dir=vendor_dir,
        command_runner=_fake_runner,
    )

    prepared = preparer.prepare(
        _manifest(tmp_path),
        storage_root / "features" / "d1" / "job-1",
    )

    assert prepared.text_list.is_file()
    assert prepared.feature_dir.is_dir()
    assert prepared.speaker_vector_dir.is_dir()
    assert prepared.feature_manifest.is_file()
    feature_manifest = json.loads(prepared.feature_manifest.read_text(encoding="utf-8"))
    assert feature_manifest["dataset_id"] == "d1"
    assert feature_manifest["segments"] == [
        {
            "segment_id": "seg-0001",
            "speaker_id": "voice-d1",
            "wav32k": "5-wav32k/000001-seg-0001.wav",
        }
    ]
    assert prepared.semantic_list.is_file()
    assert prepared.bert_dir.is_dir()
    assert prepared.split_lists["train"].read_text(encoding="utf-8").strip()
    assert len(prepared.content_hash) == 64


def test_prepare_requires_deployed_v2pro_speaker_vector_model(tmp_path: Path):
    storage_root = tmp_path / "data"
    source = storage_root / "datasets" / "d1" / "segments" / "seg-0001.wav"
    source.parent.mkdir(parents=True)
    sf.write(source, np.zeros(16_000, dtype=np.float32), 16_000)
    models, _, _ = _fake_vendor_models(tmp_path)
    (models / "gpt-sovits" / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt").unlink()
    vendor_dir = tmp_path / "vendor"
    scripts = vendor_dir / "GPT_SoVITS" / "prepare_datasets"
    scripts.mkdir(parents=True)
    for name in ("1-get-text.py", "2-get-hubert-wav32k.py", "2-get-sv.py", "3-get-semantic.py"):
        (scripts / name).write_text("# test entrypoint\n", encoding="utf-8")
    config = vendor_dir / "GPT_SoVITS" / "configs" / "s2v2Pro.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}", encoding="utf-8")

    preparer = GPTSoVITSDatasetPreparer(
        storage_root=storage_root,
        models_root=models,
        vendor_dir=vendor_dir,
        command_runner=_fake_runner,
    )

    with pytest.raises(TrainingPipelineError, match="speaker vector"):
        preparer.prepare(
            _manifest(tmp_path),
            storage_root / "features" / "d1" / "job-1",
        )
