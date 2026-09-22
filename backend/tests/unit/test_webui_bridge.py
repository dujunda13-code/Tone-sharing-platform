from pathlib import Path
import json
import pytest

from backend.app.services.webui_bridge import (
    WebUIInferenceBridge,
    WebUISynthesisSpec,
)


def test_webui_synthesis_spec_defaults():
    spec = WebUISynthesisSpec(
        text="合成目标文本。",
        text_lang="zh",
        prompt_text="参考音频文本。",
        prompt_lang="zh",
        reference_audio=Path("data/ref.wav"),
        gpt_weight=Path("models/s1.ckpt"),
        sovits_weight=Path("models/s2.pth"),
        output_wav=Path("data/out.wav"),
    )
    assert spec.top_k == 15
    assert spec.top_p == 1.0
    assert spec.temperature == 1.0
    assert spec.how_to_cut == "不切"
    assert spec.sample_steps == 8
    assert spec.speed == 1.0
    assert spec.pause_second == 0.3
    assert spec.auxiliary_audios == ()


def test_webui_bridge_build_command_and_payload(tmp_path):
    bridge = WebUIInferenceBridge(
        python_executable=Path("python.exe"),
        vendor_dir=tmp_path / "vendor",
    )
    ref_wav = tmp_path / "ref.wav"
    ref_wav.write_bytes(b"dummy_ref")
    s1_ckpt = tmp_path / "s1.ckpt"
    s1_ckpt.write_bytes(b"dummy_s1")
    s2_pth = tmp_path / "s2.pth"
    s2_pth.write_bytes(b"dummy_s2")
    aux1 = tmp_path / "aux1.wav"
    aux1.write_bytes(b"dummy_aux1")
    aux2 = tmp_path / "aux2.wav"
    aux2.write_bytes(b"dummy_aux2")

    spec = WebUISynthesisSpec(
        text="合成目标文本。",
        text_lang="zh",
        prompt_text="参考音频文本。",
        prompt_lang="zh",
        reference_audio=ref_wav,
        gpt_weight=s1_ckpt,
        sovits_weight=s2_pth,
        output_wav=tmp_path / "out.wav",
        auxiliary_audios=(aux1, aux2),
        top_k=15,
        top_p=1.0,
        temperature=1.0,
        seed=12345,
    )
    config_file = tmp_path / "config.json"
    cmd, env = bridge.build_command_and_env(spec, config_file)

    assert Path(cmd[0]).name == "python.exe"
    assert "-m" in cmd
    assert "backend.app.services.webui_runner" in cmd
    assert str(config_file.resolve()) in cmd

    assert config_file.is_file()
    payload = json.loads(config_file.read_text(encoding="utf-8"))
    assert payload["top_k"] == 15
    assert payload["top_p"] == 1.0
    assert payload["temperature"] == 1.0
    assert payload["seed"] == 12345
    assert len(payload["auxiliary_audios"]) == 2
    assert payload["version"] == "v2ProPlus"

    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["version"] == "v2ProPlus"
    assert "s1.ckpt" in env["gpt_path"]
    assert "s2.pth" in env["sovits_path"]


def test_webui_bridge_validates_required_files(tmp_path):
    bridge = WebUIInferenceBridge(
        python_executable=Path("python.exe"),
        vendor_dir=tmp_path / "vendor",
    )
    spec = WebUISynthesisSpec(
        text="测试文本",
        text_lang="zh",
        prompt_text="参考文本",
        prompt_lang="zh",
        reference_audio=tmp_path / "missing_ref.wav",
        gpt_weight=tmp_path / "s1.ckpt",
        sovits_weight=tmp_path / "s2.pth",
        output_wav=tmp_path / "out.wav",
    )
    with pytest.raises(FileNotFoundError, match="参考音频不存在"):
        bridge.validate_spec(spec)
