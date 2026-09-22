"""Real WebUI runner execution contract and bridge output gating.

These integration tests execute ``backend.app.services.webui_runner.main()``
in-process with only the heavy vendor module substituted, verifying the
real config parsing, parameter mapping, seed handling and the guarded
output WAV spec, plus the bridge's fail-closed validation of that spec.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from subprocess import CompletedProcess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from backend.app.services import webui_runner
from backend.app.services.webui_bridge import WebUIInferenceBridge, WebUISynthesisSpec


class FakeVendorWebUI:
    """Stands in for the official inference_webui module imported by the runner."""

    def __init__(self, sample_rate: int = 32_000) -> None:
        self.sample_rate = sample_rate
        self.set_seed_calls: list[int] = []
        self.get_tts_wav_calls: list[dict] = []

    @staticmethod
    def i18n(text: str) -> str:
        return text

    def set_seed(self, seed: int) -> None:
        self.set_seed_calls.append(seed)

    def get_tts_wav(self, **kwargs):
        self.get_tts_wav_calls.append(kwargs)
        yield self.sample_rate, np.zeros(3200, dtype=np.float32)


def _write_config(
    tmp_path: Path,
    output_wav: Path,
    *,
    seed: int = 12345,
    auxiliary: list[str] | None = None,
) -> Path:
    ref = tmp_path / "ref.wav"
    sf.write(ref, np.zeros(3200, dtype=np.float32), 32_000)
    config: dict[str, object] = {
        "text": "合成目标文本。",
        "text_lang": "zh",
        "prompt_text": "参考音频文本。",
        "prompt_lang": "zh",
        "reference_audio": str(ref),
        "gpt_weight": str(tmp_path / "s1.ckpt"),
        "sovits_weight": str(tmp_path / "s2.pth"),
        "output_wav": str(output_wav),
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        "how_to_cut": "不切",
        "sample_steps": 8,
        "speed": 1.0,
        "pause_second": 0.3,
        "seed": seed,
        "version": "v2ProPlus",
    }
    if auxiliary is not None:
        config["auxiliary_audios"] = auxiliary
    config_path = tmp_path / "run_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return config_path


@pytest.fixture
def restored_runner_env(monkeypatch):
    """The runner mutates os.environ while importing vendor env vars."""
    keys = ("version", "gpt_path", "sovits_path")
    saved = {key: os.environ.get(key) for key in keys}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_runner_executes_real_main_and_writes_guarded_wav_spec(
    tmp_path: Path, monkeypatch, restored_runner_env
):
    fake = FakeVendorWebUI()
    monkeypatch.setitem(sys.modules, "inference_webui", fake)
    output_wav = tmp_path / "out.wav"
    config_path = _write_config(tmp_path, output_wav)
    monkeypatch.setattr(sys, "argv", ["webui_runner", "--config", str(config_path)])

    webui_runner.main()

    info = sf.info(str(output_wav))
    assert info.samplerate == 32_000
    assert info.channels == 1
    assert info.subtype == "PCM_16"

    assert fake.set_seed_calls == [12345]
    assert len(fake.get_tts_wav_calls) == 1
    call = fake.get_tts_wav_calls[0]
    assert call["prompt_language"] == "中文"
    assert call["text_language"] == "中文"
    assert call["how_to_cut"] == "不切"
    assert call["top_k"] == 15
    assert call["top_p"] == 1.0
    assert call["temperature"] == 1.0
    assert call["sample_steps"] == 8
    assert call["ref_free"] is False
    assert call["if_freeze"] is False
    assert call["if_sr"] is False
    assert call["speed"] == 1.0
    assert call["pause_second"] == 0.3
    assert call["inp_refs"] is None


def test_runner_maps_auxiliary_references_to_inp_refs(
    tmp_path: Path, monkeypatch, restored_runner_env
):
    fake = FakeVendorWebUI()
    monkeypatch.setitem(sys.modules, "inference_webui", fake)
    output_wav = tmp_path / "out.wav"
    aux = tmp_path / "aux.wav"
    sf.write(aux, np.zeros(3200, dtype=np.float32), 32_000)
    config_path = _write_config(tmp_path, output_wav, auxiliary=[str(aux)])
    monkeypatch.setattr(sys, "argv", ["webui_runner", "--config", str(config_path)])

    webui_runner.main()

    inp_refs = fake.get_tts_wav_calls[0]["inp_refs"]
    assert inp_refs is not None
    assert [Path(item.name).name for item in inp_refs] == ["aux.wav"]
    assert all(isinstance(item, SimpleNamespace) for item in inp_refs)


def _bridge_spec(tmp_path: Path, output_wav: Path) -> WebUISynthesisSpec:
    ref = tmp_path / "ref.wav"
    sf.write(ref, np.zeros(3200, dtype=np.float32), 32_000)
    s1 = tmp_path / "s1.ckpt"
    s1.write_bytes(b"dummy_s1")
    s2 = tmp_path / "s2.pth"
    s2.write_bytes(b"dummy_s2")
    return WebUISynthesisSpec(
        text="合成目标文本。",
        text_lang="zh",
        prompt_text="参考音频文本。",
        prompt_lang="zh",
        reference_audio=ref,
        gpt_weight=s1,
        sovits_weight=s2,
        output_wav=output_wav,
    )


def test_bridge_fails_closed_when_output_wav_violates_guarded_spec(tmp_path: Path):
    spec = _bridge_spec(tmp_path, tmp_path / "out.wav")

    def wrong_spec_runner(cmd, cwd, env, timeout):
        # 16 kHz stereo violates the guarded PCM 16-bit 32kHz mono contract.
        sf.write(spec.output_wav, np.zeros((3200, 2), dtype=np.float32), 16_000)
        return CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    bridge = WebUIInferenceBridge(
        python_executable=Path("python.exe"),
        vendor_dir=tmp_path,
        command_runner=wrong_spec_runner,
    )

    with pytest.raises(RuntimeError, match="PCM 16-bit 32kHz"):
        bridge.synthesize(spec)


def test_bridge_accepts_output_wav_meeting_guarded_spec(tmp_path: Path):
    spec = _bridge_spec(tmp_path, tmp_path / "out.wav")

    def guarded_runner(cmd, cwd, env, timeout):
        sf.write(spec.output_wav, np.zeros(3200, dtype=np.float32), 32_000)
        return CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    bridge = WebUIInferenceBridge(
        python_executable=Path("python.exe"),
        vendor_dir=tmp_path,
        command_runner=guarded_runner,
    )

    assert bridge.synthesize(spec) == spec.output_wav
