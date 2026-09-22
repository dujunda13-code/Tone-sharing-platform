from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import yaml

from backend.app.services.gpt_sovits import (
    GPTSoVITSAdapter,
    ModelRuntimeInfo,
    RawSynthesis,
    SynthesisSpec,
)
from backend.app.schemas.synthesis import EmotionControl, SynthesisCreate
from backend.app.services.synthesis_pipeline import (
    GPTSoVITSLocalSynthesizer,
    ReferenceCondition,
)


def _adapter(tmp_path: Path, runner):
    vendor = tmp_path / "vendor" / "GPT-SoVITS"
    gpt_dir = vendor / "GPT_SoVITS"
    (gpt_dir / "configs").mkdir(parents=True)
    (gpt_dir / "inference_cli.py").write_text("# test entrypoint\n", encoding="utf-8")
    (gpt_dir / "configs" / "s1longer-v2.yaml").write_text(
        yaml.safe_dump({"train": {}, "data": {}}), encoding="utf-8"
    )
    (gpt_dir / "configs" / "s2v2Pro.json").write_text(
        json.dumps({"train": {}, "data": {}, "model": {}}), encoding="utf-8"
    )
    adapter = GPTSoVITSAdapter(vendor_dir=vendor, command_runner=runner)
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


def _spec(tmp_path: Path) -> SynthesisSpec:
    profile_dir = tmp_path / "profile" / "public" / "weights"
    profile_dir.mkdir(parents=True)
    gpt_weight = profile_dir / "profile-e1.ckpt"
    sovits_weight = profile_dir / "profile_e1_s1.pth"
    gpt_weight.write_bytes(b"gpt")
    sovits_weight.write_bytes(b"sovits")
    reference_audio = tmp_path / "reference.wav"
    sf.write(reference_audio, np.zeros(16_000, dtype=np.float32), 16_000)
    return SynthesisSpec(
        text="目标文本",
        text_lang="zh",
        prompt_text="参考文本",
        prompt_lang="zh",
        reference_audio=reference_audio,
        gpt_weight=gpt_weight,
        sovits_weight=sovits_weight,
        output_wav=tmp_path / "synthesis" / "raw.wav",
    )


def test_inference_adapter_returns_only_vendor_output_moved_to_requested_wav(tmp_path: Path):
    calls: list[list[str]] = []

    def runner(command, *, cwd, env, timeout):
        del cwd, env, timeout
        calls.append(command)
        output_dir = Path(command[command.index("--output_path") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        sf.write(output_dir / "output.wav", np.zeros(16_000, dtype=np.float32), 16_000)
        return subprocess.CompletedProcess(command, 0, "Audio saved", "")

    adapter = _adapter(tmp_path, runner)
    result = adapter.synthesize(_spec(tmp_path))

    assert result.output_wav.name == "raw.wav"
    assert result.output_wav.is_file()
    assert calls[0][1].replace("\\", "/").endswith("GPT_SoVITS/inference_cli.py")
    assert calls[0][calls[0].index("--ref_language") + 1] == "中文"
    assert calls[0][calls[0].index("--target_language") + 1] == "中文"
    assert not (result.output_wav.parent / "output.wav").exists()


def test_inference_adapter_rejects_nonzero_vendor_process(tmp_path: Path):
    def runner(command, *, cwd, env, timeout):
        del cwd, env, timeout
        return subprocess.CompletedProcess(command, 9, "", "vendor failure")

    adapter = _adapter(tmp_path, runner)

    try:
        adapter.synthesize(_spec(tmp_path))
    except Exception as exc:
        assert "exited with 9" in str(exc)
    else:
        raise AssertionError("failed inference process must not return an audio path")


def test_inference_subprocess_env_pins_import_time_model_paths(tmp_path: Path):
    """The vendored inference_webui runs change_gpt_weights(gpt_path) at module
    import, resolving gpt_path from the process environment, so the inference
    subprocess must receive the exact spec weights in its environment."""
    recorded: dict[str, str] = {}

    def runner(command, *, cwd, env, timeout):
        del cwd, timeout
        recorded.update(env)
        output_dir = Path(command[command.index("--output_path") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        sf.write(output_dir / "output.wav", np.zeros(16_000, dtype=np.float32), 16_000)
        return subprocess.CompletedProcess(command, 0, "Audio saved", "")

    adapter = _adapter(tmp_path, runner)
    spec = _spec(tmp_path)
    adapter.synthesize(spec)

    assert recorded["gpt_path"] == str(spec.gpt_weight.resolve())
    assert recorded["sovits_path"] == str(spec.sovits_weight.resolve())


def test_local_synthesizer_selects_only_published_profile_weights(tmp_path: Path):
    public_dir = tmp_path / "profile" / "public"
    weights_dir = public_dir / "weights"
    weights_dir.mkdir(parents=True)
    gpt_weight = weights_dir / "profile-e1.ckpt"
    sovits_weight = weights_dir / "profile_e1_s1.pth"
    gpt_weight.write_bytes(b"gpt")
    sovits_weight.write_bytes(b"sovits")
    reference_audio = tmp_path / "reference.wav"
    sf.write(reference_audio, np.zeros(16_000, dtype=np.float32), 16_000)
    output_wav = tmp_path / "raw.wav"
    calls: list[SynthesisSpec] = []

    class RecordingAdapter:
        def synthesize(self, spec: SynthesisSpec):
            calls.append(spec)
            sf.write(spec.output_wav, np.zeros(16_000, dtype=np.float32), 16_000)
            return RawSynthesis(
                output_wav=spec.output_wav,
                gpt_weight=spec.gpt_weight,
                sovits_weight=spec.sovits_weight,
                profile_dir=public_dir,
            )

    request = SynthesisCreate(
        voice_profile_id="profile-1",
        text="安全文本",
        text_lang="zh",
        emotion=EmotionControl(mode="manual", label="happy"),
        consent_confirmed=True,
    )
    reference = ReferenceCondition(reference_audio, "参考文本", "zh")
    profile = SimpleNamespace(public_weight_dir=str(public_dir))

    produced = GPTSoVITSLocalSynthesizer(RecordingAdapter()).synthesize(
        request, profile, reference, output_wav
    )

    assert produced == output_wav
    assert calls[0].gpt_weight == gpt_weight.resolve()
    assert calls[0].sovits_weight == sovits_weight.resolve()
