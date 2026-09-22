from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable, Literal

import soundfile as sf


@dataclass(frozen=True)
class WebUISynthesisSpec:
    text: str
    text_lang: Literal["zh", "en"]
    prompt_text: str
    prompt_lang: Literal["zh", "en"]
    reference_audio: Path
    gpt_weight: Path
    sovits_weight: Path
    output_wav: Path
    auxiliary_audios: tuple[Path, ...] = ()
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    how_to_cut: str = "不切"
    sample_steps: int = 8
    speed: float = 1.0
    pause_second: float = 0.3
    seed: int = 20260916
    version: str = "v2ProPlus"


class WebUIInferenceBridge:
    """Project-owned bridge to official vendor inference_webui.

    Adheres strictly to the official WebUI parameters (top_k=15, top_p=1.0, temperature=1.0)
    and passes auxiliary references without modifying vendor/GPT-SoVITS files.
    """

    CUDA_ALLOCATOR_CONF = "expandable_segments:True"

    def __init__(
        self,
        *,
        vendor_dir: Path | str = Path("vendor/GPT-SoVITS"),
        python_executable: Path | str | None = None,
        timeout_seconds: int = 3600,
        command_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.vendor_dir = Path(vendor_dir).resolve()
        self.python_executable = Path(python_executable or sys.executable).resolve()
        self.timeout_seconds = timeout_seconds
        self.command_runner = command_runner

    def validate_spec(self, spec: WebUISynthesisSpec) -> None:
        if not spec.reference_audio.is_file():
            raise FileNotFoundError(f"参考音频不存在: {spec.reference_audio}")
        if not spec.gpt_weight.is_file():
            raise FileNotFoundError(f"GPT 权重文件不存在: {spec.gpt_weight}")
        if not spec.sovits_weight.is_file():
            raise FileNotFoundError(f"SoVITS 权重文件不存在: {spec.sovits_weight}")
        for aux in spec.auxiliary_audios:
            if not aux.is_file():
                raise FileNotFoundError(f"辅助参考音频不存在: {aux}")

    @staticmethod
    def _cuda_nvrtc_bin_dir() -> Path:
        try:
            import nvidia.cuda_nvrtc
        except ImportError as exc:
            raise RuntimeError("CUDA 12.4 NVRTC runtime package is unavailable") from exc
        bin_dir = Path(nvidia.cuda_nvrtc.__path__[0]).resolve() / "bin"
        if not (bin_dir / "nvrtc-builtins64_124.dll").is_file():
            raise RuntimeError("CUDA 12.4 NVRTC runtime is incomplete: nvrtc-builtins64_124.dll")
        return bin_dir

    def build_command_and_env(
        self, spec: WebUISynthesisSpec, config_path: Path
    ) -> tuple[list[str], dict[str, str]]:
        self.validate_spec(spec)
        config_payload = {
            "text": spec.text,
            "text_lang": spec.text_lang,
            "prompt_text": spec.prompt_text,
            "prompt_lang": spec.prompt_lang,
            "reference_audio": str(spec.reference_audio.resolve()),
            "gpt_weight": str(spec.gpt_weight.resolve()),
            "sovits_weight": str(spec.sovits_weight.resolve()),
            "output_wav": str(spec.output_wav.resolve()),
            "auxiliary_audios": [str(p.resolve()) for p in spec.auxiliary_audios],
            "top_k": spec.top_k,
            "top_p": spec.top_p,
            "temperature": spec.temperature,
            "how_to_cut": spec.how_to_cut,
            "sample_steps": spec.sample_steps,
            "speed": spec.speed,
            "pause_second": spec.pause_second,
            "seed": spec.seed,
            "version": spec.version,
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        project_root = Path(__file__).resolve().parents[3]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["_CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTORCH_CUDA_ALLOC_CONF"] = self.CUDA_ALLOCATOR_CONF
        env["PATH"] = os.pathsep.join(
            (str(self._cuda_nvrtc_bin_dir()), env.get("PATH", ""))
        )
        env["version"] = spec.version
        env["gpt_path"] = str(spec.gpt_weight.resolve())
        env["sovits_path"] = str(spec.sovits_weight.resolve())
        # Let Python find both the vendor packages and backend modules
        python_paths = [str(self.vendor_dir), str(self.vendor_dir / "GPT_SoVITS"), str(project_root)]
        if "PYTHONPATH" in env:
            python_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(python_paths)

        cmd = [
            str(self.python_executable),
            "-m",
            "backend.app.services.webui_runner",
            "--config",
            str(config_path.resolve()),
        ]
        return cmd, env

    def synthesize(self, spec: WebUISynthesisSpec) -> Path:
        self.validate_spec(spec)
        output_dir = spec.output_wav.resolve().parent
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "webui_run_config.json"
        cmd, env = self.build_command_and_env(spec, config_path)

        if self.command_runner is not None:
            res = self.command_runner(
                cmd,
                cwd=self.vendor_dir,
                env=env,
                timeout=self.timeout_seconds,
            )
        else:
            res = subprocess.run(
                cmd,
                cwd=self.vendor_dir,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
            )

        if res.returncode != 0:
            raise RuntimeError(
                f"WebUI inference runner failed with return code {res.returncode}: {res.stderr[-1000:]}"
            )

        if not spec.output_wav.is_file():
            raise RuntimeError(f"WebUI inference runner did not produce output file: {spec.output_wav}")

        self._validate_output_wav(spec.output_wav)

        return spec.output_wav

    @staticmethod
    def _validate_output_wav(output_wav: Path) -> None:
        """Fail closed unless the output meets the guarded WAV spec (PCM 16-bit 32kHz mono)."""
        info = sf.info(str(output_wav))
        if info.samplerate != 32_000 or info.channels != 1 or info.subtype != "PCM_16":
            raise RuntimeError(
                "WebUI inference output violates the guarded WAV spec "
                f"(expected PCM 16-bit 32kHz mono, got {info.subtype} {info.samplerate}Hz "
                f"{info.channels}ch): {output_wav}"
            )
