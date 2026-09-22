from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from typing import Literal

import soundfile as sf
import yaml


class ModelArtifactError(ValueError):
    """Raised when a model artifact is missing or escapes its profile directory."""


class ModelRuntimeError(RuntimeError):
    """Raised when the pinned GPT-SoVITS process cannot run."""


@dataclass(frozen=True)
class ModelManifestReport:
    valid: bool
    verified_count: int
    missing_paths: tuple[Path, ...] = ()
    reason: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_manifest(manifest_path: Path, models_root: Path) -> ModelManifestReport:
    if not manifest_path.is_file():
        return ModelManifestReport(
            False,
            0,
            reason=f"model checksum manifest is missing: {manifest_path}",
        )

    repository_root = models_root.resolve().parent
    models_root = models_root.resolve()
    entries: dict[Path, str] = {}
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            return ModelManifestReport(False, 0, reason=f"invalid checksum line: {line}")
        expected, relative_text = parts
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            return ModelManifestReport(False, 0, reason=f"invalid SHA-256: {expected}")
        relative = Path(relative_text.lstrip("*").strip())
        display_path = relative.as_posix()
        if relative.is_absolute() or ".." in relative.parts:
            return ModelManifestReport(
                False,
                0,
                reason=f"model path escapes models root: {display_path}",
            )
        model_path = (repository_root / relative).resolve()
        try:
            model_path.relative_to(models_root)
        except ValueError:
            return ModelManifestReport(
                False,
                0,
                reason=f"model path escapes models root: {display_path}",
            )
        if relative in entries:
            return ModelManifestReport(
                False,
                0,
                reason=f"duplicate model path: {display_path}",
            )
        entries[relative] = expected.lower()

    if not entries:
        return ModelManifestReport(False, 0, reason="model checksum manifest has no entries")

    missing_paths: list[Path] = []
    for relative, expected in entries.items():
        model_path = (repository_root / relative).resolve()
        display_path = relative.as_posix()
        if not model_path.is_file():
            missing_paths.append(relative)
            continue
        if _sha256(model_path) != expected:
            return ModelManifestReport(
                False,
                0,
                tuple(missing_paths),
                reason=f"SHA-256 mismatch for {display_path}",
            )
    if missing_paths:
        first_missing = missing_paths[0].as_posix()
        return ModelManifestReport(
            False,
            0,
            tuple(missing_paths),
            reason=f"missing allowlisted model file: {first_missing}",
        )
    return ModelManifestReport(True, len(entries))


@dataclass(frozen=True)
class TrainSpec:
    profile_id: str
    dataset_list: Path
    output_dir: Path
    seed: int = 20260902
    phoneme_list: Path | None = None
    profile_dir: Path | None = None


@dataclass(frozen=True)
class SynthesisSpec:
    text: str
    text_lang: Literal["zh", "en"]
    prompt_text: str
    prompt_lang: Literal["zh", "en"]
    reference_audio: Path
    gpt_weight: Path
    sovits_weight: Path
    output_wav: Path


@dataclass(frozen=True)
class TrainedWeights:
    profile_id: str
    gpt_weight: Path
    sovits_weight: Path
    profile_dir: Path


@dataclass(frozen=True)
class RawSynthesis:
    output_wav: Path
    gpt_weight: Path
    sovits_weight: Path
    profile_dir: Path


@dataclass(frozen=True)
class CudaRuntimeInfo:
    torch_version: str
    cuda_version: str | None
    total_vram_bytes: int


@dataclass(frozen=True)
class ModelRuntimeInfo:
    available: bool
    tag: str
    device: str
    executable: Path
    vendor_dir: Path
    reason: str | None = None
    total_vram_bytes: int | None = None
    required_vram_bytes: int = 8000 * 1024**2
    torch_version: str | None = None
    cuda_version: str | None = None
    verified_model_count: int = 0
    missing_model_paths: tuple[Path, ...] = ()


class GPTSoVITSAdapter:
    TAG = "20250606v2pro"
    REQUIRED_TORCH_VERSION = "2.5.1"
    REQUIRED_CUDA_VERSION = "12.4"
    # NVIDIA's 8 GB-class Laptop GPU reports 8188 MiB after its firmware
    # reservation.  Keep a hard floor at 8000 MiB so that the configured
    # 8 GB profile accepts that device but still rejects smaller classes.
    REQUIRED_VRAM_BYTES = 8000 * 1024**2
    CUDA_ALLOCATOR_CONF = "expandable_segments:True"

    def __init__(
        self,
        *,
        vendor_dir: Path | str = Path("vendor/GPT-SoVITS"),
        python_executable: Path | str | None = None,
        timeout_seconds: int = 3600,
        models_root: Path | str | None = None,
        checksum_path: Path | str | None = None,
        command_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.vendor_dir = Path(vendor_dir).resolve()
        self.python_executable = Path(python_executable or sys.executable).resolve()
        self.timeout_seconds = timeout_seconds
        project_root = Path(__file__).resolve().parents[3]
        self.models_root = Path(models_root or project_root / "models").resolve()
        self.checksum_path = Path(
            checksum_path or self.models_root / "checksums.sha256"
        ).resolve()
        self.s2_train_script = self.vendor_dir / "GPT_SoVITS" / "s2_train.py"
        self.s1_train_script = self.vendor_dir / "GPT_SoVITS" / "s1_train.py"
        self.inference_script = self.vendor_dir / "GPT_SoVITS" / "inference_cli.py"
        self.s1_config_template = self.vendor_dir / "GPT_SoVITS" / "configs" / "s1longer-v2.yaml"
        self.s2_config_template = self.vendor_dir / "GPT_SoVITS" / "configs" / "s2v2Pro.json"
        self.command_runner = command_runner

    @staticmethod
    def _normal(path: Path) -> str:
        return path.as_posix()

    def build_train_command(self, spec: TrainSpec) -> list[str]:
        config_path = self.write_s2_training_config(spec)
        return self.build_s2_train_command(config_path)

    def build_s1_train_command(self, config_path: Path) -> list[str]:
        self._require_vendor_script(self.s1_train_script)
        return [
            str(self.python_executable),
            str(self.s1_train_script),
            "--config_file",
            str(Path(config_path).resolve()),
        ]

    def build_s2_train_command(self, config_path: Path) -> list[str]:
        self._require_vendor_script(self.s2_train_script)
        return [
            str(self.python_executable),
            str(self.s2_train_script),
            "--config",
            str(Path(config_path).resolve()),
        ]

    @staticmethod
    def _language_name(language: Literal["zh", "en"]) -> str:
        return "中文" if language == "zh" else "英文"

    def build_inference_command(self, spec: SynthesisSpec) -> list[str]:
        self._require_vendor_script(self.inference_script)
        output_dir = spec.output_wav.resolve().parent
        output_dir.mkdir(parents=True, exist_ok=True)
        ref_text_path = output_dir / "reference.txt"
        target_text_path = output_dir / "target.txt"
        ref_text_path.write_text(spec.prompt_text, encoding="utf-8")
        target_text_path.write_text(spec.text, encoding="utf-8")
        return [
            str(self.python_executable),
            str(self.inference_script),
            "--gpt_model",
            str(spec.gpt_weight.resolve()),
            "--sovits_model",
            str(spec.sovits_weight.resolve()),
            "--ref_audio",
            str(spec.reference_audio.resolve()),
            "--ref_text",
            str(ref_text_path),
            "--ref_language",
            self._language_name(spec.prompt_lang),
            "--target_text",
            str(target_text_path),
            "--target_language",
            self._language_name(spec.text_lang),
            "--output_path",
            str(output_dir),
        ]

    @staticmethod
    def _require_vendor_script(path: Path) -> None:
        if not path.is_file():
            raise ModelRuntimeError(f"Pinned vendor script is missing: {path}")

    def write_s1_training_config(self, spec: TrainSpec) -> Path:
        if not self.s1_config_template.is_file():
            raise ModelRuntimeError(
                f"Pinned S1 configuration is missing: {self.s1_config_template}"
            )
        template = yaml.safe_load(self.s1_config_template.read_text(encoding="utf-8"))
        if not isinstance(template, dict):
            raise ModelRuntimeError("Pinned S1 configuration must be a mapping")
        train = template.get("train")
        data = template.get("data")
        if not isinstance(train, dict) or not isinstance(data, dict):
            raise ModelRuntimeError("Pinned S1 configuration has invalid train/data sections")

        spec.output_dir.mkdir(parents=True, exist_ok=True)
        weights_dir = spec.output_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        train.update(
            {
                "batch_size": 1,
                "precision": "16-mixed",
                "seed": spec.seed,
                "epochs": 20,
                "save_every_n_epoch": 1,
                "if_save_latest": True,
                "if_save_every_weights": True,
                "if_dpo": False,
                "half_weights_save_dir": self._normal(weights_dir),
                "exp_name": spec.profile_id,
            }
        )
        data["num_workers"] = 0
        template.update(
            {
                "pretrained_s1": self._normal(
                    self.models_root / "gpt-sovits" / "v2Pro" / "s1v3.ckpt"
                ),
                "train_semantic_path": self._normal(spec.dataset_list),
                "train_phoneme_path": self._normal(
                    spec.phoneme_list or spec.dataset_list.parent / "2-name2text.txt"
                ),
                "output_dir": self._normal(spec.output_dir / "logs_s1_v2Pro"),
            }
        )
        config_path = spec.output_dir / "s1longer-v2-runtime.yaml"
        config_path.write_text(
            yaml.safe_dump(template, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return config_path

    def write_s2_training_config(self, spec: TrainSpec) -> Path:
        if not self.s2_config_template.is_file():
            raise ModelRuntimeError(
                f"Pinned S2 configuration is missing: {self.s2_config_template}"
            )
        template = json.loads(self.s2_config_template.read_text(encoding="utf-8"))
        train = template.get("train")
        data = template.get("data")
        model = template.get("model")
        if not isinstance(train, dict) or not isinstance(data, dict) or not isinstance(model, dict):
            raise ModelRuntimeError("Pinned S2 configuration has invalid train/data/model sections")

        spec.output_dir.mkdir(parents=True, exist_ok=True)
        weights_dir = spec.output_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (spec.output_dir / "logs_s2_v2Pro").mkdir(parents=True, exist_ok=True)
        (spec.dataset_list.parent / "logs_s2_v2Pro").mkdir(parents=True, exist_ok=True)
        train.update(
            {
                "batch_size": 1,
                "fp16_run": True,
                "grad_ckpt": True,
                "seed": spec.seed,
                "gpu_numbers": "0",
                "pretrained_s2G": self._normal(
                    self.models_root / "gpt-sovits" / "v2Pro" / "s2Gv2Pro.pth"
                ),
                "pretrained_s2D": self._normal(
                    self.models_root / "gpt-sovits" / "v2Pro" / "s2Dv2Pro.pth"
                ),
                "if_save_latest": 1,
                "if_save_every_weights": True,
                "save_every_epoch": 1,
            }
        )
        model["version"] = "v2Pro"
        data["exp_dir"] = self._normal(spec.dataset_list.parent)
        template["s2_ckpt_dir"] = self._normal(spec.output_dir)
        template["save_weight_dir"] = self._normal(weights_dir)
        template["name"] = spec.profile_id
        template["version"] = "v2Pro"
        config_path = spec.output_dir / "s2v2Pro-runtime.json"
        config_path.write_text(
            json.dumps(template, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return config_path

    def _run(
        self,
        command: list[str],
        *,
        log_dir: Path | None = None,
        extra_env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        runtime = self.probe()
        if not runtime.available:
            raise ModelRuntimeError(
                f"GPT-SoVITS runtime is unavailable: {runtime.reason}"
            )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["_CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTORCH_CUDA_ALLOC_CONF"] = self.CUDA_ALLOCATOR_CONF
        env["PATH"] = os.pathsep.join(
            (str(self._cuda_nvrtc_bin_dir()), env.get("PATH", ""))
        )
        env["PYTHONPATH"] = str(self.vendor_dir)
        if extra_env:
            env.update(extra_env)
        try:
            if self.command_runner is not None:
                result = self.command_runner(
                    command,
                    cwd=self.vendor_dir,
                    env=env,
                    timeout=self.timeout_seconds,
                )
            else:
                result = subprocess.run(
                    command,
                    cwd=self.vendor_dir,
                    env=env,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout_seconds,
                    check=False,
                    shell=False,
                )
        except subprocess.TimeoutExpired as exc:
            raise ModelRuntimeError("GPT-SoVITS process timed out") from exc
        except OSError as exc:
            raise ModelRuntimeError("GPT-SoVITS process could not start") from exc
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            stage_name = Path(command[1]).stem
            (log_dir / f"{stage_name}.stdout.log").write_text(
                result.stdout or "", encoding="utf-8", errors="replace"
            )
            (log_dir / f"{stage_name}.stderr.log").write_text(
                result.stderr or "", encoding="utf-8", errors="replace"
            )
        if result.returncode != 0:
            raise ModelRuntimeError(
                f"GPT-SoVITS exited with {result.returncode}: {result.stderr[-1000:]}"
            )
        return result

    @staticmethod
    def _cuda_nvrtc_bin_dir() -> Path:
        try:
            import nvidia.cuda_nvrtc
        except ImportError as exc:
            raise ModelRuntimeError(
                "CUDA 12.4 NVRTC runtime package is unavailable"
            ) from exc
        bin_dir = Path(nvidia.cuda_nvrtc.__path__[0]).resolve() / "bin"
        if not (bin_dir / "nvrtc-builtins64_124.dll").is_file():
            raise ModelRuntimeError(
                "CUDA 12.4 NVRTC runtime is incomplete: nvrtc-builtins64_124.dll"
            )
        return bin_dir

    def _require_training_models(self) -> None:
        required = (
            self.models_root / "gpt-sovits" / "v2Pro" / "s1v3.ckpt",
            self.models_root / "gpt-sovits" / "v2Pro" / "s2Gv2Pro.pth",
            self.models_root / "gpt-sovits" / "v2Pro" / "s2Dv2Pro.pth",
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise ModelRuntimeError(
                "GPT-SoVITS training model file is missing: "
                + ", ".join(str(path) for path in missing)
            )

    @staticmethod
    def _weight_candidates(output_dir: Path, pattern: str) -> list[Path]:
        resolved_output = output_dir.resolve()
        candidates: list[Path] = []
        for path in output_dir.glob(pattern):
            if not path.is_file():
                continue
            try:
                path.resolve().relative_to(resolved_output)
            except ValueError:
                raise ModelRuntimeError(
                    f"GPT-SoVITS produced a weight outside the Job directory: {path}"
                )
            candidates.append(path.resolve())
        return sorted(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))

    @staticmethod
    def _validate_weight_file(path: Path) -> None:
        if not path.is_file() or path.stat().st_size == 0:
            raise ModelRuntimeError(f"GPT-SoVITS produced an empty or missing weight: {path}")
        try:
            import torch

            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ModelRuntimeError(f"GPT-SoVITS weight is not CPU-loadable: {path}") from exc
        if not isinstance(payload, Mapping):
            raise ModelRuntimeError(f"GPT-SoVITS weight has an invalid payload: {path}")

    def _find_weight(self, output_dir: Path, pattern: str, stage: str) -> Path:
        candidates = self._weight_candidates(output_dir / "weights", pattern)
        if not candidates:
            raise ModelRuntimeError(f"GPT-SoVITS {stage} process produced no expected weight")
        weight = candidates[-1]
        self._validate_weight_file(weight)
        return weight

    @staticmethod
    def _inside(path: Path, directory: Path) -> bool:
        try:
            path.resolve().relative_to(directory.resolve())
            return True
        except ValueError:
            return False

    def validate_weights(self, result: RawSynthesis | TrainedWeights) -> None:
        profile_dir = result.profile_dir.resolve()
        for artifact in (result.gpt_weight, result.sovits_weight):
            if not self._inside(artifact, profile_dir):
                raise ModelArtifactError(
                    f"Model artifact is outside profile directory: {artifact}"
                )
            if not artifact.is_file():
                raise ModelArtifactError(f"Model artifact is missing: {artifact}")

    @staticmethod
    def _cuda_runtime() -> CudaRuntimeInfo | None:
        try:
            import torch
        except (ImportError, OSError):
            return None
        try:
            if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
                return None
            return CudaRuntimeInfo(
                torch_version=torch.__version__,
                cuda_version=torch.version.cuda,
                total_vram_bytes=torch.cuda.get_device_properties(0).total_memory,
            )
        except RuntimeError:
            return None

    def _pinned_checkout_tag(self) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(self.vendor_dir), "describe", "--tags", "--exact-match"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=20,
            ).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return None

    def _model_manifest_report(self) -> ModelManifestReport:
        return validate_model_manifest(self.checksum_path, self.models_root)

    def probe(self) -> ModelRuntimeInfo:
        if not self.vendor_dir.is_dir():
            return ModelRuntimeInfo(
                available=False,
                tag=self.TAG,
                device="cuda:0",
                executable=self.python_executable,
                vendor_dir=self.vendor_dir,
                reason="pinned checkout is missing",
            )
        tag = self._pinned_checkout_tag()
        if tag is None:
            return ModelRuntimeInfo(
                False,
                self.TAG,
                "cuda:0",
                self.python_executable,
                self.vendor_dir,
                "unable to determine pinned checkout tag",
            )
        if tag != self.TAG:
            return ModelRuntimeInfo(
                False,
                self.TAG,
                "cuda:0",
                self.python_executable,
                self.vendor_dir,
                f"checkout tag is {tag!r}",
            )
        runtime = self._cuda_runtime()
        manifest = self._model_manifest_report()
        if not manifest.valid:
            return ModelRuntimeInfo(
                False,
                self.TAG,
                "cuda:0",
                self.python_executable,
                self.vendor_dir,
                manifest.reason,
                total_vram_bytes=runtime.total_vram_bytes if runtime else None,
                required_vram_bytes=self.REQUIRED_VRAM_BYTES,
                torch_version=runtime.torch_version if runtime else None,
                cuda_version=runtime.cuda_version if runtime else None,
                verified_model_count=manifest.verified_count,
                missing_model_paths=manifest.missing_paths,
            )
        if runtime is None:
            return ModelRuntimeInfo(
                False,
                self.TAG,
                "cuda:0",
                self.python_executable,
                self.vendor_dir,
                "CUDA runtime is unavailable or no CUDA device is visible",
                verified_model_count=manifest.verified_count,
            )
        if runtime.torch_version.split("+", 1)[0] != self.REQUIRED_TORCH_VERSION:
            return ModelRuntimeInfo(
                False,
                self.TAG,
                "cuda:0",
                self.python_executable,
                self.vendor_dir,
                f"PyTorch version is {runtime.torch_version!r}, expected {self.REQUIRED_TORCH_VERSION}",
                runtime.total_vram_bytes,
                self.REQUIRED_VRAM_BYTES,
                runtime.torch_version,
                runtime.cuda_version,
                manifest.verified_count,
            )
        if runtime.cuda_version != self.REQUIRED_CUDA_VERSION:
            return ModelRuntimeInfo(
                False,
                self.TAG,
                "cuda:0",
                self.python_executable,
                self.vendor_dir,
                f"CUDA version is {runtime.cuda_version!r}, expected {self.REQUIRED_CUDA_VERSION}",
                runtime.total_vram_bytes,
                self.REQUIRED_VRAM_BYTES,
                runtime.torch_version,
                runtime.cuda_version,
                manifest.verified_count,
            )
        if runtime.total_vram_bytes < self.REQUIRED_VRAM_BYTES:
            return ModelRuntimeInfo(
                False,
                self.TAG,
                "cuda:0",
                self.python_executable,
                self.vendor_dir,
                "CUDA device 0 has less than required 8000 MiB VRAM",
                runtime.total_vram_bytes,
                self.REQUIRED_VRAM_BYTES,
                runtime.torch_version,
                runtime.cuda_version,
                manifest.verified_count,
            )
        return ModelRuntimeInfo(
            True,
            self.TAG,
            "cuda:0",
            self.python_executable,
            self.vendor_dir,
            total_vram_bytes=runtime.total_vram_bytes,
            required_vram_bytes=self.REQUIRED_VRAM_BYTES,
            torch_version=runtime.torch_version,
            cuda_version=runtime.cuda_version,
            verified_model_count=manifest.verified_count,
        )

    def train(
        self,
        spec: TrainSpec,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> TrainedWeights:
        runtime = self.probe()
        if not runtime.available:
            raise ModelRuntimeError(f"GPT-SoVITS runtime is unavailable: {runtime.reason}")
        self._require_training_models()
        output_dir = spec.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        if on_progress is not None:
            on_progress(0.0, "starting SoVITS training")
        s2_config = self.write_s2_training_config(spec)
        s1_config = self.write_s1_training_config(spec)
        self._run(self.build_s2_train_command(s2_config), log_dir=output_dir / "logs")
        sovits_weight = self._find_weight(output_dir, "*_e*_s*.pth", "SoVITS")
        if on_progress is not None:
            on_progress(0.5, "SoVITS training completed")
        self._run(self.build_s1_train_command(s1_config), log_dir=output_dir / "logs")
        gpt_weight = self._find_weight(output_dir, "*.ckpt", "GPT")
        if on_progress is not None:
            on_progress(1.0, "GPT training completed")
        profile_dir = (spec.profile_dir or output_dir.parent).resolve()
        result = TrainedWeights(
            profile_id=spec.profile_id,
            gpt_weight=gpt_weight,
            sovits_weight=sovits_weight,
            profile_dir=profile_dir,
        )
        self.validate_weights(result)
        return result

    def synthesize(self, spec: SynthesisSpec) -> RawSynthesis:
        profile_dir = spec.gpt_weight.parent.parent.resolve()
        self.validate_weights(
            RawSynthesis(spec.output_wav, spec.gpt_weight, spec.sovits_weight, profile_dir)
        )
        runtime = self.probe()
        if not runtime.available:
            raise ModelRuntimeError(f"GPT-SoVITS runtime is unavailable: {runtime.reason}")
        output_wav = spec.output_wav.resolve()
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        vendor_output = output_wav.parent / "output.wav"
        vendor_output.unlink(missing_ok=True)
        if vendor_output != output_wav:
            output_wav.unlink(missing_ok=True)
        self._run(
            self.build_inference_command(spec),
            log_dir=output_wav.parent / "logs",
            # The vendored inference_webui loads gpt_path at module import
            # (module-level change_gpt_weights) from the process environment.
            extra_env={
                "gpt_path": str(spec.gpt_weight.resolve()),
                "sovits_path": str(spec.sovits_weight.resolve()),
            },
        )
        if not vendor_output.is_file() or vendor_output.stat().st_size == 0:
            raise ModelRuntimeError("GPT-SoVITS inference produced no local WAV")
        try:
            audio_info = sf.info(vendor_output)
        except RuntimeError as exc:
            raise ModelRuntimeError("GPT-SoVITS inference produced an unreadable WAV") from exc
        if audio_info.frames < 1 or audio_info.samplerate < 1 or audio_info.channels < 1:
            raise ModelRuntimeError("GPT-SoVITS inference produced an invalid WAV")
        if vendor_output != output_wav:
            os.replace(vendor_output, output_wav)
        return RawSynthesis(
            output_wav=output_wav,
            gpt_weight=spec.gpt_weight,
            sovits_weight=spec.sovits_weight,
            profile_dir=profile_dir,
        )
