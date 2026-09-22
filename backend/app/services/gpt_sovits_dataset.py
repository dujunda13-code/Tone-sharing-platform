from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from backend.app.schemas.dataset import DatasetManifest, DatasetManifestRow
from backend.app.services.audio_validation import LocalAudioPathError, resolve_local_audio_path
from backend.app.services.training_errors import TrainingPipelineError


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class PreparedTrainingData:
    """Verified local artifacts emitted by the pinned GPT-SoVITS prep scripts."""

    dataset_id: str
    work_dir: Path
    input_list: Path
    text_list: Path
    feature_dir: Path
    wav32k_dir: Path
    speaker_vector_dir: Path
    feature_manifest: Path
    semantic_list: Path
    bert_dir: Path
    split_lists: Mapping[str, Path]
    content_hash: str

    def artifacts(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "work_dir": self.work_dir,
            "input_list": self.input_list,
            "text_list": self.text_list,
            "feature_dir": self.feature_dir,
            "wav32k_dir": self.wav32k_dir,
            "speaker_vector_dir": self.speaker_vector_dir,
            "feature_manifest": self.feature_manifest,
            "semantic_list": self.semantic_list,
            "bert_dir": self.bert_dir,
            "split_lists": dict(self.split_lists),
            "content_hash": self.content_hash,
        }


class GPTSoVITSDatasetPreparer:
    """Prepare one local, authorized manifest for the pinned vendor scripts."""

    _STAGES = (
        ("text", "1-get-text.py"),
        ("hubert", "2-get-hubert-wav32k.py"),
        ("speaker_vector", "2-get-sv.py"),
        ("semantic", "3-get-semantic.py"),
    )
    _SPEAKER_VECTOR_MODEL_NAME = "pretrained_eres2netv2w24s4ep4.ckpt"
    _G2PW_MODEL_FILES = (
        "g2pW.onnx",
        "config.py",
        "POLYPHONIC_CHARS.txt",
        "MONOPHONIC_CHARS.txt",
        "bopomofo_to_pinyin_wo_tune_dict.json",
        "char_bopomofo_dict.json",
    )

    def __init__(
        self,
        *,
        storage_root: Path | str = Path("data"),
        models_root: Path | str = Path("models"),
        vendor_dir: Path | str = Path("vendor/GPT-SoVITS"),
        python_executable: Path | str | None = None,
        ffmpeg_executable: Path | str | None = None,
        timeout_seconds: int = 3600,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.storage_root = Path(storage_root).expanduser().resolve()
        self.features_root = (self.storage_root / "features").resolve()
        self.models_root = Path(models_root).expanduser().resolve()
        self.vendor_dir = Path(vendor_dir).expanduser().resolve()
        self.python_executable = Path(python_executable or sys.executable).resolve()
        self.ffmpeg_executable = (
            Path(ffmpeg_executable).expanduser().resolve()
            if ffmpeg_executable is not None
            else None
        )
        self.timeout_seconds = timeout_seconds
        self.command_runner = command_runner or self._run_command
        self.script_root = self.vendor_dir / "GPT_SoVITS" / "prepare_datasets"
        self.bert_pretrained_dir = self.models_root / "gpt-sovits" / "bert"
        self.cnhubert_base_dir = self.models_root / "gpt-sovits" / "chinese-hubert-base"
        self.pretrained_s2g = self.models_root / "gpt-sovits" / "v2Pro" / "s2Gv2Pro.pth"
        self.speaker_vector_model = (
            self.models_root / "gpt-sovits" / "sv" / self._SPEAKER_VECTOR_MODEL_NAME
        )
        self.gpt_sovits_runtime_root = self.models_root / "gpt-sovits"
        self.g2pw_model_dir = self.gpt_sovits_runtime_root / "GPT_SoVITS" / "text" / "G2PWModel"
        self.s2config_path = self.vendor_dir / "GPT_SoVITS" / "configs" / "s2v2Pro.json"

    def prepare(self, manifest: DatasetManifest, work_dir: Path) -> PreparedTrainingData:
        output_dir = Path(work_dir).expanduser().resolve()
        self._validate_work_dir(output_dir)
        if not manifest.rows:
            raise TrainingPipelineError("authorized manifest contains no audio rows")

        output_dir.mkdir(parents=True, exist_ok=True)
        wav_dir = output_dir / "wav"
        wav_dir.mkdir(parents=True, exist_ok=True)
        rows = self._stage_audio(manifest, wav_dir)
        self._require_vendor_inputs()
        input_list = output_dir / "1-name2text.txt"
        input_list.write_text(
            "".join(
                f"{audio_name}|{self._field(row.speaker, 'speaker')}|{row.language}|{self._text(row.text)}\n"
                for row, audio_name, _ in rows
            ),
            encoding="utf-8",
        )
        split_lists = self._write_split_lists(rows, output_dir / "splits")
        environment = self._base_environment(input_list, wav_dir, output_dir, manifest.dataset_id)
        for stage_name, script_name in self._STAGES:
            self._run_stage(stage_name, script_name, environment, output_dir)

        text_list = self._merge_part_files(output_dir, "2-name2text-", "2-name2text.txt")
        semantic_list = self._merge_part_files(
            output_dir, "6-name2semantic-", "6-name2semantic.tsv"
        )
        feature_dir = output_dir / "4-cnhubert"
        wav32k_dir = output_dir / "5-wav32k"
        speaker_vector_dir = output_dir / "7-sv_cn"
        bert_dir = output_dir / "3-bert"
        self._validate_outputs(
            rows,
            text_list,
            semantic_list,
            feature_dir,
            wav32k_dir,
            speaker_vector_dir,
            bert_dir,
        )
        feature_manifest = self._write_feature_manifest(
            manifest.dataset_id, rows, output_dir / "feature-manifest.json"
        )
        content_hash = self._content_hash(manifest, rows)
        (output_dir / "manifest.sha256").write_text(content_hash + "\n", encoding="ascii")
        return PreparedTrainingData(
            dataset_id=manifest.dataset_id,
            work_dir=output_dir,
            input_list=input_list,
            text_list=text_list,
            feature_dir=feature_dir,
            wav32k_dir=wav32k_dir,
            speaker_vector_dir=speaker_vector_dir,
            feature_manifest=feature_manifest,
            semantic_list=semantic_list,
            bert_dir=bert_dir,
            split_lists=split_lists,
            content_hash=content_hash,
        )

    def _validate_work_dir(self, work_dir: Path) -> None:
        try:
            work_dir.relative_to(self.features_root)
        except ValueError as exc:
            raise TrainingPipelineError(
                "GPT-SoVITS feature output must remain under local data/features storage"
            ) from exc

    def _require_vendor_inputs(self) -> None:
        missing = [
            self.vendor_dir,
            self.bert_pretrained_dir,
            self.cnhubert_base_dir,
            self.pretrained_s2g,
            self.speaker_vector_model,
            self.s2config_path,
        ]
        missing.extend(self.g2pw_model_dir / name for name in self._G2PW_MODEL_FILES)
        missing.extend(self.script_root / name for _, name in self._STAGES)
        directories = {
            self.vendor_dir,
            self.gpt_sovits_runtime_root,
            self.bert_pretrained_dir,
            self.cnhubert_base_dir,
            self.gpt_sovits_runtime_root,
            self.g2pw_model_dir,
        }
        absent = [
            path
            for path in missing
            if not (path.is_dir() if path in directories else path.is_file())
        ]
        if absent:
            if self.speaker_vector_model in absent:
                raise TrainingPipelineError(
                    "GPT-SoVITS V2Pro speaker vector model is missing: "
                    + str(self.speaker_vector_model)
                )
            raise TrainingPipelineError(
                "GPT-SoVITS dataset preparation input is missing: "
                + ", ".join(str(path) for path in absent)
            )
        if not self.bert_pretrained_dir.is_dir() or not self.cnhubert_base_dir.is_dir():
            raise TrainingPipelineError("GPT-SoVITS feature model directories are invalid")

    def _stage_audio(
        self, manifest: DatasetManifest, wav_dir: Path
    ) -> list[tuple[DatasetManifestRow, str, Path]]:
        staged: list[tuple[DatasetManifestRow, str, Path]] = []
        for index, row in enumerate(manifest.rows, start=1):
            try:
                source = resolve_local_audio_path(row.path, self.storage_root)
            except LocalAudioPathError as exc:
                raise TrainingPipelineError(f"audio row is outside local storage: {row.path}") from exc
            safe_name = self._audio_name(index, source)
            destination = wav_dir / safe_name
            shutil.copy2(source, destination)
            staged.append((row, safe_name, source))
        return staged

    @staticmethod
    def _audio_name(index: int, source: Path) -> str:
        safe_source_name = re.sub(r"[^A-Za-z0-9_.-]", "_", source.name)
        return f"{index:06d}-{safe_source_name}"

    @staticmethod
    def _text(text: str) -> str:
        value = text.strip()
        if not value:
            raise TrainingPipelineError("GPT-SoVITS manifest row has no transcript text")
        if any(character in value for character in "|\r\n"):
            raise TrainingPipelineError("GPT-SoVITS transcript contains a reserved separator")
        return value

    @staticmethod
    def _field(value: str, field_name: str) -> str:
        normalized = value.strip()
        if not normalized or any(character in normalized for character in "|\r\n"):
            raise TrainingPipelineError(f"GPT-SoVITS manifest {field_name} is invalid")
        return normalized

    def _write_split_lists(
        self, rows: list[tuple[DatasetManifestRow, str, Path]], split_dir: Path
    ) -> dict[str, Path]:
        split_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        for split in ("train", "validation", "test"):
            path = split_dir / f"{split}.list"
            path.write_text(
                "".join(audio_name + "\n" for row, audio_name, _ in rows if row.split == split),
                encoding="utf-8",
            )
            paths[split] = path
        return paths

    def _base_environment(
        self, input_list: Path, wav_dir: Path, output_dir: Path, dataset_id: str
    ) -> dict[str, str]:
        ffmpeg_bin_dir = self._ensure_local_ffmpeg_command()
        cuda_nvrtc_bin_dir = self._cuda_nvrtc_bin_dir()
        return {
            "inp_text": str(input_list),
            "inp_wav_dir": str(wav_dir),
            "exp_name": dataset_id,
            "i_part": "0",
            "all_parts": "1",
            "_CUDA_VISIBLE_DEVICES": "0",
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "opt_dir": str(output_dir),
            "bert_pretrained_dir": str(self.bert_pretrained_dir),
            "bert_path": str(self.bert_pretrained_dir),
            "cnhubert_base_dir": str(self.cnhubert_base_dir),
            "pretrained_s2G": str(self.pretrained_s2g),
            "sv_path": str(self.speaker_vector_model),
            "s2config_path": str(self.s2config_path),
            "is_half": "True",
            "version": "v2Pro",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PATH": os.pathsep.join(
                (
                    str(ffmpeg_bin_dir),
                    str(cuda_nvrtc_bin_dir),
                    os.environ.get("PATH", ""),
                )
            ),
            "FFMPEG_BINARY": str(ffmpeg_bin_dir / "ffmpeg.exe"),
        }

    def _ensure_local_ffmpeg_command(self) -> Path:
        """Expose the deployment-bundled FFmpeg as the command vendor scripts require.

        The pinned vendor's ``tools.my_utils`` invokes the literal ``ffmpeg``
        command through ffmpeg-python.  ``imageio-ffmpeg`` ships a verified
        local executable with this client environment, but its filename is
        versioned.  A local hard-link (or same-content copy fallback) gives
        the subprocess an ``ffmpeg.exe`` command without any runtime download.
        """

        source = self.ffmpeg_executable or self._bundled_ffmpeg_executable()
        if not source.is_file():
            raise TrainingPipelineError("local bundled FFmpeg executable is unavailable")
        runtime_bin_dir = self.storage_root / "runtime" / "bin"
        runtime_bin_dir.mkdir(parents=True, exist_ok=True)
        source_hash = self._file_sha256(source)
        for name in ("ffmpeg.exe",):
            command = runtime_bin_dir / name
            if command.exists():
                if not command.is_file() or self._file_sha256(command) != source_hash:
                    raise TrainingPipelineError("local FFmpeg command does not match the bundled executable")
                continue
            try:
                os.link(source, command)
            except OSError:
                shutil.copy2(source, command)
        return runtime_bin_dir

    @staticmethod
    def _cuda_nvrtc_bin_dir() -> Path:
        try:
            import nvidia.cuda_nvrtc
        except ImportError as exc:
            raise TrainingPipelineError(
                "CUDA 12.4 NVRTC runtime package is unavailable"
            ) from exc
        bin_dir = Path(nvidia.cuda_nvrtc.__path__[0]).resolve() / "bin"
        required = bin_dir / "nvrtc-builtins64_124.dll"
        if not required.is_file():
            raise TrainingPipelineError(
                "CUDA 12.4 NVRTC runtime is incomplete: nvrtc-builtins64_124.dll"
            )
        return bin_dir

    @staticmethod
    def _bundled_ffmpeg_executable() -> Path:
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise TrainingPipelineError("local FFmpeg runtime package is unavailable") from exc
        binaries_dir = Path(imageio_ffmpeg.__file__).resolve().parent / "binaries"
        candidates = sorted(
            path
            for path in binaries_dir.glob("ffmpeg-*.exe")
            if path.is_file()
        )
        if len(candidates) != 1:
            raise TrainingPipelineError("local bundled FFmpeg executable is unavailable")
        return candidates[0]

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _run_stage(
        self, stage_name: str, script_name: str, base_environment: Mapping[str, str], output_dir: Path
    ) -> None:
        script = self.script_root / script_name
        command = [str(self.python_executable), str(script)]
        environment = os.environ.copy()
        environment.update(base_environment)
        package_root = self.vendor_dir / "GPT_SoVITS"
        pythonpath_entries = [
            str(package_root / "eres2net"),
            str(package_root),
            str(self.vendor_dir),
        ]
        inherited_pythonpath = environment.get("PYTHONPATH")
        if inherited_pythonpath:
            pythonpath_entries.append(inherited_pythonpath)
        environment["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
        try:
            result = self.command_runner(
                command,
                cwd=self.gpt_sovits_runtime_root,
                env=environment,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise TrainingPipelineError(f"GPT-SoVITS {stage_name} feature stage timed out") from exc
        except OSError as exc:
            raise TrainingPipelineError(
                f"GPT-SoVITS {stage_name} feature stage could not start"
            ) from exc
        (output_dir / "logs").mkdir(parents=True, exist_ok=True)
        (output_dir / "logs" / f"{stage_name}.stdout.log").write_text(
            result.stdout or "", encoding="utf-8", errors="replace"
        )
        (output_dir / "logs" / f"{stage_name}.stderr.log").write_text(
            result.stderr or "", encoding="utf-8", errors="replace"
        )
        if result.returncode != 0:
            raise TrainingPipelineError(
                f"GPT-SoVITS {stage_name} feature stage exited with {result.returncode}"
            )

    @staticmethod
    def _run_command(
        command: list[str], *, cwd: Path, env: Mapping[str, str], timeout: int
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=cwd,
            env=dict(env),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            shell=False,
        )

    @staticmethod
    def _merge_part_files(output_dir: Path, prefix: str, destination_name: str) -> Path:
        extension = "tsv" if "semantic" in prefix else "txt"
        parts = sorted(output_dir.glob(f"{prefix}*.{extension}"))
        if not parts:
            raise TrainingPipelineError(f"GPT-SoVITS did not produce {prefix} part files")
        destination = output_dir / destination_name
        content = "\n".join(path.read_text(encoding="utf-8").strip() for path in parts).strip()
        if not content:
            raise TrainingPipelineError(f"GPT-SoVITS produced an empty {destination_name}")
        destination.write_text(content + "\n", encoding="utf-8")
        return destination

    @staticmethod
    def _validate_outputs(
        rows: list[tuple[DatasetManifestRow, str, Path]],
        text_list: Path,
        semantic_list: Path,
        feature_dir: Path,
        wav32k_dir: Path,
        speaker_vector_dir: Path,
        bert_dir: Path,
    ) -> None:
        expected_names = {audio_name for _, audio_name, _ in rows}
        text_names = {
            line.split("\t", 1)[0]
            for line in text_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        semantic_names = {
            line.split("\t", 1)[0]
            for line in semantic_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        missing_text = expected_names - text_names
        missing_semantic = expected_names - semantic_names
        missing_hubert = {f"{name}.pt" for name in expected_names} - {
            path.name for path in feature_dir.glob("*.pt")
        }
        missing_wav = expected_names - {path.name for path in wav32k_dir.glob("*.wav")}
        missing_speaker_vector = {f"{name}.pt" for name in expected_names} - {
            path.name for path in speaker_vector_dir.glob("*.pt")
        }
        expected_bert = {f"{name}.pt" for row, name, _ in rows if row.language == "zh"}
        missing_bert = expected_bert - {path.name for path in bert_dir.glob("*.pt")}
        if (
            missing_text
            or missing_semantic
            or missing_hubert
            or missing_wav
            or missing_speaker_vector
            or missing_bert
        ):
            raise TrainingPipelineError(
                "GPT-SoVITS feature output is incomplete: "
                + json.dumps(
                    {
                        "text": sorted(missing_text),
                        "semantic": sorted(missing_semantic),
                        "cnhubert": sorted(missing_hubert),
                        "wav32k": sorted(missing_wav),
                        "speaker_vector": sorted(missing_speaker_vector),
                        "bert": sorted(missing_bert),
                    },
                    ensure_ascii=False,
                )
            )

    @staticmethod
    def _write_feature_manifest(
        dataset_id: str,
        rows: list[tuple[DatasetManifestRow, str, Path]],
        destination: Path,
    ) -> Path:
        payload = {
            "dataset_id": dataset_id,
            "segments": [
                {
                    "segment_id": row.segment_id,
                    "speaker_id": row.speaker,
                    "wav32k": (Path("5-wav32k") / audio_name).as_posix(),
                }
                for row, audio_name, _ in rows
            ],
        }
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return destination

    @staticmethod
    def _content_hash(
        manifest: DatasetManifest, rows: list[tuple[DatasetManifestRow, str, Path]]
    ) -> str:
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                manifest.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        for _, _, source in rows:
            digest.update(source.name.encode("utf-8"))
            with source.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()
