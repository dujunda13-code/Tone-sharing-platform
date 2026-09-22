from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Any, Protocol

from backend.app.schemas.common import JobKind
from backend.app.schemas.dataset import DatasetManifest
from backend.app.schemas.synthesis import SynthesisCreate
from backend.app.services.disentanglement import AdapterWeights, DisentanglerTrainer
from backend.app.services.gpt_sovits import GPTSoVITSAdapter, TrainSpec, TrainedWeights
from backend.app.services.job_queue import JobQueue, JobRecord
from backend.app.services.metrics import EvaluationContractError, EvaluationRunner
from backend.app.services.synthesis_pipeline import (
    WATERMARK_PROBABILITY_MINIMUM,
    SynthesisPipeline,
    SynthesisRunResult,
)
from backend.app.services.training_errors import TrainingPipelineError
from backend.app.services.voice_profiles import VoiceProfileStateError, VoiceProfileStore


TRAINING_STAGES = (
    "preflight",
    "feature_extract",
    "gpt_sovits_train",
    "disentangler_train",
    "evaluate",
)
TRAINING_SEED = 20260902
MODEL_VERSION = "20250606v2pro"
QUALITY_MEDIAN_MINIMUM = 0.90
QUALITY_P10_MINIMUM = 0.85
GPU_OOM_RETRIES = 1
STAGE_RECORD_FIELDS = frozenset(
    {
        "input_hash",
        "config_hash",
        "model_version",
        "seed",
        "started_at",
        "finished_at",
        "artifacts",
        "artifact_hash",
    }
)


class TrainingInterrupted(RuntimeError):
    """Signals a process-style interruption after a durable stage checkpoint."""


class QualityGateFailed(TrainingPipelineError):
    """Raised before publication when the immutable profile quality gate fails."""


class GPUOutOfMemory(TrainingPipelineError):
    """Raised after the one permitted local CUDA-cache retry is exhausted."""


class TrainingAdapters(Protocol):
    def preflight(self, manifest: DatasetManifest, work_dir: Path) -> Mapping[str, object]: ...

    def feature_extract(self, manifest: DatasetManifest, work_dir: Path) -> Mapping[str, object]: ...

    def gpt_sovits_train(self, manifest: DatasetManifest, work_dir: Path) -> Mapping[str, object]: ...

    def disentangler_train(self, manifest: DatasetManifest, work_dir: Path) -> Mapping[str, object]: ...

    def evaluate(self, manifest: DatasetManifest, work_dir: Path) -> Mapping[str, object]: ...


class TrainingDatasetPreparer(Protocol):
    def prepare(self, manifest: DatasetManifest, work_dir: Path) -> object: ...


class FeatureBatchLoader(Protocol):
    def __call__(self, manifest_path: Path) -> object: ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _hash_json(value: object) -> str:
    encoded = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


class GPUWorker:
    """Claim one persisted training job and advance its local, restartable stage machine."""

    def __init__(
        self,
        *,
        queue: JobQueue,
        profiles: VoiceProfileStore,
        manifest_loader: Callable[[str], DatasetManifest],
        adapters: TrainingAdapters,
        profiles_root: Path | str = Path("data/profiles"),
        worker_id: str = "gpu-worker-0",
        on_stage_complete: Callable[[str], None] | None = None,
        clear_cuda_cache: Callable[[], None] | None = None,
        synthesis_pipeline: SynthesisPipeline | None = None,
        evaluation_runner: EvaluationRunner | None = None,
        dataset_preparer: TrainingDatasetPreparer | None = None,
        features_root: Path | str = Path("data/features"),
        gpt_sovits_adapter: GPTSoVITSAdapter | None = None,
        disentangler_trainer: DisentanglerTrainer | None = None,
        feature_batch_loader: FeatureBatchLoader | None = None,
    ) -> None:
        self.queue = queue
        self.profiles = profiles
        self.manifest_loader = manifest_loader
        self.adapters = adapters
        self.profiles_root = Path(profiles_root).resolve()
        self.worker_id = worker_id
        self.on_stage_complete = on_stage_complete
        self.clear_cuda_cache = clear_cuda_cache or self._clear_cuda_cache
        self.synthesis_pipeline = synthesis_pipeline
        self.evaluation_runner = evaluation_runner
        self.dataset_preparer = dataset_preparer
        self.features_root = Path(features_root).resolve()
        self.gpt_sovits_adapter = gpt_sovits_adapter
        self.disentangler_trainer = disentangler_trainer
        self.feature_batch_loader = feature_batch_loader

    def recover_interrupted(self) -> int:
        return self.queue.recover_interrupted()

    def run_one(self) -> JobRecord | None:
        job = self.queue.claim_next(self.worker_id)
        if job is None:
            return None
        if job.kind is JobKind.SYNTHESIZE:
            return self._run_synthesis(job)
        if job.kind is JobKind.EVALUATE:
            return self._run_evaluation(job)
        if job.kind is JobKind.TRAIN:
            # The client is zero-shot only: every new or recovered training job
            # fails closed before any feature extraction, S1/S2 or disentangler
            # code can run. The training stage machine below stays importable
            # for the separate future server-side training project.
            return self.queue.fail(
                job.id,
                "CLIENT_TRAINING_DISABLED",
                "客户端使用基础模型零样本合成，不创建本地训练任务",
            )
        return self.queue.fail(job.id, "UNSUPPORTED_JOB", f"GPU worker cannot run {job.kind.value}")

    def run_claimed_training(self, job: JobRecord) -> JobRecord:
        """Advance an already-claimed training job (offline server tooling only).

        The zero-shot client worker never routes training jobs here; this
        entrypoint exists so the stage machine remains testable and reusable
        when training is extracted into the server-side project.
        """
        try:
            result = self._run_training(job)
        except TrainingInterrupted:
            # Preserve RUNNING so a restarted process can requeue and resume from checkpoint.json.
            raise
        except GPUOutOfMemory as exc:
            self._fail_profile(job, "GPU_OUT_OF_MEMORY", str(exc))
            return self.queue.get(job.id)
        except QualityGateFailed as exc:
            self._fail_profile(job, "QUALITY_GATE_FAILED", str(exc))
            return self.queue.get(job.id)
        except Exception as exc:
            self._fail_profile(job, "TRAINING_PIPELINE_FAILED", str(exc))
            return self.queue.get(job.id)
        return self.queue.succeed(job.id, result)

    def _run_synthesis(self, job: JobRecord) -> JobRecord:
        if self.synthesis_pipeline is None:
            return self.queue.fail(
                job.id,
                "SYNTHESIS_PIPELINE_UNAVAILABLE",
                "local synthesis safety pipeline is not configured",
            )
        try:
            request = SynthesisCreate.model_validate(job.payload)
        except Exception as exc:
            return self.queue.fail(job.id, "SYNTHESIS_REQUEST_INVALID", str(exc))
        result = self.synthesis_pipeline.run(
            job.id,
            request,
            progress=lambda message: self.queue.update_progress(job.id, message),
        )
        if result.status != "succeeded":
            return self.queue.fail(
                job.id,
                result.error_code or "SYNTHESIS_PIPELINE_FAILED",
                result.error_message or "local synthesis pipeline failed",
            )
        try:
            return self.queue.succeed(job.id, self._verified_synthesis_result(result))
        except Exception as exc:
            return self.queue.fail(job.id, "SYNTHESIS_RESULT_INVALID", str(exc))

    def _run_evaluation(self, job: JobRecord) -> JobRecord:
        if self.evaluation_runner is None:
            return self.queue.fail(
                job.id,
                "EVALUATION_RUNNER_UNAVAILABLE",
                "local evaluation runner is not configured",
            )
        try:
            result = self.evaluation_runner.run(job.id, job.payload)
            if not isinstance(result, Mapping):
                raise EvaluationContractError("evaluation runner returned an invalid result")
            return self.queue.succeed(job.id, dict(_jsonable(result)))
        except EvaluationContractError as exc:
            return self.queue.fail(job.id, "EVALUATION_CONTRACT_FAILED", str(exc))
        except Exception:
            return self.queue.fail(job.id, "EVALUATION_FAILED", "local evaluation failed")

    def _verified_synthesis_result(self, result: SynthesisRunResult) -> dict[str, object]:
        if result.public_audio_path is None or result.fingerprint is None:
            raise TrainingPipelineError("successful synthesis is missing guarded local artifacts")
        if result.watermark_probability is None or result.watermark_probability < WATERMARK_PROBABILITY_MINIMUM:
            raise TrainingPipelineError("successful synthesis did not pass watermark verification")
        outputs_root = self.synthesis_pipeline.storage.resolve("outputs", ".")
        output_path = result.public_audio_path.resolve()
        try:
            relative_output_path = output_path.relative_to(outputs_root)
        except ValueError as exc:
            raise TrainingPipelineError("successful synthesis output escapes local outputs storage") from exc
        if not output_path.is_file():
            raise TrainingPipelineError("successful synthesis output does not exist")
        return {
            "public_audio_path": relative_output_path.as_posix(),
            "watermark_probability": result.watermark_probability,
            "reference_language": result.reference_language,
            "speaker_similarity": result.speaker_similarity,
            "quality_warning_codes": list(result.quality_warning_codes),
            "candidate_count": result.candidate_count,
            "selected_candidate_index": result.selected_candidate_index,
            "fingerprint": {
                "sample_rate": result.fingerprint.sample_rate,
                "high_band_hz": list(result.fingerprint.high_band_hz),
                "high_band_energy_ratio": result.fingerprint.high_band_energy_ratio,
                "spectral_flatness": result.fingerprint.spectral_flatness,
                "zscore_vs_profile": result.fingerprint.zscore_vs_profile,
                "anomaly": result.fingerprint.anomaly,
            },
        }

    def _fail_profile(self, job: JobRecord, code: str, message: str) -> None:
        profile_id = str(job.payload.get("profile_id", ""))
        try:
            if profile_id:
                self.profiles.fail(profile_id)
        except (KeyError, VoiceProfileStateError):
            pass
        self.queue.fail(job.id, code, message)

    def _run_training(self, job: JobRecord) -> dict[str, object]:
        profile_id = str(job.payload.get("profile_id", ""))
        dataset_id = str(job.payload.get("dataset_id", ""))
        if not profile_id or not dataset_id:
            raise TrainingPipelineError("training job is missing profile_id or dataset_id")
        if job.payload.get("consent_confirmed") is not True:
            raise TrainingPipelineError("training job requires literal consent confirmation")

        profile = self.profiles.get(profile_id)
        if profile.dataset_id != dataset_id:
            raise TrainingPipelineError("training job dataset does not match the voice profile")
        manifest = self.manifest_loader(dataset_id)
        if manifest.dataset_id != dataset_id:
            raise TrainingPipelineError("loaded manifest does not match the training job dataset")

        profile_dir = self.profiles_root / profile_id
        work_dir = profile_dir / "runs" / job.id
        checkpoint_path = work_dir / "checkpoint.json"
        if profile.status == "ready":
            return self._resume_published_profile(profile, profile_dir)
        self.profiles.mark_training(profile_id)
        work_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = self._load_checkpoint(checkpoint_path, job, profile_id, dataset_id)
        input_hash = _hash_json({"payload": job.payload, "manifest": manifest.model_dump(mode="json")})
        config_hash = _hash_json(
            {
                "seed": TRAINING_SEED,
                "batch_size": 1,
                "gradient_accumulation": 8,
                "fp16": True,
                "device": "cuda:0",
            }
        )
        self._validate_checkpoint_records(checkpoint, input_hash, config_hash)

        for stage in TRAINING_STAGES:
            if stage in checkpoint["stages"]:
                continue
            started_at = _utc_now()
            artifacts = self._run_stage(stage, manifest, work_dir, profile_dir, profile_id)
            serialized_artifacts = _jsonable(artifacts)
            checkpoint["stages"][stage] = {
                "input_hash": input_hash,
                "config_hash": config_hash,
                "model_version": MODEL_VERSION,
                "seed": TRAINING_SEED,
                "started_at": started_at,
                "finished_at": _utc_now(),
                "artifacts": serialized_artifacts,
                "artifact_hash": _hash_json(serialized_artifacts),
            }
            self._write_checkpoint(checkpoint_path, checkpoint)
            if self.on_stage_complete is not None:
                self.on_stage_complete(stage)

        evaluation = checkpoint["stages"]["evaluate"]["artifacts"]
        self._require_quality(evaluation)
        public_dir = self._publish(profile_dir, work_dir, checkpoint_path, checkpoint, job.id)
        self.profiles.publish(profile_id, str(public_dir))
        return {"profile_id": profile_id, "status": "ready"}

    def _load_checkpoint(
        self,
        checkpoint_path: Path,
        job: JobRecord,
        profile_id: str,
        dataset_id: str,
    ) -> dict[str, Any]:
        if not checkpoint_path.is_file():
            return {
                "version": 1,
                "job_id": job.id,
                "profile_id": profile_id,
                "dataset_id": dataset_id,
                "stages": {},
            }
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            checkpoint.get("job_id") != job.id
            or checkpoint.get("profile_id") != profile_id
            or checkpoint.get("dataset_id") != dataset_id
        ):
            raise TrainingPipelineError("checkpoint does not belong to the claimed training job")
        stages = checkpoint.get("stages")
        if not isinstance(stages, dict):
            raise TrainingPipelineError("checkpoint stages are invalid")
        for stage, record in stages.items():
            if stage not in TRAINING_STAGES or not isinstance(record, dict):
                raise TrainingPipelineError("checkpoint contains an invalid completed stage")
            if not STAGE_RECORD_FIELDS.issubset(record):
                raise TrainingPipelineError("checkpoint contains an incomplete completed stage")
            if record.get("artifact_hash") != _hash_json(record.get("artifacts")):
                raise TrainingPipelineError("checkpoint artifact hash does not match its recorded artifact")
        return checkpoint

    @staticmethod
    def _validate_checkpoint_records(
        checkpoint: Mapping[str, object],
        input_hash: str,
        config_hash: str,
    ) -> None:
        stages = checkpoint["stages"]
        if not isinstance(stages, Mapping):
            raise TrainingPipelineError("checkpoint stages are invalid")
        for record in stages.values():
            if not isinstance(record, Mapping):
                raise TrainingPipelineError("checkpoint contains an invalid completed stage")
            if (
                record.get("input_hash") != input_hash
                or record.get("config_hash") != config_hash
                or record.get("model_version") != MODEL_VERSION
                or record.get("seed") != TRAINING_SEED
            ):
                raise TrainingPipelineError("checkpoint was created with different training inputs")

    def _run_stage(
        self,
        stage: str,
        manifest: DatasetManifest,
        work_dir: Path,
        profile_dir: Path,
        profile_id: str,
    ) -> dict[str, object]:
        for attempt in range(GPU_OOM_RETRIES + 1):
            try:
                prepared_artifacts: dict[str, object] = {}
                if stage == "feature_extract" and self.dataset_preparer is not None:
                    prepared = self.dataset_preparer.prepare(
                        manifest,
                        self.features_root / manifest.dataset_id / work_dir.name,
                    )
                    artifact_builder = getattr(prepared, "artifacts", None)
                    if not callable(artifact_builder):
                        raise TrainingPipelineError(
                            "GPT-SoVITS dataset preparer returned no artifact contract"
                        )
                    prepared_artifacts = {"gpt_sovits_dataset": _jsonable(artifact_builder())}
                    self._write_prepared_artifacts(work_dir, prepared_artifacts)
                    return prepared_artifacts
                if stage == "gpt_sovits_train" and self.gpt_sovits_adapter is not None:
                    return {
                        **prepared_artifacts,
                        **self._run_gpt_sovits_training(
                            profile_id=profile_id,
                            profile_dir=profile_dir,
                            work_dir=work_dir,
                        ),
                    }
                if stage == "disentangler_train" and self.disentangler_trainer is not None:
                    return {
                        **prepared_artifacts,
                        **self._run_disentangler_training(work_dir),
                    }
                return {**prepared_artifacts, **dict(getattr(self.adapters, stage)(manifest, work_dir))}
            except Exception as exc:
                if not self._is_cuda_oom(exc):
                    raise
                if attempt >= GPU_OOM_RETRIES:
                    raise GPUOutOfMemory(
                        f"{stage} exceeded the local CUDA memory budget after one retry"
                    ) from exc
                self.clear_cuda_cache()
        raise AssertionError("unreachable OOM retry state")

    @staticmethod
    def _write_prepared_artifacts(work_dir: Path, artifacts: Mapping[str, object]) -> None:
        work_dir.mkdir(parents=True, exist_ok=True)
        path = work_dir / "prepared-training-data.json"
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            temporary.write_text(
                json.dumps(artifacts, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _run_gpt_sovits_training(
        self,
        *,
        profile_id: str,
        profile_dir: Path,
        work_dir: Path,
    ) -> dict[str, object]:
        contract_path = work_dir / "prepared-training-data.json"
        if not contract_path.is_file():
            raise TrainingPipelineError(
                "GPT-SoVITS training requires the verified feature preparation contract"
            )
        try:
            payload = json.loads(contract_path.read_text(encoding="utf-8"))
            prepared = payload["gpt_sovits_dataset"]
            semantic_path = Path(prepared["semantic_list"]).resolve()
            text_path = Path(prepared["text_list"]).resolve()
            prepared_work_dir = Path(prepared["work_dir"]).resolve()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TrainingPipelineError("GPT-SoVITS feature preparation contract is invalid") from exc
        for path, label in (
            (semantic_path, "semantic list"),
            (text_path, "phoneme list"),
            (prepared_work_dir, "feature directory"),
        ):
            try:
                path.relative_to(self.features_root)
            except ValueError as exc:
                raise TrainingPipelineError(
                    f"GPT-SoVITS {label} escapes local feature storage"
                ) from exc
        if not semantic_path.is_file() or not text_path.is_file() or not prepared_work_dir.is_dir():
            raise TrainingPipelineError("GPT-SoVITS feature preparation artifacts are missing")
        result = self.gpt_sovits_adapter.train(
            TrainSpec(
                profile_id=profile_id,
                dataset_list=semantic_path,
                phoneme_list=text_path,
                output_dir=work_dir,
                profile_dir=profile_dir,
            )
        )
        if not isinstance(result, TrainedWeights):
            raise TrainingPipelineError("GPT-SoVITS adapter returned an invalid weight contract")
        if result.profile_dir.resolve() != profile_dir.resolve():
            raise TrainingPipelineError("GPT-SoVITS weights belong to a different profile")
        return {
            "gpt_weight": self._relative_job_artifact(result.gpt_weight, work_dir),
            "sovits_weight": self._relative_job_artifact(result.sovits_weight, work_dir),
        }

    @staticmethod
    def _relative_job_artifact(path: Path, work_dir: Path) -> str:
        resolved_path = path.resolve()
        try:
            return resolved_path.relative_to(work_dir.resolve()).as_posix()
        except ValueError as exc:
            raise TrainingPipelineError(
                f"GPT-SoVITS weight escapes the local Job directory: {path}"
            ) from exc

    def _run_disentangler_training(self, work_dir: Path) -> dict[str, object]:
        if self.feature_batch_loader is None:
            raise TrainingPipelineError(
                "disentangler training requires a local feature batch loader"
            )
        contract_path = work_dir / "prepared-training-data.json"
        if not contract_path.is_file():
            raise TrainingPipelineError(
                "disentangler training requires the verified feature preparation contract"
            )
        try:
            payload = json.loads(contract_path.read_text(encoding="utf-8"))
            feature_manifest_path = Path(
                payload["gpt_sovits_dataset"]["feature_manifest"]
            ).resolve()
            feature_manifest_path.relative_to(self.features_root)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TrainingPipelineError(
                "disentangler feature manifest is missing or invalid"
            ) from exc
        if not feature_manifest_path.is_file():
            raise TrainingPipelineError("disentangler feature manifest does not exist")
        feature_batches = self.feature_batch_loader(feature_manifest_path)
        result = self.disentangler_trainer.fit(feature_batches, work_dir / "weights")
        if not isinstance(result, AdapterWeights):
            raise TrainingPipelineError("disentangler trainer returned an invalid artifact contract")
        adapter_path = result.path.resolve()
        try:
            adapter_path.relative_to(work_dir.resolve())
        except ValueError as exc:
            raise TrainingPipelineError(
                f"disentangler adapter escapes the local Job directory: {result.path}"
            ) from exc
        if not adapter_path.is_file():
            raise TrainingPipelineError("disentangler trainer returned a missing adapter")
        return {
            "adapter_weight": adapter_path.relative_to(work_dir.resolve()).as_posix(),
            "losses": dict(result.losses),
            "steps": result.steps,
        }

    @staticmethod
    def _is_cuda_oom(exc: Exception) -> bool:
        if isinstance(exc, GPUOutOfMemory):
            return True
        message = str(exc).casefold()
        if "cuda out of memory" in message or "cuda error: out of memory" in message:
            return True
        try:
            import torch
        except ImportError:
            return False
        return isinstance(exc, torch.cuda.OutOfMemoryError)

    @staticmethod
    def _clear_cuda_cache() -> None:
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            with torch.cuda.device(0):
                torch.cuda.empty_cache()

    @staticmethod
    def _resume_published_profile(profile: object, profile_dir: Path) -> dict[str, object]:
        public_weight_dir = getattr(profile, "public_weight_dir", None)
        expected_public_dir = (profile_dir / "public").resolve()
        if public_weight_dir is None or Path(public_weight_dir).resolve() != expected_public_dir:
            raise TrainingPipelineError("ready profile has an invalid local publication directory")
        if not expected_public_dir.is_dir():
            raise TrainingPipelineError("ready profile has no published local weights")
        return {"profile_id": getattr(profile, "id"), "status": "ready"}

    @staticmethod
    def _write_checkpoint(checkpoint_path: Path, checkpoint: Mapping[str, object]) -> None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
        try:
            temporary.write_text(
                json.dumps(checkpoint, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, checkpoint_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _require_quality(evaluation: object) -> None:
        if not isinstance(evaluation, Mapping):
            raise QualityGateFailed("evaluation did not produce profile metrics")
        try:
            median = float(evaluation["speaker_similarity_median"])
            p10 = float(evaluation["speaker_similarity_p10"])
        except (KeyError, TypeError, ValueError) as exc:
            raise QualityGateFailed("evaluation did not produce required speaker metrics") from exc
        if median < QUALITY_MEDIAN_MINIMUM or p10 < QUALITY_P10_MINIMUM:
            raise QualityGateFailed(
                f"speaker quality gate failed: median={median:.3f}, p10={p10:.3f}"
            )

    def _publish(
        self,
        profile_dir: Path,
        work_dir: Path,
        checkpoint_path: Path,
        checkpoint: dict[str, Any],
        job_id: str,
    ) -> Path:
        public_dir = profile_dir / "public"
        if public_dir.is_dir():
            return public_dir
        weights_dir = work_dir / "weights"
        if not weights_dir.is_dir():
            raise TrainingPipelineError("completed training stages produced no local weights directory")
        staging_dir = profile_dir / f".public-{job_id}"
        if staging_dir.exists():
            raise TrainingPipelineError("stale local publication staging directory exists")
        try:
            shutil.copytree(weights_dir, staging_dir / "weights")
            shutil.copy2(checkpoint_path, staging_dir / "checkpoint.json")
            os.replace(staging_dir, public_dir)
        finally:
            if staging_dir.exists():
                self._remove_local_staging_dir(staging_dir, profile_dir)
        checkpoint["published_at"] = _utc_now()
        checkpoint["public_weight_dir"] = public_dir.as_posix()
        self._write_checkpoint(checkpoint_path, checkpoint)
        return public_dir

    @staticmethod
    def _remove_local_staging_dir(staging_dir: Path, profile_dir: Path) -> None:
        resolved_profile_dir = profile_dir.resolve()
        resolved_staging_dir = staging_dir.resolve()
        try:
            resolved_staging_dir.relative_to(resolved_profile_dir)
        except ValueError as exc:
            raise TrainingPipelineError("publication staging path escapes the local profile directory") from exc
        shutil.rmtree(resolved_staging_dir)
