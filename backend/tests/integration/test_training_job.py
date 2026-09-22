from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.routes.voices import VoiceService, router
from backend.app.api.dependencies import get_current_user
from backend.app.core.config import get_settings
from backend.app.db.session import create_database_engine
from backend.app.schemas.auth import UserRecord
from backend.app.schemas.common import JobKind, JobStatus
from backend.app.schemas.dataset import DatasetManifest, DatasetManifestRow
from backend.app.api.routes.datasets import DatasetService
from backend.app.services.job_queue import JobQueue
from backend.app.services.storage import LocalStorage
from backend.app.services.disentanglement import AdapterWeights
from backend.app.services.gpt_sovits import TrainSpec, TrainedWeights
from backend.app.services.voice_profiles import VoiceProfileStore
from backend.app.workers.gpu_worker import GPUOutOfMemory, GPUWorker


STAGES = (
    "preflight",
    "feature_extract",
    "gpt_sovits_train",
    "disentangler_train",
    "evaluate",
)


class RecordingAdapters:
    def __init__(self, *, similarity_median: float = 0.91, similarity_p10: float = 0.86) -> None:
        self.calls: list[str] = []
        self.similarity_median = similarity_median
        self.similarity_p10 = similarity_p10

    def _run(self, stage: str, work_dir: Path) -> dict[str, str]:
        self.calls.append(stage)
        artifact = work_dir / "weights" / f"{stage}.bin"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(stage, encoding="utf-8")
        return {"artifact": artifact.relative_to(work_dir).as_posix()}

    def preflight(self, manifest: DatasetManifest, work_dir: Path) -> dict[str, str]:
        return self._run("preflight", work_dir)

    def feature_extract(self, manifest: DatasetManifest, work_dir: Path) -> dict[str, str]:
        return self._run("feature_extract", work_dir)

    def gpt_sovits_train(self, manifest: DatasetManifest, work_dir: Path) -> dict[str, str]:
        return self._run("gpt_sovits_train", work_dir)

    def disentangler_train(self, manifest: DatasetManifest, work_dir: Path) -> dict[str, str]:
        return self._run("disentangler_train", work_dir)

    def evaluate(self, manifest: DatasetManifest, work_dir: Path) -> dict[str, float | str]:
        result = self._run("evaluate", work_dir)
        return {
            **result,
            "speaker_similarity_median": self.similarity_median,
            "speaker_similarity_p10": self.similarity_p10,
        }


class RecordingDatasetPreparer:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def prepare(self, manifest: DatasetManifest, work_dir: Path):
        del manifest
        self.calls.append(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        semantic_path = work_dir / "6-name2semantic.tsv"
        text_path = work_dir / "2-name2text.txt"
        feature_manifest_path = work_dir / "feature-manifest.json"
        semantic_path.write_text("segment.wav\t1 2 3\n", encoding="utf-8")
        text_path.write_text("segment.wav\tphone\t[1]\t文本\n", encoding="utf-8")
        feature_manifest_path.write_text("{\"segments\": []}\n", encoding="utf-8")
        return type(
            "Prepared",
            (),
            {
                "artifacts": lambda self: {
                    "work_dir": work_dir,
                    "semantic_list": semantic_path,
                    "text_list": text_path,
                    "feature_manifest": feature_manifest_path,
                    "content_hash": "a" * 64,
                }
            },
        )()


class RecordingGPTSoVITSAdapter:
    def __init__(self) -> None:
        self.calls: list[TrainSpec] = []

    def train(self, spec):
        self.calls.append(spec)
        weights_dir = spec.output_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        gpt_weight = weights_dir / "profile-1-e1.ckpt"
        sovits_weight = weights_dir / "profile-1_e1_s1.pth"
        gpt_weight.write_bytes(b"test-gpt-weight")
        sovits_weight.write_bytes(b"test-sovits-weight")
        return TrainedWeights(
            profile_id=spec.profile_id,
            gpt_weight=gpt_weight,
            sovits_weight=sovits_weight,
            profile_dir=spec.profile_dir,
        )


class FailingGPTSoVITSAdapter:
    def train(self, spec):
        del spec
        raise RuntimeError("real GPT-SoVITS process failed")


class RecordingDisentanglerTrainer:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def fit(self, feature_batches, output_dir: Path):
        del feature_batches
        self.calls.append(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        adapter_path = output_dir / "disentangler-adapter.pt"
        adapter_path.write_bytes(b"test-disentangler-adapter")
        return AdapterWeights(path=adapter_path, losses={"total": 0.1}, steps=1)


def _manifest(tmp_path: Path) -> DatasetManifest:
    manifest_path = tmp_path / "manifest.jsonl"
    row = DatasetManifestRow(
        segment_id="seg-0001",
        path="datasets/d1/segments/seg-0001.wav",
        speaker="voice-d1",
        text="authorized training sample",
        duration_seconds=6.0,
        snr_db=30.0,
        clipping_ratio=0.0,
        split="train",
    )
    manifest_path.write_text(json.dumps(row.model_dump(mode="json")) + "\n", encoding="utf-8")
    return DatasetManifest(
        dataset_id="d1",
        effective_seconds=600.0,
        rows=[row],
        manifest_path=manifest_path,
    )


def _worker(
    tmp_path: Path,
    adapters: RecordingAdapters,
    **worker_options: object,
) -> tuple[GPUWorker, JobQueue, VoiceProfileStore, str]:
    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    queue = JobQueue(engine=engine)
    profiles = VoiceProfileStore(engine)
    profile = profiles.create("d1")
    profiles.mark_queued(profile.id)
    job = queue.enqueue(
        JobKind.TRAIN,
        {"profile_id": profile.id, "dataset_id": "d1", "consent_confirmed": True},
    )
    worker = GPUWorker(
        queue=queue,
        profiles=profiles,
        manifest_loader=lambda dataset_id: _manifest(tmp_path),
        adapters=adapters,
        profiles_root=tmp_path / "data" / "profiles",
        **worker_options,
    )
    return worker, queue, profiles, job.id


def test_training_pipeline_publishes_only_after_all_stages(tmp_path: Path):
    adapters = RecordingAdapters()
    worker, queue, profiles, job_id = _worker(tmp_path, adapters)

    claimed = worker.queue.claim_next(worker.worker_id)
    assert claimed is not None
    worker.run_claimed_training(claimed)

    profile = profiles.get(queue.get(job_id).payload["profile_id"])
    checkpoint_path = tmp_path / "data" / "profiles" / profile.id / "runs" / job_id / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert adapters.calls == list(STAGES)
    assert queue.get(job_id).status is JobStatus.SUCCEEDED
    assert profile.status == "ready"
    assert profile.public_weight_dir is not None
    assert Path(profile.public_weight_dir).is_dir()
    assert tuple(checkpoint["stages"]) == STAGES
    for stage in STAGES:
        assert {
            "input_hash",
            "config_hash",
            "model_version",
            "seed",
            "started_at",
            "finished_at",
            "artifact_hash",
        }.issubset(checkpoint["stages"][stage])


def test_failed_evaluation_keeps_profile_unavailable(tmp_path: Path):
    adapters = RecordingAdapters(similarity_median=0.89)
    worker, queue, profiles, job_id = _worker(tmp_path, adapters)

    claimed = worker.queue.claim_next(worker.worker_id)
    assert claimed is not None
    worker.run_claimed_training(claimed)

    profile = profiles.get(queue.get(job_id).payload["profile_id"])
    assert queue.get(job_id).status is JobStatus.FAILED
    assert queue.get(job_id).error_code == "QUALITY_GATE_FAILED"
    assert profile.status == "failed"
    assert profile.public_weight_dir is None


def test_worker_clears_cache_and_retries_oom_exactly_once(tmp_path: Path):
    class AlwaysOOMAdapters(RecordingAdapters):
        def __init__(self) -> None:
            super().__init__()
            self.preflight_attempts = 0

        def preflight(self, manifest: DatasetManifest, work_dir: Path) -> dict[str, str]:
            self.preflight_attempts += 1
            raise GPUOutOfMemory("simulated CUDA allocation failure")

    adapters = AlwaysOOMAdapters()
    cache_clears: list[str] = []
    worker, queue, profiles, job_id = _worker(
        tmp_path,
        adapters,
        clear_cuda_cache=lambda: cache_clears.append("cleared"),
    )

    claimed = worker.queue.claim_next(worker.worker_id)
    assert claimed is not None
    worker.run_claimed_training(claimed)

    profile = profiles.get(queue.get(job_id).payload["profile_id"])
    assert adapters.preflight_attempts == 2
    assert cache_clears == ["cleared"]
    assert queue.get(job_id).status is JobStatus.FAILED
    assert queue.get(job_id).error_code == "GPU_OUT_OF_MEMORY"
    assert profile.status == "failed"


def test_worker_persists_local_gpt_sovits_dataset_contract(tmp_path: Path):
    adapters = RecordingAdapters()
    preparer = RecordingDatasetPreparer()
    worker, queue, profiles, job_id = _worker(
        tmp_path,
        adapters,
        dataset_preparer=preparer,
        features_root=tmp_path / "data" / "features",
    )

    claimed = worker.queue.claim_next(worker.worker_id)
    assert claimed is not None
    worker.run_claimed_training(claimed)

    profile = profiles.get(queue.get(job_id).payload["profile_id"])
    contract_path = (
        tmp_path
        / "data"
        / "profiles"
        / profile.id
        / "runs"
        / job_id
        / "prepared-training-data.json"
    )
    assert queue.get(job_id).status is JobStatus.SUCCEEDED
    assert preparer.calls == [tmp_path / "data" / "features" / "d1" / job_id]
    assert json.loads(contract_path.read_text(encoding="utf-8"))["gpt_sovits_dataset"]["content_hash"] == "a" * 64


def test_worker_does_not_call_feature_adapter_after_real_dataset_preparation(tmp_path: Path):
    class RejectingFeatureAdapter(RecordingAdapters):
        def feature_extract(self, manifest: DatasetManifest, work_dir: Path) -> dict[str, str]:
            del manifest, work_dir
            raise AssertionError("real dataset preparation must be the only feature_extract implementation")

    adapters = RejectingFeatureAdapter()
    worker, queue, profiles, job_id = _worker(
        tmp_path,
        adapters,
        dataset_preparer=RecordingDatasetPreparer(),
        features_root=tmp_path / "data" / "features",
    )

    claimed = worker.queue.claim_next(worker.worker_id)
    assert claimed is not None
    worker.run_claimed_training(claimed)

    profile = profiles.get(queue.get(job_id).payload["profile_id"])
    checkpoint = json.loads(
        (
            tmp_path / "data" / "profiles" / profile.id / "runs" / job_id / "checkpoint.json"
        ).read_text(encoding="utf-8")
    )
    assert queue.get(job_id).status is JobStatus.SUCCEEDED
    assert "feature_extract" not in adapters.calls
    assert checkpoint["stages"]["feature_extract"]["artifacts"]["gpt_sovits_dataset"][
        "content_hash"
    ] == "a" * 64


def test_worker_connects_verified_dataset_to_real_gpt_sovits_adapter(tmp_path: Path):
    adapters = RecordingAdapters()
    preparer = RecordingDatasetPreparer()
    gpt_adapter = RecordingGPTSoVITSAdapter()
    worker, queue, profiles, job_id = _worker(
        tmp_path,
        adapters,
        dataset_preparer=preparer,
        features_root=tmp_path / "data" / "features",
        gpt_sovits_adapter=gpt_adapter,
    )

    claimed = worker.queue.claim_next(worker.worker_id)
    assert claimed is not None
    worker.run_claimed_training(claimed)

    profile = profiles.get(queue.get(job_id).payload["profile_id"])
    assert queue.get(job_id).status is JobStatus.SUCCEEDED
    assert len(gpt_adapter.calls) == 1
    assert gpt_adapter.calls[0].dataset_list.is_file()
    assert gpt_adapter.calls[0].phoneme_list.is_file()
    assert gpt_adapter.calls[0].profile_dir == (tmp_path / "data" / "profiles" / profile.id).resolve()
    assert "gpt_weight" in json.loads(
        (tmp_path / "data" / "profiles" / profile.id / "runs" / job_id / "checkpoint.json").read_text(
            encoding="utf-8"
        )
    )["stages"]["gpt_sovits_train"]["artifacts"]


def test_worker_does_not_publish_when_real_gpt_sovits_stage_fails(tmp_path: Path):
    adapters = RecordingAdapters()
    worker, queue, profiles, job_id = _worker(
        tmp_path,
        adapters,
        dataset_preparer=RecordingDatasetPreparer(),
        features_root=tmp_path / "data" / "features",
        gpt_sovits_adapter=FailingGPTSoVITSAdapter(),
    )

    claimed = worker.queue.claim_next(worker.worker_id)
    assert claimed is not None
    worker.run_claimed_training(claimed)

    profile = profiles.get(queue.get(job_id).payload["profile_id"])
    assert queue.get(job_id).status is JobStatus.FAILED
    assert profile.status == "failed"
    assert profile.public_weight_dir is None
    assert not (tmp_path / "data" / "profiles" / profile.id / "public").exists()


def test_worker_connects_local_feature_manifest_to_disentangler(tmp_path: Path):
    adapters = RecordingAdapters()
    trainer = RecordingDisentanglerTrainer()
    loader_calls: list[Path] = []

    def loader(manifest_path: Path):
        loader_calls.append(manifest_path)
        return ("verified-feature-batch",)

    worker, queue, profiles, job_id = _worker(
        tmp_path,
        adapters,
        dataset_preparer=RecordingDatasetPreparer(),
        features_root=tmp_path / "data" / "features",
        disentangler_trainer=trainer,
        feature_batch_loader=loader,
    )

    claimed = worker.queue.claim_next(worker.worker_id)
    assert claimed is not None
    worker.run_claimed_training(claimed)

    profile = profiles.get(queue.get(job_id).payload["profile_id"])
    assert queue.get(job_id).status is JobStatus.SUCCEEDED
    assert len(loader_calls) == 1
    assert loader_calls[0].is_file()
    assert trainer.calls[0].is_dir()
    assert profile.public_weight_dir is not None


def test_voice_api_creates_zero_shot_profile_and_train_endpoint_is_fail_closed(tmp_path):
    """Client profile creation never queues work; /train stays a compat 409."""
    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    queue = JobQueue(engine=engine)
    profiles = VoiceProfileStore(engine)
    datasets = DatasetService(storage=LocalStorage(tmp_path / "data"), engine=engine)
    datasets.register_dataset(
        "d1",
        effective_seconds=6.5,
        owner_user_id="test-user",
        authorization_confirmed=True,
    )
    settings = get_settings()
    app = FastAPI()
    app.state.voice_service = VoiceService(
        profiles=profiles, datasets=datasets, settings=settings
    )
    app.dependency_overrides[get_current_user] = lambda: UserRecord(
        id="test-user",
        username="test-user",
        role="user",
        status="active",
        created_at=datetime.now(timezone.utc),
    )
    app.include_router(router)
    client = TestClient(app)

    created = client.post(
            "/api/voices",
            json={"dataset_id": "d1", "display_name": "训练合同音色"},
        )
    profile_id = created.json()["id"]
    rejected = client.post(
        f"/api/voices/{profile_id}/train", json={"consent_confirmed": True}
    )
    profile = client.get(f"/api/voices/{profile_id}")

    assert created.status_code == 201
    assert created.json()["status"] == "ready"
    assert created.json()["mode"] == "zero_shot"
    assert queue.count_jobs() == 0
    assert rejected.status_code == 409
    # This bare app lacks create_app's unified error handlers, so the code
    # may arrive under either envelope; the full envelope with request_id is
    # covered by the create_app-based zero-shot API test.
    body = rejected.json()
    code = body.get("error", {}).get("code") or body.get("detail", {}).get("code")
    assert code == "CLIENT_TRAINING_DISABLED"
    assert queue.count_jobs() == 0
    assert profile.status_code == 200
    assert profile.json()["status"] == "ready"
    stored = profiles.get(profile_id)
    assert stored.mode == "zero_shot"
    assert stored.public_weight_dir is None
