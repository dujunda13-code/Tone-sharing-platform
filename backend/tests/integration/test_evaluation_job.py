from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.routes.jobs import EvaluationService, router
from backend.app.db.session import create_database_engine
from backend.app.schemas.common import JobKind, JobStatus
from backend.app.services.job_queue import JobQueue
from backend.app.services.storage import LocalStorage
from backend.app.services.voice_profiles import VoiceProfileStore
from backend.app.services.workspace import WorkspaceService
from backend.tests.integration.auth_test_helpers import install_authenticated_user
from backend.app.workers.gpu_worker import GPUWorker


def _ready_profile(tmp_path: Path):
    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    profiles = VoiceProfileStore(engine)
    profile = profiles.create("dataset-1", owner_user_id="test-user")
    profiles.mark_queued(profile.id)
    profiles.mark_training(profile.id)
    public_weights = tmp_path / "data" / "profiles" / profile.id / "public"
    public_weights.mkdir(parents=True)
    return engine, profiles, profiles.publish(profile.id, str(public_weights))


class RecordingEvaluationRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def run(self, job_id: str, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append((job_id, payload))
        return {
            "report_path": f"evaluations/{payload['profile_id']}/{job_id}.json",
            "report": {"profile_id": payload["profile_id"], "passed": True},
        }


def test_ready_profile_evaluation_is_queued_and_job_can_be_read(tmp_path: Path):
    engine, profiles, profile = _ready_profile(tmp_path)
    queue = JobQueue(engine=engine)
    app = FastAPI()
    app.state.evaluation_service = EvaluationService(
        profiles=profiles,
        queue=queue,
        texts_path=Path("config/evaluation_texts.zh-en.json"),
    )
    app.state.workspace_service = WorkspaceService(
        engine=engine,
        storage=LocalStorage(tmp_path / "data"),
    )
    install_authenticated_user(app)
    app.include_router(router)

    response = TestClient(app).post(f"/api/voices/{profile.id}/evaluate")

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    job = queue.get(body["job_id"])
    assert job.kind is JobKind.EVALUATE
    assert job.payload["profile_id"] == profile.id

    job_response = TestClient(app).get(f"/api/jobs/{job.id}")
    assert job_response.status_code == 200
    assert job_response.json()["status"] == JobStatus.QUEUED.value


def test_non_ready_profile_cannot_be_evaluated(tmp_path: Path):
    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    profiles = VoiceProfileStore(engine)
    profile = profiles.create("dataset-1", owner_user_id="test-user")
    queue = JobQueue(engine=engine)
    app = FastAPI()
    app.state.evaluation_service = EvaluationService(
        profiles=profiles,
        queue=queue,
        texts_path=Path("config/evaluation_texts.zh-en.json"),
    )
    install_authenticated_user(app)
    app.include_router(router)

    response = TestClient(app).post(f"/api/voices/{profile.id}/evaluate")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "VOICE_PROFILE_NOT_READY"
    assert queue.count_jobs() == 0


def test_gpu_worker_persists_evaluation_report(tmp_path: Path):
    engine, profiles, profile = _ready_profile(tmp_path)
    queue = JobQueue(engine=engine)
    job = queue.enqueue(
        JobKind.EVALUATE,
        {
            "profile_id": profile.id,
            "dataset_id": profile.dataset_id,
            "evaluation_texts_path": "config/evaluation_texts.zh-en.json",
        },
    )
    runner = RecordingEvaluationRunner()
    worker = GPUWorker(
        queue=queue,
        profiles=profiles,
        manifest_loader=lambda dataset_id: None,
        adapters=object(),
        evaluation_runner=runner,
    )

    completed = worker.run_one()

    assert completed is not None
    assert completed.status is JobStatus.SUCCEEDED
    assert completed.result == {
        "report_path": f"evaluations/{profile.id}/{job.id}.json",
        "report": {"passed": True, "profile_id": profile.id},
    }
    assert runner.calls == [(job.id, job.payload)]


def test_gpu_worker_fails_evaluation_when_runner_is_not_configured(tmp_path: Path):
    engine, profiles, profile = _ready_profile(tmp_path)
    queue = JobQueue(engine=engine)
    queue.enqueue(JobKind.EVALUATE, {"profile_id": profile.id})
    worker = GPUWorker(
        queue=queue,
        profiles=profiles,
        manifest_loader=lambda dataset_id: None,
        adapters=object(),
    )

    failed = worker.run_one()

    assert failed is not None
    assert failed.status is JobStatus.FAILED
    assert failed.error_code == "EVALUATION_RUNNER_UNAVAILABLE"
