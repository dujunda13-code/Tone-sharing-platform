from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.db.session import create_database_engine
from backend.app.schemas.common import JobKind, JobStatus
from backend.app.services.job_queue import JobQueue
from backend.app.services.voice_profiles import VoiceProfileStore
from backend.app.workers.gpu_worker import GPUWorker, TrainingInterrupted
from backend.tests.integration.test_training_job import RecordingAdapters, STAGES, _manifest


def _seed_worker(engine, queue, profiles, tmp_path: Path, *, adapters_override=None, **worker_options):
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
        adapters=adapters_override or RecordingAdapters(),
        profiles_root=tmp_path / "data" / "profiles",
        **worker_options,
    )
    return worker, job, profile


def _claim(worker: GPUWorker):
    claimed = worker.queue.claim_next(worker.worker_id)
    assert claimed is not None
    return claimed


@pytest.mark.parametrize("interrupted_stage", STAGES)
def test_worker_resumes_after_each_completed_stage_without_republishing(
    tmp_path: Path,
    interrupted_stage: str,
):
    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    queue = JobQueue(engine=engine)
    profiles = VoiceProfileStore(engine)
    profiles_root = tmp_path / "data" / "profiles"

    def interrupt_after(stage: str) -> None:
        if stage == interrupted_stage:
            raise TrainingInterrupted(f"simulated interruption after {stage}")

    interrupted_worker, job, profile = _seed_worker(
        engine,
        queue,
        profiles,
        tmp_path,
        on_stage_complete=interrupt_after,
    )

    with pytest.raises(TrainingInterrupted, match=interrupted_stage):
        interrupted_worker.run_claimed_training(_claim(interrupted_worker))

    assert queue.get(job.id).status is JobStatus.RUNNING
    assert queue.recover_interrupted() == 1

    recovered_worker = GPUWorker(
        queue=queue,
        profiles=profiles,
        manifest_loader=lambda dataset_id: _manifest(tmp_path),
        adapters=RecordingAdapters(),
        profiles_root=profiles_root,
    )
    recovered_worker.run_claimed_training(_claim(recovered_worker))

    restored = profiles.get(profile.id)
    assert queue.get(job.id).status is JobStatus.SUCCEEDED
    assert restored.status == "ready"
    assert restored.public_weight_dir is not None
    assert Path(restored.public_weight_dir).is_dir()


def test_worker_recovers_after_publication_without_retraining(tmp_path: Path):
    class InterruptAfterPublicationQueue(JobQueue):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.interrupt_succeed_once = True

        def succeed(self, job_id: str, result: dict[str, object]):
            if self.interrupt_succeed_once:
                self.interrupt_succeed_once = False
                raise TrainingInterrupted("simulated interruption after publication")
            return super().succeed(job_id, result)

    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    queue = InterruptAfterPublicationQueue(engine=engine)
    profiles = VoiceProfileStore(engine)
    worker, job, profile = _seed_worker(engine, queue, profiles, tmp_path)

    with pytest.raises(TrainingInterrupted, match="after publication"):
        worker.run_claimed_training(_claim(worker))

    assert queue.get(job.id).status is JobStatus.RUNNING
    assert profiles.get(profile.id).status == "ready"
    assert queue.recover_interrupted() == 1

    worker.run_claimed_training(_claim(worker))

    assert queue.get(job.id).status is JobStatus.SUCCEEDED
    assert profiles.get(profile.id).status == "ready"


def test_worker_fails_recovered_training_jobs_with_client_training_disabled(tmp_path: Path):
    """A train job recovered from an interrupted run must fail closed, not resume."""
    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    queue = JobQueue(engine=engine)
    profiles = VoiceProfileStore(engine)
    adapters = RecordingAdapters()
    worker, job, _profile = _seed_worker(
        engine, queue, profiles, tmp_path, adapters_override=adapters
    )
    claimed = worker.queue.claim_next(worker.worker_id)
    assert claimed is not None
    # Simulate a process restart while the train job is running: recovery
    # requeues it, and the next run_one must fail it without resuming stages.
    assert queue.recover_interrupted() == 1
    assert queue.get(job.id).status is JobStatus.QUEUED

    worker.run_one()

    completed = queue.get(job.id)
    assert completed.status is JobStatus.FAILED
    assert completed.error_code == "CLIENT_TRAINING_DISABLED"
    assert adapters.calls == []
