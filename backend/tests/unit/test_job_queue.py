import pytest

from backend.app.schemas.common import JobKind, JobStatus
from backend.app.services.job_queue import InvalidJobTransition, JobQueue


@pytest.fixture
def queue():
    return JobQueue("sqlite+pysqlite:///:memory:")


def test_queue_is_fifo_and_claim_is_atomic(queue):
    first = queue.enqueue(JobKind.PREPROCESS, {"dataset_id": "d1"})
    queue.enqueue(JobKind.TRAIN, {"profile_id": "v1"})

    claimed = queue.claim_next("gpu-0")

    assert claimed.id == first.id
    assert claimed.status is JobStatus.RUNNING
    assert claimed.worker_id == "gpu-0"
    assert queue.claim_next("gpu-1") is None


def test_restart_recovers_running_job(queue):
    job = queue.enqueue(JobKind.TRAIN, {"profile_id": "v1"})
    queue.claim_next("gpu-0")

    assert queue.recover_interrupted() == 1
    assert queue.get(job.id).status is JobStatus.QUEUED
    assert queue.get(job.id).worker_id is None


def test_queue_enforces_terminal_state(queue):
    job = queue.enqueue(JobKind.EVALUATE, {"profile_id": "v1"})
    queue.claim_next("gpu-0")
    queue.succeed(job.id, {"passed": True})

    with pytest.raises(InvalidJobTransition):
        queue.fail(job.id, "LATE_FAILURE", "cannot fail a succeeded job")

    assert queue.get(job.id).status is JobStatus.SUCCEEDED


def test_update_progress_persists_message_for_running_job(queue):
    job = queue.enqueue(JobKind.SYNTHESIZE, {"voice_profile_id": "v1"})
    queue.claim_next("gpu-0")

    queue.update_progress(job.id, "正在生成候选 1/3...")

    assert queue.get(job.id).progress_message == "正在生成候选 1/3..."


def test_update_progress_rejects_job_that_is_not_running(queue):
    job = queue.enqueue(JobKind.SYNTHESIZE, {"voice_profile_id": "v1"})

    with pytest.raises(InvalidJobTransition):
        queue.update_progress(job.id, "正在生成候选 1/3...")

    assert queue.get(job.id).progress_message is None


def test_restart_clears_progress_of_recovered_job(queue):
    job = queue.enqueue(JobKind.SYNTHESIZE, {"voice_profile_id": "v1"})
    queue.claim_next("gpu-0")
    queue.update_progress(job.id, "正在验证候选 2/3...")

    assert queue.recover_interrupted() == 1

    recovered = queue.get(job.id)
    assert recovered.status is JobStatus.QUEUED
    assert recovered.progress_message is None
