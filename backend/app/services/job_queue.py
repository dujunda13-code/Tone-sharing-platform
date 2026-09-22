import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from backend.app.db.models import Job
from backend.app.db.session import create_database_engine, init_db, session_factory
from backend.app.schemas.common import JobKind, JobStatus


class InvalidJobTransition(ValueError):
    """Raised when a job state transition violates the queue state machine."""


@dataclass(frozen=True)
class JobRecord:
    id: str
    owner_user_id: str | None
    kind: JobKind
    status: JobStatus
    payload: dict[str, Any]
    result: dict[str, Any] | None
    error_code: str | None
    error_message: str | None
    worker_id: str | None
    progress_message: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class JobQueue:
    def __init__(self, database_url: str = "sqlite:///data/app.db", engine: Engine | None = None) -> None:
        self.engine = engine or create_database_engine(database_url)
        init_db(self.engine)
        self._session_factory = session_factory(self.engine)

    @staticmethod
    def _record(job: Job) -> JobRecord:
        return JobRecord(
            id=job.id,
            owner_user_id=job.owner_user_id,
            kind=JobKind(job.kind),
            status=JobStatus(job.status),
            payload=json.loads(job.payload_json),
            result=json.loads(job.result_json) if job.result_json else None,
            error_code=job.error_code,
            error_message=job.error_message,
            worker_id=job.worker_id,
            progress_message=job.progress_message,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )

    def enqueue(
        self,
        kind: JobKind,
        payload: dict[str, Any],
        *,
        owner_user_id: str | None = None,
    ) -> JobRecord:
        with self._session_factory() as session:
            last_order = session.scalar(select(func.max(Job.queue_order))) or 0
            job = Job(
                id=uuid4().hex,
                owner_user_id=owner_user_id,
                kind=JobKind(kind).value,
                status=JobStatus.QUEUED.value,
                queue_order=int(last_order) + 1,
                payload_json=json.dumps(payload, ensure_ascii=False, sort_keys=True),
                created_at=datetime.now(timezone.utc),
            )
            session.add(job)
            session.commit()
            return self._record(job)

    def get(self, job_id: str, owner_user_id: str | None = None) -> JobRecord:
        with self._session_factory() as session:
            job = session.get(Job, job_id)
            if job is None or (owner_user_id is not None and job.owner_user_id != owner_user_id):
                raise KeyError(job_id)
            return self._record(job)

    def count_jobs(self) -> int:
        with self._session_factory() as session:
            return int(session.scalar(select(func.count()).select_from(Job)) or 0)

    def claim_next(self, worker_id: str) -> JobRecord | None:
        with self.engine.connect() as connection:
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            session = Session(bind=connection, expire_on_commit=False)
            try:
                has_running = session.scalar(
                    select(Job.id).where(Job.status == JobStatus.RUNNING.value).limit(1)
                )
                if has_running is not None:
                    connection.rollback()
                    return None
                job = session.scalar(
                    select(Job)
                    .where(Job.status == JobStatus.QUEUED.value)
                    .order_by(Job.queue_order, Job.created_at, Job.id)
                    .limit(1)
                )
                if job is None:
                    connection.rollback()
                    return None
                job.status = JobStatus.RUNNING.value
                job.worker_id = worker_id
                job.started_at = datetime.now(timezone.utc)
                session.flush()
                record = self._record(job)
                connection.commit()
                return record
            except Exception:
                connection.rollback()
                raise
            finally:
                session.close()

    def succeed(self, job_id: str, result: dict[str, Any]) -> JobRecord:
        return self._finish(job_id, JobStatus.SUCCEEDED, result=result)

    def fail(self, job_id: str, code: str, message: str) -> JobRecord:
        return self._finish(job_id, JobStatus.FAILED, error_code=code, error_message=message)

    def update_progress(self, job_id: str, message: str) -> None:
        """Persist a live progress message for a running job (worker progress surface)."""
        with self._session_factory() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status != JobStatus.RUNNING.value:
                raise InvalidJobTransition(f"{job.status} -> progress update")
            job.progress_message = message
            session.commit()

    def _finish(
        self,
        job_id: str,
        status: JobStatus,
        *,
        result: dict[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> JobRecord:
        with self._session_factory() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status != JobStatus.RUNNING.value:
                raise InvalidJobTransition(f"{job.status} -> {status.value}")
            job.status = status.value
            job.result_json = json.dumps(result, ensure_ascii=False, sort_keys=True) if result else None
            job.error_code = error_code
            job.error_message = error_message
            job.finished_at = datetime.now(timezone.utc)
            session.commit()
            return self._record(job)

    def recover_interrupted(self) -> int:
        with self._session_factory() as session:
            result = session.execute(
                update(Job)
                .where(Job.status == JobStatus.RUNNING.value)
                .values(
                    status=JobStatus.QUEUED.value,
                    worker_id=None,
                    started_at=None,
                    progress_message=None,
                    error_code="WORKER_INTERRUPTED",
                    error_message="Worker interrupted before completion",
                )
            )
            session.commit()
            return int(result.rowcount or 0)
