import sqlite3

import pytest
from sqlalchemy.exc import OperationalError

from backend.app.workers.worker_loop import SQLITE_LOCK_RETRY_DELAY_SECONDS, run_worker_cycle


class _LockingWorker:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def run_one(self) -> None:
        self.calls += 1
        raise self.error


def _sqlite_operational_error(message: str) -> OperationalError:
    return OperationalError("BEGIN IMMEDIATE", {}, sqlite3.OperationalError(message))


def test_worker_cycle_waits_and_keeps_running_when_sqlite_is_temporarily_locked():
    worker = _LockingWorker(_sqlite_operational_error("database is locked"))
    delays: list[float] = []

    completed = run_worker_cycle(worker, sleep=delays.append)

    assert completed is False
    assert worker.calls == 1
    assert delays == [SQLITE_LOCK_RETRY_DELAY_SECONDS]


def test_worker_cycle_does_not_hide_non_lock_database_errors():
    worker = _LockingWorker(_sqlite_operational_error("disk I/O error"))

    with pytest.raises(OperationalError, match="disk I/O error"):
        run_worker_cycle(worker, sleep=lambda _: None)
