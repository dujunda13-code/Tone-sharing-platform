"""Resilient polling helpers for the single local GPU worker."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Protocol

from sqlalchemy.exc import OperationalError


SQLITE_LOCK_RETRY_DELAY_SECONDS = 1.0
LOGGER = logging.getLogger(__name__)


class WorkerCycleTarget(Protocol):
    def run_one(self) -> object: ...


def run_worker_cycle(
    worker: WorkerCycleTarget,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Run one worker poll and retain the process through transient SQLite locks."""
    try:
        worker.run_one()
    except OperationalError as exc:
        if not _is_transient_sqlite_lock(exc):
            raise
        LOGGER.warning(
            "local SQLite queue is busy; retaining GPU worker and retrying in %.1f seconds",
            SQLITE_LOCK_RETRY_DELAY_SECONDS,
        )
        sleep(SQLITE_LOCK_RETRY_DELAY_SECONDS)
        return False
    return True


def _is_transient_sqlite_lock(exc: OperationalError) -> bool:
    message = str(exc.orig).lower()
    return "database is locked" in message or "database is busy" in message
