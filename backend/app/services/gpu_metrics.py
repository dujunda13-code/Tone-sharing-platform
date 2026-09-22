"""Whole-device GPU memory peak sampling via nvidia-smi.

The synthesis inference runs in child processes, so parent-process
``torch.cuda.max_memory_allocated`` cannot observe it. The sampler polls
``nvidia-smi`` for total device memory used while a track runs and records
the observed peak (MiB).
"""

from __future__ import annotations

import subprocess
import threading
from typing import Callable


class GpuMemorySampler:
    """Polls nvidia-smi and records the peak whole-device memory usage."""

    def __init__(
        self,
        *,
        interval_seconds: float = 0.5,
        command_runner: Callable[..., object] | None = None,
    ) -> None:
        self.interval_seconds = interval_seconds
        self.peak_mib: float = 0.0
        self._command_runner = command_runner or subprocess.run
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "GpuMemorySampler":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._poll_once()
            self._stop.wait(self.interval_seconds)

    def _poll_once(self) -> None:
        value = self._read_used_mib()
        if value is not None and value > self.peak_mib:
            self.peak_mib = value

    def _read_used_mib(self) -> float | None:
        try:
            result = self._command_runner(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if getattr(result, "returncode", 1) != 0:
            return None
        try:
            return float(str(result.stdout).strip().splitlines()[0])
        except (ValueError, IndexError):
            return None
