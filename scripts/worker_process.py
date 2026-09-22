"""Run the single local queue worker with the real local assembly (Phase D2).

The worker is assembled exclusively from pinned local components
(`backend.app.services.local_worker.assemble_local_worker`): the GPT-SoVITS
dataset preparer and training adapter, the disentangler trainer, the guarded
synthesis pipeline and the CAM++ evaluation. Local models that a deployment
has not provided fail their jobs with fixed public errors — the worker never
fabricates weights, audio or scores.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Running `python scripts/worker_process.py` puts scripts/ (not the repo root)
# on sys.path, so the backend package needs an explicit path bootstrap to work
# no matter which cwd or PYTHONPATH the caller provides.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.services.local_worker import assemble_local_worker  # noqa: E402
from backend.app.workers.worker_loop import run_worker_cycle  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    python_executable = Path(sys.executable)
    worker = assemble_local_worker(root=root, python_executable=python_executable)
    recovered = worker.recover_interrupted()
    print(f"GPU_WORKER=REAL_ASSEMBLY recovered_jobs={recovered}", flush=True)
    while True:
        run_worker_cycle(worker)
        time.sleep(0.25)


if __name__ == "__main__":
    main()
