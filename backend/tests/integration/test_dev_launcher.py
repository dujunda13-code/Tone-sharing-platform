"""Guard the local dev launcher scripts.

`scripts/run-dev.ps1` is documented to start with Windows PowerShell
(`powershell -ExecutionPolicy Bypass -File ...`), so the script must stay
runnable on the 5.1 engine that ships with Windows, not only on PowerShell 7.
`scripts/worker_process.py` is started by that script from the repository root;
Python puts the script's own directory (`scripts/`) on `sys.path` instead of
the repository root, so the worker must bootstrap the import path itself
instead of relying on `PYTHONPATH` being set by the caller.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
WORKER_BANNER_TIMEOUT_SECONDS = 60


def test_worker_process_starts_without_pythonpath_from_any_directory(tmp_path: Path):
    """The dev worker must reach its fail-closed polling loop without PYTHONPATH.

    run-dev.ps1 starts `python scripts/worker_process.py` with no PYTHONPATH, and
    a bare `python scripts/worker_process.py` from any directory must not die on
    `ModuleNotFoundError: backend` (which makes the launcher kill backend and
    frontend too). The worker runs against a throwaway cwd so the dev database
    under `data/app.db` is not touched by the test.
    """
    workdir = tmp_path / "cwd"
    (workdir / "data").mkdir(parents=True)  # sqlite:///data/app.db resolves under cwd
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    process = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "worker_process.py")],
        cwd=workdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines: list[str] = []

    def drain() -> None:
        if process.stdout is not None:
            for line in process.stdout:
                lines.append(line)

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    try:
        banner_seen = False
        deadline = time.monotonic() + WORKER_BANNER_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if any("GPU_WORKER=" in line for line in lines):
                banner_seen = True
                break
            exit_code = process.poll()
            if exit_code is not None:
                reader.join(timeout=2)
                pytest.fail(
                    f"worker_process.py exited with code {exit_code} before printing its banner:\n"
                    + "".join(lines)
                )
            time.sleep(0.1)
        if not banner_seen:
            pytest.fail(
                f"worker_process.py did not print its banner within {WORKER_BANNER_TIMEOUT_SECONDS}s:\n"
                + "".join(lines)
            )
        combined = "".join(lines)
        assert "ModuleNotFoundError" not in combined
        # After the banner the worker must keep polling the queue instead of exiting.
        time.sleep(2)
        assert process.poll() is None
    finally:
        process.kill()
        process.wait(timeout=10)
        reader.join(timeout=2)


def test_worker_process_has_no_placeholder_adapters():
    """Phase D2: the launcher worker must be the real local assembly.

    `UnconfiguredAdapters`/`LOCAL_FAIL_CLOSED` were the pre-D2 placeholders;
    the delivered worker delegates to `assemble_local_worker`, which wires the
    real GPT-SoVITS preparer/adapter, disentangler trainer, guarded synthesis
    pipeline and evaluation runner.
    """
    source = (ROOT / "scripts" / "worker_process.py").read_text(encoding="utf-8")
    assert "UnconfiguredAdapters" not in source
    assert "LOCAL_FAIL_CLOSED" not in source
    assert "assemble_local_worker" in source
    assembler = (
        ROOT / "backend" / "app" / "services" / "local_worker.py"
    ).read_text(encoding="utf-8")
    for real_component in (
        "GPTSoVITSDatasetPreparer",
        "GPTSoVITSAdapter",
        "DisentanglerTrainer",
        "SynthesisPipeline",
    ):
        assert real_component in assembler


def test_worker_banner_reports_real_assembly(tmp_path: Path):
    """The startup banner must identify the real assembly, not fail-closed mode."""
    workdir = tmp_path / "cwd"
    (workdir / "data").mkdir(parents=True)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    process = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "worker_process.py")],
        cwd=workdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines: list[str] = []

    def drain() -> None:
        if process.stdout is not None:
            for line in process.stdout:
                lines.append(line)

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    try:
        banner = None
        deadline = time.monotonic() + WORKER_BANNER_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            banner = next((line for line in lines if line.startswith("GPU_WORKER=")), None)
            if banner is not None:
                break
            exit_code = process.poll()
            if exit_code is not None:
                reader.join(timeout=2)
                pytest.fail(
                    f"worker_process.py exited with code {exit_code} before printing its banner:\n"
                    + "".join(lines)
                )
            time.sleep(0.1)
        if banner is None:
            pytest.fail("worker_process.py did not print a GPU_WORKER banner in time")
        assert banner.startswith("GPU_WORKER=REAL")
        time.sleep(1)
        assert process.poll() is None
    finally:
        process.kill()
        process.wait(timeout=10)
        reader.join(timeout=2)


def test_run_dev_script_stays_windows_powershell_5_1_compatible():
    """run-dev.ps1 must not use PowerShell-7-only process launch internals.

    `ProcessStartInfo.ArgumentList` does not exist on the .NET Framework behind
    Windows PowerShell 5.1 (the documented `powershell` entry point), and
    `add_OutputDataReceived` scriptblock delegates crash with "no Runspace
    available" on threadpool threads under `pwsh`. Both regressions took down
    the whole dev stack, so guard against them coming back.
    """
    script = (ROOT / "scripts" / "run-dev.ps1").read_text(encoding="utf-8-sig")
    assert "ArgumentList.Add" not in script
    assert "add_OutputDataReceived" not in script
    assert "add_ErrorDataReceived" not in script
    assert "Start-Process" in script


def test_run_dev_script_uses_direct_python_and_node_without_cmd_wrappers():
    script = (ROOT / "scripts" / "run-dev.ps1").read_text(encoding="utf-8-sig")
    assert "npm.cmd" not in script
    assert "cmd.exe" not in script
    assert "$env:ComSpec" not in script
    assert "node.exe" in script


def test_run_dev_script_uses_exact_process_identity_for_cleanup_and_quoting():
    script = (ROOT / "scripts" / "run-dev.ps1").read_text(encoding="utf-8-sig")
    assert "ConvertTo-ProcessArgument" in script
    assert "command_line_marker" in script
    assert "*scripts/worker_process.py*" not in script
    assert "*vite*" not in script
    assert "expectedNames" not in script


@pytest.mark.skipif(
    shutil.which("powershell") is None or os.name != "nt",
    reason="Windows PowerShell is not available on this host",
)
def test_run_dev_script_parses_under_windows_powershell():
    """The documented `powershell -File scripts/run-dev.ps1` entry must at least parse on 5.1."""
    script_path = ROOT / "scripts" / "run-dev.ps1"
    check = (
        "$tokens = $null; $errors = $null; "
        f"[System.Management.Automation.Language.Parser]::ParseFile('{script_path}', [ref]$tokens, [ref]$errors) | Out-Null; "
        "if ($errors.Count -gt 0) { $errors | ForEach-Object { Write-Output $_.Message }; exit 1 }"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", check],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
