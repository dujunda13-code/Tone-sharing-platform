"""Unit tests for the nvidia-smi whole-device GPU memory peak sampler."""

from __future__ import annotations

from subprocess import CompletedProcess

from backend.app.services.gpu_metrics import GpuMemorySampler


def _fake_runner(outputs: list[str | Exception]):
    calls = iter(outputs)

    def runner(cmd, capture_output, text, timeout):  # noqa: ANN001
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return CompletedProcess(args=cmd, returncode=0, stdout=item, stderr="")

    return runner


def test_read_used_mib_parses_nvidia_smi_output():
    sampler = GpuMemorySampler(
        command_runner=_fake_runner(["8188\n"]),
    )

    assert sampler._read_used_mib() == 8188.0


def test_poll_once_tracks_peak_across_polls():
    sampler = GpuMemorySampler(
        command_runner=_fake_runner(["3100\n", "2525\n", "4100\n", "3980\n"]),
    )

    for _ in range(4):
        sampler._poll_once()

    assert sampler.peak_mib == 4100.0


def test_read_used_mib_returns_none_when_nvidia_smi_fails():
    sampler = GpuMemorySampler(
        command_runner=_fake_runner(
            [FileNotFoundError("nvidia-smi"), FileNotFoundError("nvidia-smi"), "123\n"]
        ),
    )

    assert sampler._read_used_mib() is None
    sampler._poll_once()
    assert sampler.peak_mib == 0.0
    sampler._poll_once()
    assert sampler.peak_mib == 123.0


def test_read_used_mib_returns_none_for_nonzero_exit():
    sampler = GpuMemorySampler(
        command_runner=_fake_runner([CompletedProcess(args=[], returncode=1, stdout="", stderr="err")]),
    )

    assert sampler._read_used_mib() is None
