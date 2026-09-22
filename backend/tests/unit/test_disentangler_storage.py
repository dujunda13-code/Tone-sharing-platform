import shutil
from pathlib import Path

import pytest
import torch

from backend.app.services import disentanglement


def test_adapter_weight_directory_rejects_paths_outside_local_data_and_models(tmp_path):
    with pytest.raises(ValueError, match="data/ or models/"):
        disentanglement.validate_adapter_output_dir(tmp_path)


def test_adapter_weight_directory_accepts_local_data_and_models_paths():
    assert disentanglement.validate_adapter_output_dir("data/profiles/profile-a/adapter") == Path(
        "data/profiles/profile-a/adapter"
    ).resolve()
    assert disentanglement.validate_adapter_output_dir("models/adapters/profile-a") == Path(
        "models/adapters/profile-a"
    ).resolve()


def _artifact_directory(tmp_path: Path) -> tuple[Path, Path]:
    root = (Path("data") / "temp" / "pytest-task7-atomic-adapter").resolve()
    return root, (root / tmp_path.name).resolve()


def test_adapter_payload_is_atomically_published_without_a_temporary_artifact(tmp_path):
    root, artifact_dir = _artifact_directory(tmp_path)
    destination = artifact_dir / "adapter.pt"
    try:
        published = disentanglement.atomic_save_adapter_payload(
            destination,
            {"metadata": {"test_marker": "published"}},
        )

        assert published == destination
        assert torch.load(published, weights_only=True)["metadata"]["test_marker"] == "published"
        assert not list(artifact_dir.glob(".adapter.pt.*.tmp"))
    finally:
        if artifact_dir.exists():
            assert artifact_dir.is_relative_to(root)
            shutil.rmtree(artifact_dir)


def test_failed_adapter_serialization_leaves_no_partial_output(tmp_path, monkeypatch):
    root, artifact_dir = _artifact_directory(tmp_path)
    destination = artifact_dir / "adapter.pt"

    def fail_after_writing_temporary(_: object, temporary_path: Path) -> None:
        Path(temporary_path).write_bytes(b"partial")
        raise OSError("serialization failed")

    monkeypatch.setattr(disentanglement.torch, "save", fail_after_writing_temporary)
    try:
        with pytest.raises(OSError, match="serialization failed"):
            disentanglement.atomic_save_adapter_payload(destination, {"metadata": {}})

        assert not destination.exists()
        assert not list(artifact_dir.glob(".adapter.pt.*.tmp"))
    finally:
        if artifact_dir.exists():
            assert artifact_dir.is_relative_to(root)
            shutil.rmtree(artifact_dir)
