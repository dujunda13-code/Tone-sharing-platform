from pathlib import Path

import pytest

from backend.app.core.errors import PathOutsideStorage
from backend.app.services.storage import LocalStorage


def test_storage_rejects_escape(tmp_path: Path):
    storage = LocalStorage(tmp_path)

    with pytest.raises(PathOutsideStorage):
        storage.resolve("uploads", "../../secret.txt")


def test_storage_publishes_atomically_inside_area(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    source = tmp_path / "temp" / "generated.wav"
    source.parent.mkdir()
    source.write_bytes(b"audio")

    published = storage.atomic_publish(source, "outputs", "job-1/audio.wav")

    assert published == tmp_path / "outputs" / "job-1" / "audio.wav"
    assert published.read_bytes() == b"audio"
    assert not source.exists()
