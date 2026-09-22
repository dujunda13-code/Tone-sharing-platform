from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from backend.app.services.authorized_dataset import (
    AuthorizedDatasetError,
    load_authorized_manifest,
)


def _row(path: str) -> dict[str, object]:
    return {
        "segment_id": "seg-0001",
        "path": path,
        "speaker": "voice-1",
        "language": "zh",
        "text": "授权样本",
        "duration_seconds": 600.0,
        "snr_db": 30.0,
        "clipping_ratio": 0.0,
        "split": "train",
    }


def test_authorized_manifest_resolves_only_local_audio(tmp_path: Path):
    storage = tmp_path / "data"
    audio = storage / "datasets" / "d1" / "segments" / "seg-0001.wav"
    audio.parent.mkdir(parents=True)
    sf.write(audio, np.zeros(16_000, dtype=np.float32), 16_000)
    manifest = storage / "datasets" / "d1" / "manifest.jsonl"
    manifest.write_text(json.dumps(_row("datasets/d1/segments/seg-0001.wav")) + "\n", encoding="utf-8")

    loaded = load_authorized_manifest(manifest, storage)

    assert loaded.dataset_id == "d1"
    assert loaded.effective_seconds == 600.0
    assert loaded.rows[0].path.startswith("datasets/")


def test_authorized_manifest_rejects_audio_outside_local_storage(tmp_path: Path):
    storage = tmp_path / "data"
    manifest = storage / "datasets" / "d1" / "manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(_row("C:/outside/audio.wav")) + "\n", encoding="utf-8")

    with pytest.raises(AuthorizedDatasetError, match="local storage"):
        load_authorized_manifest(manifest, storage)
