from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.app.api.routes.datasets import DatasetService
from backend.app.core.config import get_settings
from backend.app.db.session import create_database_engine, init_db
from backend.app.main import create_app
from backend.app.services.storage import LocalStorage


def _settings(tmp_path):
    settings = get_settings()
    return replace(settings, storage=replace(settings.storage, root=tmp_path / "data"))


def _register_and_login(client: TestClient, username: str, password: str) -> str:
    assert client.post(
        "/api/auth/register", json={"username": username, "password": password}
    ).status_code == 201
    assert client.post(
        "/api/auth/login", json={"username": username, "password": password}
    ).status_code == 200
    return client.get("/api/auth/me").json()["id"]


def test_uploaded_dataset_survives_service_recreation_and_is_owner_scoped(tmp_path):
    """Would fail if uploads remain only in DatasetService._datasets."""
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        alice_id = _register_and_login(client, "alice", "Correct-Horse-42")
        uploaded = client.post(
            "/api/datasets",
            files={"file": ("voice.wav", b"local bytes", "audio/wav")},
            data={"consent_confirmed": "true"},
        )
        assert uploaded.status_code == 200
        dataset_id = uploaded.json()["dataset_id"]

        owner_records = app.state.dataset_service.list_for_owner(alice_id)
        assert [record.id for record in owner_records] == [dataset_id]
        assert owner_records[0].authorization_confirmed is True
        assert owner_records[0].effective_seconds is None
        assert owner_records[0].asset_relative_paths[0].startswith(f"{dataset_id}/")
        assert ":" not in owner_records[0].asset_relative_paths[0]
        assert not Path(owner_records[0].asset_relative_paths[0]).is_absolute()
        assert app.state.dataset_service.get_for_owner(dataset_id, alice_id).id == dataset_id

        client.post("/api/auth/logout")
        bob_id = _register_and_login(client, "bob", "Correct-Horse-43")
        assert app.state.dataset_service.list_for_owner(bob_id) == []
        with pytest.raises(KeyError):
            app.state.dataset_service.get_for_owner(dataset_id, bob_id)

        rejected = client.post(
            "/api/datasets",
            files={"file": ("voice.wav", b"local bytes", "audio/wav")},
        )

    assert rejected.status_code == 422
    assert rejected.json()["error"]["code"] == "DATASET_AUTHORIZATION_REQUIRED"


class _FakeTranscriber:
    def available(self) -> bool:
        return True

    def transcribe(self, wav_path, language="zh") -> str:
        return "自动转写文本。"


class _FakeEmotionAnalyzer:
    def available(self) -> bool:
        return True

    def analyze(self, wav_path):
        return SimpleNamespace(label="happy", confidence=0.9)


def test_preprocess_persists_effective_seconds(monkeypatch, tmp_path):
    """Would fail if CPU preprocessing returns only a queued GPU job or loses its result."""
    app = create_app(_settings(tmp_path))
    app.state.dataset_service = DatasetService(
        storage=LocalStorage(app.state.dataset_service.storage.root),
        transcriber=_FakeTranscriber(),
        emotion_analyzer=_FakeEmotionAnalyzer(),
    )

    def _fake_prepare(dataset_id, *_args, **_kwargs):
        return SimpleNamespace(
            effective_seconds=6.0,
            rows=[
                SimpleNamespace(
                    segment_id="seg_0001",
                    path=f"datasets/{dataset_id}/segments/seg_0001.wav",
                    duration_seconds=6.0,
                    snr_db=27.0,
                    clipping_ratio=0.0,
                    split="reference",
                    language="zh",
                )
            ],
        )

    monkeypatch.setattr(
        "backend.app.api.routes.datasets.prepare_reference_dataset", _fake_prepare
    )
    with TestClient(app) as client:
        owner_id = _register_and_login(client, "alice", "Correct-Horse-42")
        dataset_id = client.post(
            "/api/datasets",
            files={"file": ("voice.wav", b"local bytes", "audio/wav")},
            data={"consent_confirmed": "true"},
        ).json()["dataset_id"]
        response = client.post(f"/api/datasets/{dataset_id}/preprocess")

    assert response.status_code == 200
    assert response.json() == {
        "dataset_id": dataset_id,
        "effective_seconds": 6.0,
        "status": "review_required",
        "segment_count": 1,
        "snr_warning_db": 20.0,
        "warning_codes": [],
    }
    record = app.state.dataset_service.get_for_owner(dataset_id, owner_id)
    assert record.effective_seconds == 6.0
    assert record.status == "review_required"


def test_init_db_adds_dataset_columns_without_losing_legacy_row(tmp_path):
    """Would fail if the SQLite migration drops legacy rows or omits new columns."""
    database = tmp_path / "legacy.db"
    engine = create_database_engine(f"sqlite+pysqlite:///{database.as_posix()}")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE datasets ("
            "id VARCHAR(64) PRIMARY KEY, owner_user_id VARCHAR(64), "
            "status VARCHAR(32), created_at DATETIME)"
        )
        connection.exec_driver_sql(
            "INSERT INTO datasets (id, owner_user_id, status, created_at) "
            "VALUES ('legacy-1', 'user-1', 'uploaded', '2026-09-06T00:00:00Z')"
        )

    init_db(engine)

    with engine.connect() as connection:
        columns = {
            row[1] for row in connection.exec_driver_sql("PRAGMA table_info(datasets)")
        }
        assert "authorization_confirmed_at" in columns
        assert "effective_seconds" in columns
        assert connection.exec_driver_sql("SELECT id FROM datasets").scalar_one() == "legacy-1"
