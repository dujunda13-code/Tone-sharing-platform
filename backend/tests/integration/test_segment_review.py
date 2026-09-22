"""Short-reference gate and review loop contract (zero-shot client).

Upload → preprocess must persist the single normalized reference segment
with transcription and emotion state in SQLite (surviving service
recreation); only reviewed, non-empty-text 3–10 second references may
become a voice profile, and missing local models keep the dataset
unreviewable with a fixed public error instead of a GPU job.
"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from backend.app.api.routes.datasets import DatasetService
from backend.app.core.config import get_settings
from backend.app.db.session import create_database_engine, init_db
from backend.app.main import create_app
from backend.app.services.storage import LocalStorage
from backend.app.services.transcription import LocalFunASRTranscriber


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _settings(tmp_path):
    settings = get_settings()
    return replace(settings, storage=replace(settings.storage, root=tmp_path / "data"))


def _write_low_snr_reference(path: Path) -> None:
    sample_rate = 16_000
    seconds = 6
    time = np.arange(sample_rate * seconds, dtype=np.float32) / sample_rate
    rng = np.random.default_rng(20260912)
    voice = 0.08 * np.sin(2 * np.pi * 220 * time)
    noise = 0.05 * rng.standard_normal(time.shape[0])
    sf.write(path, np.clip(voice + noise, -0.8, 0.8), sample_rate, subtype="PCM_16")


def _register_and_login(client: TestClient, username: str, password: str) -> str:
    assert client.post(
        "/api/auth/register", json={"username": username, "password": password}
    ).status_code == 201
    assert client.post(
        "/api/auth/login", json={"username": username, "password": password}
    ).status_code == 200
    return client.get("/api/auth/me").json()["id"]


class FakeTranscriber:
    def __init__(self, text: str = "自动转写文本。") -> None:
        self.text = text

    def available(self) -> bool:
        return True

    def transcribe(self, wav_path, language="zh") -> str:
        return self.text


class FakeEmotionAnalyzer:
    def __init__(self, label: str = "happy", confidence: float = 0.9) -> None:
        self.label = label
        self.confidence = confidence
        self.analyzed_paths: list[Path] = []

    def available(self) -> bool:
        return True

    def analyze(self, wav_path):
        self.analyzed_paths.append(Path(wav_path))
        return SimpleNamespace(label=self.label, confidence=self.confidence)


def _manifest_rows(dataset_id: str):
    return [
        SimpleNamespace(
            segment_id="seg_0001",
            path=f"datasets/{dataset_id}/segments/seg_0001.wav",
            duration_seconds=6.5,
            snr_db=27.0,
            clipping_ratio=0.0,
            warning_codes=(),
            split="reference",
            language="zh",
        )
    ]


def _install_real_service(app, settings, monkeypatch):
    service = DatasetService(
        storage=LocalStorage(settings.storage.root),
        transcriber=FakeTranscriber(),
        emotion_analyzer=FakeEmotionAnalyzer(),
    )
    app.state.dataset_service = service
    monkeypatch.setattr(
        "backend.app.api.routes.datasets.prepare_reference_dataset",
        lambda dataset_id, sources, output_dir=None: SimpleNamespace(
            effective_seconds=6.5, rows=_manifest_rows(dataset_id)
        ),
    )
    return service


def _upload(client: TestClient) -> str:
    response = client.post(
        "/api/datasets",
        files={"file": ("voice.wav", b"local bytes", "audio/wav")},
        data={"consent_confirmed": "true"},
    )
    assert response.status_code == 200
    return response.json()["dataset_id"]


def test_preprocess_persists_segments_and_requires_review(tmp_path, monkeypatch):
    app = create_app(_settings(tmp_path))
    _install_real_service(app, _settings(tmp_path), monkeypatch)
    with TestClient(app) as client:
        alice_id = _register_and_login(client, "alice", "Correct-Horse-42")
        dataset_id = _upload(client)

        response = client.post(f"/api/datasets/{dataset_id}/preprocess")

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == dataset_id
    assert body["status"] == "review_required"
    assert body["effective_seconds"] == 6.5
    assert body["segment_count"] == 1

    segments = app.state.dataset_service.list_segments(dataset_id, alice_id)
    assert [segment.segment_id for segment in segments] == ["seg_0001"]
    first = segments[0]
    assert first.split == "reference"
    assert first.auto_transcript == "自动转写文本。"
    assert first.auto_emotion_label == "happy"
    assert first.auto_emotion_confidence == 0.9
    assert first.reviewed_at is None


def test_preprocess_uses_configured_snr_threshold_across_response_paths(
    tmp_path, monkeypatch
):
    settings = _settings(tmp_path)
    app = create_app(settings)
    app.state.dataset_service = DatasetService(
        storage=LocalStorage(settings.storage.root),
        transcriber=FakeTranscriber(),
        emotion_analyzer=FakeEmotionAnalyzer(),
        snr_warning_db=30.0,
    )
    monkeypatch.setattr(
        "backend.app.api.routes.datasets.prepare_reference_dataset",
        lambda dataset_id, sources, output_dir=None: SimpleNamespace(
            effective_seconds=6.5, rows=_manifest_rows(dataset_id)
        ),
    )

    with TestClient(app) as client:
        _register_and_login(client, "alice", "Correct-Horse-42")
        dataset_id = _upload(client)

        full_preprocess = client.post(f"/api/datasets/{dataset_id}/preprocess")
        fast_preprocess = client.post(f"/api/datasets/{dataset_id}/preprocess")
        segments = client.get(f"/api/datasets/{dataset_id}/segments")

    assert full_preprocess.status_code == 200
    assert full_preprocess.json()["snr_warning_db"] == 30.0
    assert full_preprocess.json()["warning_codes"] == ["SNR_BELOW_RECOMMENDED"]
    assert fast_preprocess.status_code == 200
    assert fast_preprocess.json()["warning_codes"] == ["SNR_BELOW_RECOMMENDED"]
    assert segments.status_code == 200
    assert segments.json()["items"][0]["snr_db"] == 27.0
    assert segments.json()["items"][0]["snr_warning_db"] == 30.0
    assert segments.json()["items"][0]["warning_codes"] == [
        "SNR_BELOW_RECOMMENDED"
    ]


def test_preprocess_uses_emotion_analyzer_injected_by_application(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    analyzer = FakeEmotionAnalyzer()
    monkeypatch.setattr("backend.app.main.EmotionAnalyzer", lambda: analyzer)
    monkeypatch.setattr(
        "backend.app.api.routes.datasets.prepare_reference_dataset",
        lambda dataset_id, sources, output_dir=None: SimpleNamespace(
            effective_seconds=6.5, rows=_manifest_rows(dataset_id)
        ),
    )
    app = create_app(settings)
    app.state.dataset_service.transcriber = FakeTranscriber()

    with TestClient(app) as client:
        _register_and_login(client, "alice", "Correct-Horse-42")
        dataset_id = _upload(client)
        response = client.post(f"/api/datasets/{dataset_id}/preprocess")

    assert response.status_code == 200
    assert len(analyzer.analyzed_paths) == 1


def test_reference_requires_review_and_confirms_ready_for_profile(tmp_path, monkeypatch):
    app = create_app(_settings(tmp_path))
    _install_real_service(app, _settings(tmp_path), monkeypatch)
    with TestClient(app) as client:
        _register_and_login(client, "alice", "Correct-Horse-42")
        dataset_id = _upload(client)
        assert client.post(f"/api/datasets/{dataset_id}/preprocess").status_code == 200

        blocked = client.post(
            "/api/voices",
            json={
                "dataset_id": dataset_id,
                "display_name": "审核测试音色",
                "consent_confirmed": True,
            },
        )
        assert blocked.status_code == 409
        assert blocked.json()["error"]["code"] == "REFERENCE_REVIEW_REQUIRED"

        edited = client.patch(
            f"/api/datasets/{dataset_id}/segments/seg_0001",
            json={
                "transcript": "人工校对后的文本。",
                "emotion_label": "sad",
                "reason": "听录修正",
            },
        )
        assert edited.status_code == 200
        edited_body = edited.json()
        assert edited_body["auto_transcript"] == "自动转写文本。"
        assert edited_body["effective_transcript"] == "人工校对后的文本。"
        assert edited_body["transcript_source"] == "manual"
        assert edited_body["auto_emotion_label"] == "happy"
        assert edited_body["effective_emotion_label"] == "sad"
        assert edited_body["emotion_source"] == "manual"
        assert edited_body["reviewed"] is True

        relanguaged = client.patch(
            f"/api/datasets/{dataset_id}/segments/seg_0001",
            json={"language": "en", "reason": "参考语言修正"},
        )
        assert relanguaged.status_code == 200
        assert relanguaged.json()["language"] == "en"

        listed = client.get(f"/api/datasets/{dataset_id}/segments")
        assert listed.status_code == 200
        listed_body = listed.json()
        assert listed_body["dataset_id"] == dataset_id
        assert listed_body["status"] == "review_required"
        assert listed_body["reviewed_segments"] == 1
        assert listed_body["total_segments"] == 1
        assert listed_body["ready_for_profile"] is False
        assert "auto_transcript" in listed_body["items"][0]
        assert ":" not in listed_body["items"][0]["relative_path"]
        assert not Path(listed_body["items"][0]["relative_path"]).is_absolute()

        confirmed = client.post(
            f"/api/datasets/{dataset_id}/segments/confirm",
            json={"reason": "批量确认自动结果"},
        )
        assert confirmed.status_code == 200
        assert confirmed.json()["status"] == "ready_for_profile"

        listed_after = client.get(f"/api/datasets/{dataset_id}/segments")
        assert listed_after.json()["ready_for_profile"] is True

        allowed = client.post(
            "/api/voices",
            json={
                "dataset_id": dataset_id,
                "display_name": "审核通过音色",
                "consent_confirmed": True,
            },
        )
        assert allowed.status_code == 201


def test_blank_manual_transcript_is_rejected(tmp_path, monkeypatch):
    app = create_app(_settings(tmp_path))
    _install_real_service(app, _settings(tmp_path), monkeypatch)
    with TestClient(app) as client:
        _register_and_login(client, "alice", "Correct-Horse-42")
        dataset_id = _upload(client)
        assert client.post(f"/api/datasets/{dataset_id}/preprocess").status_code == 200

        response = client.patch(
            f"/api/datasets/{dataset_id}/segments/seg_0001",
            json={"transcript": "   ", "reason": "误操作"},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "SEGMENT_TEXT_REQUIRED"


def test_segments_survive_service_recreation_and_are_owner_scoped(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    app = create_app(settings)
    service = _install_real_service(app, settings, monkeypatch)
    with TestClient(app) as client:
        alice_id = _register_and_login(client, "alice", "Correct-Horse-42")
        dataset_id = _upload(client)
        assert client.post(f"/api/datasets/{dataset_id}/preprocess").status_code == 200
        assert (
            client.patch(
                f"/api/datasets/{dataset_id}/segments/seg_0001",
                json={"transcript": "人工校对后的文本。", "reason": "听录修正"},
            ).status_code
            == 200
        )

        recreated = DatasetService(
            storage=LocalStorage(settings.storage.root),
            transcriber=FakeTranscriber(),
            emotion_analyzer=FakeEmotionAnalyzer(),
        )
        segments = recreated.list_segments(dataset_id, owner_user_id=alice_id)
        assert [segment.segment_id for segment in segments] == ["seg_0001"]
        assert segments[0].manual_transcript == "人工校对后的文本。"
        assert segments[0].reviewed_at is not None

        client.post("/api/auth/logout")
        _register_and_login(client, "bob", "Correct-Horse-43")
        foreign = client.get(f"/api/datasets/{dataset_id}/segments")
        assert foreign.status_code == 404
        assert foreign.json()["error"]["code"] == "DATASET_NOT_FOUND"
    assert service is not None


def test_missing_local_models_keep_dataset_untrainable(tmp_path):
    app = create_app(_settings(tmp_path))
    app.state.dataset_service.transcriber = LocalFunASRTranscriber(
        model_dir=tmp_path / "models" / "funasr" / "paraformer-zh",
        models_root=tmp_path / "models",
    )
    with TestClient(app) as client:
        alice_id = _register_and_login(client, "alice", "Correct-Horse-42")
        dataset_id = _upload(client)

        response = client.post(f"/api/datasets/{dataset_id}/preprocess")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "TRANSCRIPTION_MODEL_UNAVAILABLE"
    record = app.state.dataset_service.get_for_owner(dataset_id, alice_id)
    assert record.status == "uploaded"
    assert record.effective_seconds is None
    assert app.state.dataset_service.list_segments(dataset_id, alice_id) == []


def test_real_reference_upload_flows_through_review_to_ready_for_profile(
    tmp_path, monkeypatch
):
    """Authorization, quality gates, transcription and review precede the profile."""
    app = create_app(_settings(tmp_path))
    service = DatasetService(
        storage=LocalStorage(_settings(tmp_path).storage.root),
        transcriber=FakeTranscriber(),
        emotion_analyzer=FakeEmotionAnalyzer(),
    )
    app.state.dataset_service = service
    with TestClient(app) as client:
        alice_id = _register_and_login(client, "alice", "Correct-Horse-42")
        uploaded = client.post(
            "/api/datasets",
            files={
                "file": (
                    "reference.wav",
                    (FIXTURES / "clean_10s.wav").read_bytes(),
                    "audio/wav",
                )
            },
            data={"consent_confirmed": "true"},
        )
        dataset_id = uploaded.json()["dataset_id"]
        response = client.post(f"/api/datasets/{dataset_id}/preprocess")

    assert uploaded.status_code == 200
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "review_required"
    assert body["segment_count"] == 1
    assert 3.0 <= body["effective_seconds"] <= 10.0

    segments = service.list_segments(dataset_id, owner_user_id=alice_id)
    assert len(segments) == 1
    assert segments[0].split == "reference"
    assert segments[0].auto_transcript == "自动转写文本。"


def test_low_snr_reference_warns_and_can_be_confirmed(tmp_path):
    source = tmp_path / "low-snr.wav"
    _write_low_snr_reference(source)
    app = create_app(_settings(tmp_path))
    app.state.dataset_service = DatasetService(
        storage=LocalStorage(_settings(tmp_path).storage.root),
        transcriber=FakeTranscriber(),
        emotion_analyzer=FakeEmotionAnalyzer(),
    )
    with TestClient(app) as client:
        _register_and_login(client, "alice", "Correct-Horse-42")
        uploaded = client.post(
            "/api/datasets",
            files={"file": ("low-snr.wav", source.read_bytes(), "audio/wav")},
            data={"consent_confirmed": "true"},
        )
        dataset_id = uploaded.json()["dataset_id"]

        preprocess = client.post(f"/api/datasets/{dataset_id}/preprocess")

        assert preprocess.status_code == 200
        assert preprocess.json()["warning_codes"] == ["SNR_BELOW_RECOMMENDED"]
        segments = client.get(f"/api/datasets/{dataset_id}/segments").json()
        assert segments["items"][0]["snr_db"] < 20.0
        assert segments["items"][0]["snr_warning_db"] == 20.0
        assert segments["items"][0]["warning_codes"] == ["SNR_BELOW_RECOMMENDED"]
        assert (
            client.post(
                f"/api/datasets/{dataset_id}/segments/confirm",
                json={"reason": "确认低信噪比风险"},
            ).json()["status"]
            == "ready_for_profile"
        )


def test_clipping_remains_blocking_before_review(tmp_path):
    """Clipping—not low SNR—keeps a failing upload away from review."""
    app = create_app(_settings(tmp_path))
    service = DatasetService(
        storage=LocalStorage(_settings(tmp_path).storage.root),
        transcriber=FakeTranscriber(),
        emotion_analyzer=FakeEmotionAnalyzer(),
    )
    app.state.dataset_service = service
    with TestClient(app) as client:
        alice_id = _register_and_login(client, "alice", "Correct-Horse-42")
        uploaded = client.post(
            "/api/datasets",
            files={
                "file": (
                    "clipped.wav",
                    (FIXTURES / "clipped_3s.wav").read_bytes(),
                    "audio/wav",
                )
            },
            data={"consent_confirmed": "true"},
        )
        dataset_id = uploaded.json()["dataset_id"]
        response = client.post(f"/api/datasets/{dataset_id}/preprocess")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "REFERENCE_DURATION_OUT_OF_RANGE"
    assert service.list_segments(dataset_id, owner_user_id=alice_id) == []


def test_init_db_adds_segment_table_without_dropping_legacy_rows(tmp_path):
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
        tables = {
            row[0]
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "dataset_segments" in tables
        assert connection.exec_driver_sql("SELECT id FROM datasets").scalar_one() == "legacy-1"
