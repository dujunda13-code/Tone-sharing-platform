"""Zero-shot voice creation API contract (no client training anywhere)."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend.app.api.routes.datasets import DatasetService
from backend.app.core.config import get_settings
from backend.app.main import create_app
from backend.app.schemas.common import JobKind
from backend.app.services.storage import LocalStorage


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


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


def _install_reference_service(app, settings, monkeypatch, *, snr_db=27.0):
    service = DatasetService(
        storage=LocalStorage(settings.storage.root),
        transcriber=_FakeTranscriber(),
        emotion_analyzer=_FakeEmotionAnalyzer(),
        snr_warning_db=settings.audio.snr_warning_db,
    )
    app.state.dataset_service = service
    monkeypatch.setattr(
        "backend.app.api.routes.datasets.prepare_reference_dataset",
        lambda dataset_id, sources, output_dir=None: SimpleNamespace(
            effective_seconds=6.5,
            rows=[
                SimpleNamespace(
                    segment_id="seg_0001",
                    path=f"datasets/{dataset_id}/segments/seg_0001.wav",
                    duration_seconds=6.5,
                    snr_db=snr_db,
                    clipping_ratio=0.0,
                    split="reference",
                    language="zh",
                )
            ],
        ),
    )
    return service


def _prepare_confirmed_dataset(app, client: TestClient, monkeypatch) -> str:
    # The caller must already have installed a tmp-rooted reference service.
    uploaded = client.post(
        "/api/datasets",
        files={
            "file": ("reference.wav", b"local bytes", "audio/wav"),
        },
        data={"consent_confirmed": "true"},
    )
    assert uploaded.status_code == 200
    dataset_id = uploaded.json()["dataset_id"]
    assert client.post(f"/api/datasets/{dataset_id}/preprocess").status_code == 200
    confirmed = client.post(
        f"/api/datasets/{dataset_id}/segments/confirm",
        json={"reason": "批量确认自动结果"},
    )
    assert confirmed.status_code == 200
    assert confirmed.json()["status"] == "ready_for_profile"
    return dataset_id


def _valid_synthesis_payload(profile_id: str) -> dict:
    return {
        "voice_profile_id": profile_id,
        "text": "safe local text",
        "text_lang": "zh",
        "emotion": {"mode": "manual", "label": "happy", "strength": 0.7},
        "consent_confirmed": True,
    }


def test_create_voice_returns_ready_zero_shot_profile_without_any_training_job(
    tmp_path, monkeypatch
):
    settings = _settings(tmp_path)
    app = create_app(settings)
    _install_reference_service(app, settings, monkeypatch, snr_db=11.361)
    with TestClient(app) as client:
        owner_id = _register_and_login(client, "alice", "Correct-Horse-42")
        dataset_id = _prepare_confirmed_dataset(app, client, monkeypatch)

        jobs_before = app.state.queue.count_jobs()
        response = client.post(
            "/api/voices",
            json={"dataset_id": dataset_id, "display_name": "零样本音色"},
        )
        jobs_after = app.state.queue.count_jobs()

        profile_id = response.json()["id"]
        stored = app.state.profiles.get(profile_id, owner_user_id=owner_id)
        queued = client.post(
            "/api/syntheses", json=_valid_synthesis_payload(profile_id)
        )
        queued_job = app.state.queue.get(
            queued.json()["job_id"], owner_user_id=owner_id
        )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "ready"
    assert body["mode"] == "zero_shot"
    assert body["base_model_id"] == "gpt-sovits-v2proplus-official"
    assert body["reference_emotions"] == ["happy"]
    assert body["reference_snr_db"] == 11.361
    assert body["quality_warning_codes"] == ["SNR_BELOW_RECOMMENDED"]
    assert jobs_before == 0
    assert jobs_after == 0
    assert queued.status_code == 202
    assert queued_job.kind is JobKind.SYNTHESIZE
    assert app.state.queue.count_jobs() == 1
    assert stored.public_weight_dir is None
    assert stored.reference_asset_id
    assert stored.reference_segment_id == "seg_0001"
    assert stored.prompt_text == "自动转写文本。"
    assert stored.prompt_language == "zh"
    assert not (settings.storage.root / "features").exists()
    assert not (settings.storage.root / "profiles" / profile_id).exists()
    assert not any(
        "checkpoint" in path.name.casefold()
        for path in settings.storage.root.rglob("*")
    )


def test_create_voice_requires_reviewed_reference(tmp_path, monkeypatch):
    app = create_app(_settings(tmp_path))
    _install_reference_service(app, _settings(tmp_path), monkeypatch)
    with TestClient(app) as client:
        _register_and_login(client, "alice", "Correct-Horse-42")
        uploaded = client.post(
            "/api/datasets",
            files={"file": ("reference.wav", b"local bytes", "audio/wav")},
            data={"consent_confirmed": "true"},
        )
        dataset_id = uploaded.json()["dataset_id"]
        # Preprocessed but not confirmed yet.
        assert client.post(f"/api/datasets/{dataset_id}/preprocess").status_code == 200

        response = client.post(
            "/api/voices",
            json={"dataset_id": dataset_id, "display_name": "零样本音色"},
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "REFERENCE_REVIEW_REQUIRED"


def test_create_voice_rejects_out_of_range_reference_duration(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        owner_id = _register_and_login(client, "alice", "Correct-Horse-42")
        app.state.dataset_service.register_dataset(
            "long-reference",
            effective_seconds=10.01,
            owner_user_id=owner_id,
            authorization_confirmed=True,
        )

        response = client.post(
            "/api/voices",
            json={"dataset_id": "long-reference", "display_name": "超长音色"},
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "REFERENCE_DURATION_OUT_OF_RANGE"


def test_create_voice_rejects_empty_confirmed_transcript(tmp_path):
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        owner_id = _register_and_login(client, "alice", "Correct-Horse-42")
        app.state.dataset_service.register_dataset(
            "silent-reference",
            effective_seconds=6.0,
            owner_user_id=owner_id,
            authorization_confirmed=True,
        )
        with app.state.dataset_service.engine.begin() as connection:
            connection.exec_driver_sql(
                "UPDATE dataset_segments SET auto_transcript='', manual_transcript=NULL "
                "WHERE dataset_id='silent-reference'"
            )

        response = client.post(
            "/api/voices",
            json={"dataset_id": "silent-reference", "display_name": "静音音色"},
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "REFERENCE_REVIEW_REQUIRED"


def test_create_voice_fails_closed_when_base_model_is_not_registered(tmp_path):
    settings = get_settings()
    settings = replace(
        settings,
        storage=replace(settings.storage, root=tmp_path / "data"),
        models=replace(settings.models, active_voice_base="unknown-base"),
    )
    app = create_app(settings)
    with TestClient(app) as client:
        owner_id = _register_and_login(client, "alice", "Correct-Horse-42")
        app.state.dataset_service.register_dataset(
            "dataset-1",
            effective_seconds=6.0,
            owner_user_id=owner_id,
            authorization_confirmed=True,
        )

        response = client.post(
            "/api/voices",
            json={"dataset_id": "dataset-1", "display_name": "基础模型音色"},
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "BASE_MODEL_UNAVAILABLE"


def test_create_voice_rejects_foreign_dataset(tmp_path, monkeypatch):
    app = create_app(_settings(tmp_path))
    _install_reference_service(app, _settings(tmp_path), monkeypatch)
    with TestClient(app) as client:
        alice_id = _register_and_login(client, "alice", "Correct-Horse-42")
        dataset_id = _prepare_confirmed_dataset(app, client, monkeypatch)
        client.post("/api/auth/logout")
        _register_and_login(client, "bob", "Correct-Horse-43")

        response = client.post(
            "/api/voices",
            json={"dataset_id": dataset_id, "display_name": "零样本音色"},
        )
    del alice_id

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DATASET_NOT_FOUND"


def test_reference_endpoint_adds_non_primary_emotion_reference(tmp_path, monkeypatch):
    app = create_app(_settings(tmp_path))
    _install_reference_service(app, _settings(tmp_path), monkeypatch)
    with TestClient(app) as client:
        owner_id = _register_and_login(client, "alice", "Correct-Horse-42")
        first_dataset = _prepare_confirmed_dataset(app, client, monkeypatch)
        created = client.post(
            "/api/voices",
            json={"dataset_id": first_dataset, "display_name": "首参考音色"},
        )
        profile_id = created.json()["id"]

        second_dataset = _prepare_confirmed_dataset(app, client, monkeypatch)
        added = client.post(
            f"/api/voices/{profile_id}/references",
            json={"dataset_id": second_dataset},
        )

        references = app.state.profiles.references(profile_id, owner_id)

    assert added.status_code == 201
    body = added.json()
    assert body["mode"] == "zero_shot"
    assert body["reference_emotions"] == ["happy", "happy"]
    assert len(references) == 2
    assert sum(1 for reference in references if reference.is_primary) == 1


def test_train_endpoint_is_fail_closed_without_side_effects(tmp_path, monkeypatch):
    app = create_app(_settings(tmp_path))
    _install_reference_service(app, _settings(tmp_path), monkeypatch)
    with TestClient(app) as client:
        owner_id = _register_and_login(client, "alice", "Correct-Horse-42")
        dataset_id = _prepare_confirmed_dataset(app, client, monkeypatch)
        created = client.post(
            "/api/voices",
            json={"dataset_id": dataset_id, "display_name": "停用音色"},
        )
        profile_id = created.json()["id"]
        jobs_before = app.state.queue.count_jobs()

        response = client.post(
            f"/api/voices/{profile_id}/train", json={"consent_confirmed": True}
        )
        jobs_after = app.state.queue.count_jobs()
        profile = client.get(f"/api/voices/{profile_id}")

    del owner_id
    assert response.status_code == 409
    body = response.json()
    assert body["error"]["code"] == "CLIENT_TRAINING_DISABLED"
    assert body["error"]["request_id"] == response.headers["X-Request-ID"]
    assert jobs_before == 0
    assert jobs_after == 0
    assert profile.status_code == 200
    assert profile.json()["status"] == "ready"
    # No train (or any) Job was ever created by creation or the disabled endpoint.
    assert jobs_after == jobs_before


def test_voice_api_requires_and_returns_display_name(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    app = create_app(settings)
    _install_reference_service(app, settings, monkeypatch, snr_db=27.0)
    with TestClient(app) as client:
        _register_and_login(client, "namer", "Correct-Horse-42")
        ready_dataset_id = _prepare_confirmed_dataset(app, client, monkeypatch)

        missing = client.post("/api/voices", json={"dataset_id": ready_dataset_id})
        created = client.post(
            "/api/voices",
            json={"dataset_id": ready_dataset_id, "display_name": "产品旁白"},
        )

    assert missing.status_code == 422
    assert created.status_code == 201
    assert created.json()["display_name"] == "产品旁白"


def test_legacy_voice_summary_uses_stable_fallback_name(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    app = create_app(settings)
    _install_reference_service(app, settings, monkeypatch, snr_db=27.0)
    with TestClient(app) as client:
        owner_id = _register_and_login(client, "legacy", "Correct-Horse-42")
        ready_dataset_id = _prepare_confirmed_dataset(app, client, monkeypatch)
        # A profile created before the display-name feature has a NULL name.
        legacy_profile = app.state.profiles.create(
            ready_dataset_id, owner_user_id=owner_id
        )

        listed = client.get("/api/voices")

    assert listed.status_code == 200
    item = next(
        item
        for item in listed.json()["items"]
        if item["id"] == legacy_profile.id
    )
    assert item["display_name"] == f"音色 {legacy_profile.id[:6]}"
