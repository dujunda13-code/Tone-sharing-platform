from dataclasses import replace
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.routes.datasets import DatasetService, router as dataset_router
from backend.app.api.dependencies import get_current_user
from backend.app.api.routes import jobs as jobs_module
from backend.app.api.routes.safety import WatermarkVerificationError
from backend.app.api.routes import safety as safety_module
from backend.app.api.routes.synthesis import SynthesisService, router as synthesis_router
from backend.app.core.config import get_settings
from backend.app.db.session import create_database_engine
from backend.app.main import create_app
from backend.app.schemas.auth import UserRecord
from backend.app.schemas.common import JobKind
from backend.app.services.job_queue import JobQueue
from backend.app.services.storage import LocalStorage
from backend.app.services.workspace import WorkspaceService
from backend.tests.integration.auth_test_helpers import install_authenticated_user


def _settings(tmp_path):
    settings = get_settings()
    return replace(settings, storage=replace(settings.storage, root=tmp_path / "data"))


def test_local_app_routes_share_state_and_return_contract_errors(tmp_path):
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        login = client.post(
            "/api/auth/register",
            json={"username": "route-user", "password": "Correct-Horse-42"},
        )
        assert login.status_code == 201
        login = client.post(
            "/api/auth/login",
            json={"username": "route-user", "password": "Correct-Horse-42"},
        )
        assert login.status_code == 200
        route_user = app.state.auth_service.authenticate("route-user", "Correct-Horse-42")
        upload = client.post(
            "/api/datasets",
            files={"file": ("sample.wav", b"local sample", "audio/wav")},
            data={"consent_confirmed": "true"},
        )
        dataset_id = upload.json()["dataset_id"]
        app.state.dataset_service.register_dataset(
            dataset_id,
            effective_seconds=6.5,
            owner_user_id=route_user.id,
            authorization_confirmed=True,
        )
        missing_dataset = client.post("/api/datasets/missing/preprocess")

        created = client.post(
            "/api/voices",
            json={"dataset_id": dataset_id, "display_name": "路由测试音色"},
        )
        profile_id = created.json()["id"]
        missing_profile = client.get("/api/voices/missing")
        missing_train = client.post(
            "/api/voices/missing/train", json={"consent_confirmed": True}
        )
        missing_segment = client.put(
            f"/api/datasets/{dataset_id}/segments/missing/emotion",
            json={"label": "happy", "reason": "manual review"},
        )

        assert upload.status_code == 200
        assert missing_dataset.status_code == 422
        assert missing_dataset.json()["error"]["code"] == "DATASET_NOT_FOUND"
        assert created.status_code == 201
        assert missing_profile.status_code == 404
        # Client training is fail-closed for every profile without existence probing.
        assert missing_train.status_code == 409
        train_body = missing_train.json()
        train_code = train_body.get("error", {}).get("code") or train_body.get(
            "detail", {}
        ).get("code")
        assert train_code == "CLIENT_TRAINING_DISABLED"
        assert missing_segment.status_code == 404

        app.state.dataset_service.register_dataset(
            "ready-dataset",
            effective_seconds=6.5,
            owner_user_id=route_user.id,
            authorization_confirmed=True,
        )
        preprocessed = client.post("/api/datasets/ready-dataset/preprocess")
        assert preprocessed.status_code == 200

        missing_evaluation = client.post("/api/voices/missing/evaluate")
        missing_job = client.get("/api/jobs/missing")

        profile = app.state.profiles.get(profile_id)
        # Zero-shot profiles are born ready and never transition. Simulate the
        # legacy (offline-training) lifecycle directly on the store so the
        # synthesis/evaluation route contracts stay exercised until the
        # zero-shot synthesis task rewrites them.
        legacy_profile = app.state.profiles.create(
            profile.dataset_id, owner_user_id=route_user.id
        )
        app.state.profiles.mark_queued(legacy_profile.id)
        app.state.profiles.mark_training(legacy_profile.id)
        weights = app.state.storage.resolve("profiles", f"{legacy_profile.id}/public")
        weights.mkdir(parents=True, exist_ok=True)
        app.state.profiles.publish(legacy_profile.id, str(weights))

        synthesis_body = {
            "voice_profile_id": legacy_profile.id,
            "text": "safe local text",
            "text_lang": "en",
            "emotion": {"mode": "manual", "label": "happy", "strength": 0.7},
            "consent_confirmed": True,
        }
        queued = client.post("/api/syntheses", json=synthesis_body)
        job_id = queued.json()["job_id"]
        job = client.get(f"/api/syntheses/{job_id}")
        audio = client.get(f"/api/syntheses/{job_id}/audio")
        unknown_synthesis = client.get("/api/syntheses/missing")
        blocked = client.post(
            "/api/syntheses",
            json={**synthesis_body, "text": "bad-word"},
        )
        unknown_safety = client.post(
            "/api/safety/detect-watermark", json={"job_id": "missing"}
        )

        class RejectingSafety:
            def detect_watermark(self, job_id, **kwargs):
                del kwargs
                raise WatermarkVerificationError(job_id)

        app.state.safety_service = RejectingSafety()
        rejected_safety = client.post(
            "/api/safety/detect-watermark", json={"job_id": "published"}
        )
        evaluation = client.post(f"/api/voices/{legacy_profile.id}/evaluate")
        evaluation_job = client.get(f"/api/jobs/{evaluation.json()['job_id']}")

    assert queued.status_code == 202
    assert job.status_code == 200
    assert audio.status_code == 404
    assert unknown_synthesis.status_code == 404
    assert blocked.status_code == 422
    assert blocked.json()["error"]["code"] == "SENSITIVE_TEXT_BLOCKED"
    assert unknown_safety.status_code == 404
    assert rejected_safety.status_code == 409
    assert missing_evaluation.status_code == 404
    assert missing_job.status_code == 404
    assert evaluation.status_code == 202
    assert evaluation_job.status_code == 200


def test_evaluation_and_default_route_services_fail_closed(monkeypatch, tmp_path):
    app = create_app(_settings(tmp_path))
    install_authenticated_user(app, "route-eval-user")
    with TestClient(app) as client:
        profile = app.state.profiles.create(
            "dataset-missing-weights", owner_user_id="route-eval-user"
        )
        app.state.profiles.mark_queued(profile.id)
        app.state.profiles.mark_training(profile.id)
        missing_weights = tmp_path / "not-created" / profile.id
        app.state.profiles.publish(profile.id, str(missing_weights))
        unavailable = client.post(f"/api/voices/{profile.id}/evaluate")
        assert unavailable.status_code == 409
        assert unavailable.json()["error"]["code"] == "VOICE_PROFILE_WEIGHTS_UNAVAILABLE"

        missing_weights.mkdir(parents=True)
        app.state.evaluation_service.texts_path = tmp_path / "missing-evaluation.json"
        contract_error = client.post(f"/api/voices/{profile.id}/evaluate")
        assert contract_error.status_code == 409
        assert contract_error.json()["error"]["code"] == "EVALUATION_CONTRACT_INVALID"

    class EmptyEvaluationService:
        def get(self, job_id, **kwargs):
            del kwargs
            raise KeyError(job_id)

    class EmptySafetyService:
        def detect_watermark(self, job_id, **kwargs):
            del kwargs
            raise KeyError(job_id)

    monkeypatch.setattr(jobs_module, "EvaluationService", EmptyEvaluationService)
    monkeypatch.setattr(safety_module, "SafetyService", EmptySafetyService)
    isolated = FastAPI()
    isolated_engine = create_database_engine("sqlite+pysqlite:///:memory:")
    JobQueue(engine=isolated_engine)
    isolated.state.workspace_service = WorkspaceService(
        engine=isolated_engine,
        storage=LocalStorage(tmp_path / "isolated-data"),
    )
    install_authenticated_user(isolated)
    isolated.include_router(jobs_module.router)
    isolated.include_router(safety_module.router)
    with TestClient(isolated) as client:
        assert client.get("/api/jobs/unknown").status_code == 404
        assert client.post(
            "/api/safety/detect-watermark", json={"job_id": "unknown"}
        ).status_code == 404


def test_synthesis_audio_download_is_verified_and_owner_scoped(tmp_path):
    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    queue = JobQueue(engine=engine)
    storage = LocalStorage(tmp_path / "data")
    job = queue.enqueue(
        JobKind.SYNTHESIZE,
        {"voice_profile_id": "profile-1"},
        owner_user_id="alice",
    )
    assert queue.claim_next("local-worker") is not None
    output = storage.resolve("outputs", "profile-1/verified.wav")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"verified local audio")
    queue.succeed(job.id, {"public_audio_path": "profile-1/verified.wav"})

    app = FastAPI()
    app.state.synthesis_service = SynthesisService(queue=queue, storage=storage)
    install_authenticated_user(app, "alice")
    app.include_router(synthesis_router)
    with TestClient(app) as client:
        response = client.get(f"/api/syntheses/{job.id}/audio")
    assert response.status_code == 200
    assert response.content == output.read_bytes()

    install_authenticated_user(app, "bob")
    with TestClient(app) as foreign_client:
        foreign_response = foreign_client.get(f"/api/syntheses/{job.id}/audio")
    assert foreign_response.status_code == 404
    assert foreign_response.json()["error"]["code"] == "SYNTHESIS_NOT_FOUND"


def test_dataset_service_and_route_reject_unsupported_uploads(tmp_path):
    service = DatasetService(storage=LocalStorage(tmp_path / "data"))
    service.register_dataset(
        "local", effective_seconds=6.5, authorization_confirmed=True
    )
    assert service.preprocess("local") == {
        "dataset_id": "local",
        "effective_seconds": 6.5,
        "status": "ready_for_profile",
        "segment_count": 1,
        "snr_warning_db": 20.0,
        "warning_codes": [],
    }

    app = FastAPI()
    app.dependency_overrides[get_current_user] = lambda: UserRecord(
        id="test-user",
        username="test-user",
        role="user",
        status="active",
        created_at=datetime.now(timezone.utc),
    )
    app.include_router(dataset_router)
    with TestClient(app) as client:
        response = client.post(
            "/api/datasets",
            files={"file": ("sample.txt", b"not audio", "text/plain")},
            data={"consent_confirmed": "true"},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "UNSUPPORTED_AUDIO_FORMAT"


def test_app_injects_configured_snr_warning_threshold_into_voice_routes(tmp_path):
    settings = _settings(tmp_path)
    settings = replace(
        settings, audio=replace(settings.audio, snr_warning_db=30.0)
    )
    app = create_app(settings)

    with TestClient(app) as client:
        assert client.post(
            "/api/auth/register",
            json={"username": "snr-user", "password": "Correct-Horse-42"},
        ).status_code == 201
        assert client.post(
            "/api/auth/login",
            json={"username": "snr-user", "password": "Correct-Horse-42"},
        ).status_code == 200
        owner_id = client.get("/api/auth/me").json()["id"]
        app.state.dataset_service.register_dataset(
            "configured-snr",
            effective_seconds=6.5,
            owner_user_id=owner_id,
            authorization_confirmed=True,
        )

        created = client.post(
            "/api/voices",
            json={"dataset_id": "configured-snr", "display_name": "信噪比音色"},
        )
        listed = client.get("/api/voices")

    assert created.status_code == 201
    assert created.json()["reference_snr_db"] == 27.0
    assert created.json()["quality_warning_codes"] == ["SNR_BELOW_RECOMMENDED"]
    assert listed.status_code == 200
    assert listed.json()["items"][0]["reference_snr_db"] == 27.0
    assert listed.json()["items"][0]["quality_warning_codes"] == [
        "SNR_BELOW_RECOMMENDED"
    ]
