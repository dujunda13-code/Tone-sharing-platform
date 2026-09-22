from dataclasses import replace

from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.main import create_app


def _settings(tmp_path):
    settings = get_settings()
    return replace(settings, storage=replace(settings.storage, root=tmp_path / "data"))


def test_register_login_me_and_logout(tmp_path):
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        created = client.post(
            "/api/auth/register",
            json={"username": "alice", "password": "Correct-Horse-42"},
        )
        logged_in = client.post(
            "/api/auth/login",
            json={"username": "alice", "password": "Correct-Horse-42"},
        )
        current = client.get("/api/auth/me")
        logged_out = client.post("/api/auth/logout")
        after_logout = client.get("/api/auth/me")

    assert created.status_code == 201
    assert logged_in.status_code == 200
    assert current.status_code == 200
    assert current.json()["username"] == "alice"
    assert logged_out.status_code == 204
    assert after_logout.status_code == 401


def test_eight_character_password_can_register_and_login(tmp_path):
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        created = client.post(
            "/api/auth/register",
            json={"username": "eight-user", "password": "eight888"},
        )
        logged_in = client.post(
            "/api/auth/login",
            json={"username": "eight-user", "password": "eight888"},
        )

    assert created.status_code == 201
    assert logged_in.status_code == 200
    assert logged_in.json()["username"] == "eight-user"


def test_admin_bootstrap_is_one_time(tmp_path):
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        first = client.post(
            "/api/admin/bootstrap",
            json={"username": "admin", "password": "Admin-Horse-42"},
        )
        second = client.post(
            "/api/admin/bootstrap",
            json={"username": "other", "password": "Other-Horse-42"},
        )

    assert first.status_code == 201
    assert first.json()["role"] == "admin"
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "ADMIN_ALREADY_INITIALIZED"


def test_protected_training_route_requires_login(tmp_path):
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/api/voices/profile-1/train",
            json={"consent_confirmed": True},
        )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "AUTH_REQUIRED"


def test_health_remains_anonymous(tmp_path):
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        response = client.get("/api/health/live")

    assert response.status_code == 200


def test_dataset_upload_requires_login(tmp_path):
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/api/datasets",
            files={"file": ("sample.wav", b"local sample", "audio/wav")},
        )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "AUTH_REQUIRED"


def test_user_cannot_read_another_users_voice_profile(tmp_path):
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        client.post(
            "/api/auth/register",
            json={"username": "alice", "password": "Correct-Horse-42"},
        )
        client.post(
            "/api/auth/login",
            json={"username": "alice", "password": "Correct-Horse-42"},
        )
        alice_id = client.get("/api/auth/me").json()["id"]
        app.state.dataset_service.register_dataset(
            "alice-dataset",
            effective_seconds=6.5,
            owner_user_id=alice_id,
            authorization_confirmed=True,
        )
        created = client.post(
            "/api/voices",
            json={"dataset_id": "alice-dataset", "display_name": "认证测试音色"},
        )
        assert created.status_code == 201
        profile_id = created.json()["id"]
        client.post("/api/auth/logout")
        client.post(
            "/api/auth/register",
            json={"username": "bob", "password": "Correct-Horse-43"},
        )
        client.post(
            "/api/auth/login",
            json={"username": "bob", "password": "Correct-Horse-43"},
        )
        response = client.get(f"/api/voices/{profile_id}")

    assert response.status_code == 404


def test_jobs_synthesis_and_safety_require_login(tmp_path):
    app = create_app(_settings(tmp_path))
    synthesis_body = {
        "voice_profile_id": "profile-1",
        "text": "safe local text",
        "text_lang": "en",
        "emotion": {"mode": "manual", "label": "happy", "strength": 0.7},
        "consent_confirmed": True,
    }

    with TestClient(app) as client:
        synthesis = client.post("/api/syntheses", json=synthesis_body)
        job = client.get("/api/jobs/missing")
        safety = client.post("/api/safety/detect-watermark", json={"job_id": "missing"})

    assert synthesis.status_code == 401
    assert job.status_code == 401
    assert safety.status_code == 401
