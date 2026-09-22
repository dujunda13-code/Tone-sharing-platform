from dataclasses import replace

from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.main import create_app
from backend.app.schemas.common import JobKind
from backend.app.schemas.voice import VoiceProfileCreate


def _settings(tmp_path):
    settings = get_settings()
    return replace(settings, storage=replace(settings.storage, root=tmp_path / "data"))


def _login(client: TestClient, username: str, password: str) -> str:
    assert client.post(
        "/api/auth/register", json={"username": username, "password": password}
    ).status_code == 201
    assert client.post(
        "/api/auth/login", json={"username": username, "password": password}
    ).status_code == 200
    return client.get("/api/auth/me").json()["id"]


def test_admin_can_change_other_user_status_and_records_sanitized_audit_event(tmp_path):
    """Would fail if admin controls allow self-lockout, preserve disabled sessions, or leak secrets."""
    app = create_app(_settings(tmp_path))
    with TestClient(app) as admin, TestClient(app) as alice:
        assert admin.post(
            "/api/admin/bootstrap",
            json={"username": "admin", "password": "Admin-Horse-42"},
        ).status_code == 201
        assert admin.post(
            "/api/auth/login",
            json={"username": "admin", "password": "Admin-Horse-42"},
        ).status_code == 200
        admin_id = admin.get("/api/auth/me").json()["id"]
        alice_id = _login(alice, "alice", "Correct-Horse-42")
        app.state.dataset_service.register_dataset(
            "alice-dataset",
            effective_seconds=6.5,
            owner_user_id=alice_id,
            authorization_confirmed=True,
        )
        voice = app.state.voice_service.create(
            VoiceProfileCreate(dataset_id="alice-dataset", display_name="Alice 音色"),
            owner_user_id=alice_id
        )
        app.state.queue.enqueue(
            JobKind.TRAIN,
            {"profile_id": voice.id},
            owner_user_id=alice_id,
        )

        forbidden = alice.patch(
            f"/api/admin/users/{alice_id}/status", json={"status": "disabled"}
        )
        blocked_list = alice.get("/api/admin/users")
        blocked_audit = alice.get("/api/admin/audit-events?limit=1")
        assert forbidden.status_code == 403
        assert blocked_list.status_code == 403
        assert blocked_audit.status_code == 403
        self_disable = admin.patch(
            f"/api/admin/users/{admin_id}/status", json={"status": "disabled"}
        )
        assert self_disable.status_code == 409
        changed = admin.patch(
            f"/api/admin/users/{alice_id}/status", json={"status": "disabled"}
        )
        assert changed.status_code == 200
        disabled_me = alice.get("/api/auth/me")
        assert disabled_me.status_code == 401
        restored = admin.patch(
            f"/api/admin/users/{alice_id}/status", json={"status": "active"}
        )
        assert restored.status_code == 200
        users = admin.get("/api/admin/users")
        assert users.status_code == 200
        first_response = admin.get("/api/admin/audit-events?limit=1")
        assert first_response.status_code == 200
        first_page = first_response.json()
        assert first_page["next_before_id"] is not None
        second_page = admin.get(
            f"/api/admin/audit-events?limit=1&before_id={first_page['next_before_id']}"
        ).json()

    assert self_disable.status_code == 409
    assert self_disable.json()["error"]["code"] == "ADMIN_SELF_STATUS_CHANGE_FORBIDDEN"
    alice_summary = next(item for item in users.json()["items"] if item["id"] == alice_id)
    assert alice_summary["status"] == "active"
    assert {
        "id",
        "username",
        "role",
        "status",
        "created_at",
        "dataset_count",
        "voice_count",
        "job_count",
    } == set(alice_summary)
    assert (
        alice_summary["dataset_count"],
        alice_summary["voice_count"],
        alice_summary["job_count"],
    ) == (1, 1, 1)
    assert first_page["items"][0]["event_type"] == "admin.user_status_changed"
    assert first_page["next_before_id"] is not None
    assert second_page["items"][0]["id"] < first_page["items"][0]["id"]
    assert "password" not in str(first_page).lower()
    assert "token" not in str(first_page).lower()
    assert "hash" not in str(first_page).lower()


def test_local_admin_endpoints_allow_patch_from_the_local_frontend(tmp_path):
    """Would fail if CORS omits PATCH for the only permitted local frontend origin."""
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        response = client.options(
            "/api/admin/users/local-user/status",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "PATCH",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
    assert "PATCH" in response.headers["access-control-allow-methods"]
