"""User profile API: read/update, avatar upload/serve, scoping."""

from dataclasses import replace

from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.main import create_app

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 64


def _settings(tmp_path):
    settings = get_settings()
    return replace(settings, storage=replace(settings.storage, root=tmp_path / "data"))


def _register_and_login(client: TestClient, username: str, password: str) -> str:
    assert client.post("/api/auth/register", json={"username": username, "password": password}).status_code == 201
    assert client.post("/api/auth/login", json={"username": username, "password": password}).status_code == 200
    return client.get("/api/auth/me").json()["id"]


def test_profile_defaults_to_username_and_zero_stats(tmp_path):
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        _register_and_login(client, "alice", "Correct-Horse-42")

        response = client.get("/api/users/me/profile")

    assert response.status_code == 200
    body = response.json()
    assert body["display_name"] == "alice"
    assert body["stats"] == {"published": 0, "likes_received": 0}
    assert body["has_avatar"] is False


def test_patch_updates_display_name_and_bio(tmp_path):
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        _register_and_login(client, "alice", "Correct-Horse-42")

        ok = client.patch(
            "/api/users/me/profile", json={"display_name": "小雅", "bio": "爱配音"}
        )
        listed = client.get("/api/users/me/profile")
        too_long = client.patch(
            "/api/users/me/profile", json={"display_name": "x" * 33}
        )

    assert ok.status_code == 200
    assert listed.json()["display_name"] == "小雅"
    assert listed.json()["bio"] == "爱配音"
    assert too_long.status_code == 422


def test_avatar_upload_and_serve(tmp_path):
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        user_id = _register_and_login(client, "alice", "Correct-Horse-42")

        uploaded = client.post(
            "/api/users/me/avatar", files={"file": ("a.png", PNG_BYTES, "image/png")}
        )
        served = client.get(f"/api/avatars/{user_id}")
        missing_user = client.get("/api/avatars/nonexistent")
        profile = client.get("/api/users/me/profile")

    assert uploaded.status_code == 200
    assert uploaded.json()["has_avatar"] is True
    assert served.status_code == 200
    assert served.headers["content-type"].startswith("image/png")
    assert served.content == PNG_BYTES
    assert missing_user.status_code == 404
    assert profile.json()["has_avatar"] is True


def test_avatar_rejects_bad_format_and_size(tmp_path):
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        _register_and_login(client, "alice", "Correct-Horse-42")

        bad_format = client.post(
            "/api/users/me/avatar", files={"file": ("a.txt", b"plain text", "text/plain")}
        )
        too_big = client.post(
            "/api/users/me/avatar",
            files={"file": ("a.png", b"\x89PNG\r\n\x1a\n" + b"0" * (2 * 1024 * 1024 + 1), "image/png")},
        )

    assert bad_format.status_code == 400
    assert bad_format.json()["error"]["code"] == "AVATAR_FORMAT_UNSUPPORTED"
    assert too_big.status_code == 400
    assert too_big.json()["error"]["code"] == "AVATAR_TOO_LARGE"


def test_profile_and_avatar_require_login(tmp_path):
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        for response in [
            client.get("/api/users/me/profile"),
            client.patch("/api/users/me/profile", json={"display_name": "x"}),
            client.post("/api/users/me/avatar", files={"file": ("a.png", PNG_BYTES, "image/png")}),
            client.get("/api/avatars/someone"),
        ]:
            assert response.status_code == 401
