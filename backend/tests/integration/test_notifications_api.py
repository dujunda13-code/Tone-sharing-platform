"""Notifications API: list, unread count, mark-read, recipient scoping."""

from dataclasses import replace

from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.main import create_app


def _settings(tmp_path):
    settings = get_settings()
    return replace(settings, storage=replace(settings.storage, root=tmp_path / "data"))


def _register_and_login(client: TestClient, username: str) -> str:
    assert client.post("/api/auth/register", json={"username": username, "password": "Correct-Horse-42"}).status_code == 201
    assert client.post("/api/auth/login", json={"username": username, "password": "Correct-Horse-42"}).status_code == 200
    return client.get("/api/auth/me").json()["id"]


def test_notification_list_unread_and_mark_read(tmp_path):
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        _register_and_login(client, "bob")
        alice_id = _register_and_login(client, "alice")
        store = app.state.notification_store
        first = store.create(alice_id, "bob-like", type="like", post_id="p1")
        store.create(alice_id, "bob-comment", type="comment", post_id="p1")

        listed = client.get("/api/notifications")
        count = client.get("/api/notifications/unread-count")
        marked = client.post(
            "/api/notifications/mark-read", json={"notification_ids": [first.id]}
        )
        after = client.get("/api/notifications/unread-count")
        marked_all = client.post("/api/notifications/mark-read", json={"all": True})
        empty = client.get("/api/notifications/unread-count")

    assert listed.status_code == 200
    items = listed.json()["items"]
    assert listed.json()["total"] == 2 and len(items) == 2
    assert items[0]["is_read"] is False
    assert count.json()["count"] == 2
    assert marked.json()["updated"] == 1
    assert after.json()["count"] == 1
    assert marked_all.json()["updated"] >= 1
    assert empty.json()["count"] == 0


def test_notifications_are_recipient_scoped(tmp_path):
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        alice_id = _register_and_login(client, "alice")
        bob_id = _register_and_login(client, "bob")
        app.state.notification_store.create(alice_id, bob_id, type="like", post_id="p1")

        bob_list = client.get("/api/notifications")
        bob_count = client.get("/api/notifications/unread-count")
        bob_mark = client.post(
            "/api/notifications/mark-read", json={"all": True}
        )

    assert bob_list.json()["total"] == 0
    assert bob_count.json()["count"] == 0
    assert bob_mark.json()["updated"] == 0
