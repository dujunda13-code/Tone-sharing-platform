"""Plaza API: publish → browse → like → comment → notify → delete cascade."""

from dataclasses import replace

from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.main import create_app
from backend.app.services.sensitive_filter import SensitiveFilter


def _settings(tmp_path):
    settings = get_settings()
    return replace(settings, storage=replace(settings.storage, root=tmp_path / "data"))


def _register_and_login(client: TestClient, username: str, password="Correct-Horse-42") -> str:
    assert client.post("/api/auth/register", json={"username": username, "password": password}).status_code == 201
    assert client.post("/api/auth/login", json={"username": username, "password": password}).status_code == 200
    return client.get("/api/auth/me").json()["id"]


def _login(client: TestClient, username: str, password: str = "Correct-Horse-42") -> str:
    assert client.post("/api/auth/login", json={"username": username, "password": password}).status_code == 200
    return client.get("/api/auth/me").json()["id"]


def _seed_ready_profile(app, owner_id: str, name: str = "广场音色") -> str:
    app.state.dataset_service.register_dataset(
        f"ds-{name}", effective_seconds=6.0, owner_user_id=owner_id, authorization_confirmed=True
    )
    profile = app.state.profiles.create_zero_shot(
        dataset_id=f"ds-{name}",
        owner_user_id=owner_id,
        reference_asset_id=f"asset-{name}",
        reference_segment_id="seg_0001",
        prompt_text="确认文本。",
        prompt_language="zh",
        emotion_label="neutral",
        emotion_confidence=0.9,
        base_model_id="gpt-sovits-v2proplus-official",
        display_name=name,
    )
    return profile.id


def test_full_plaza_flow_with_two_users(tmp_path):
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        alice_id = _register_and_login(client, "alice")
        profile_id = _seed_ready_profile(app, alice_id)

        published = client.post(
            "/api/plaza/posts",
            json={"voice_profile_id": profile_id, "description": "我的第一个音色"},
        )
        duplicate = client.post("/api/plaza/posts", json={"voice_profile_id": profile_id})

        client.post("/api/auth/logout")
        _register_and_login(client, "bob")
        listed = client.get("/api/plaza/posts")
        item = listed.json()["items"][0]
        post_id = item["id"]

        liked = client.post(f"/api/plaza/posts/{post_id}/likes")
        liked_again = client.post(f"/api/plaza/posts/{post_id}/likes")
        commented = client.post(
            f"/api/plaza/posts/{post_id}/comments", json={"content": "非常好听"}
        )
        favorited = client.post(f"/api/plaza/posts/{post_id}/favorites")
        unfavorited = client.delete(f"/api/plaza/posts/{post_id}/favorites")

        client.post("/api/auth/logout")
        _login(client, "alice")
        unread = client.get("/api/notifications/unread-count")
        alice_list = client.get("/api/plaza/posts")
        alice_item = alice_list.json()["items"][0]

        deleted = client.delete(f"/api/plaza/posts/{post_id}")
        gone = client.get("/api/plaza/posts")
        bob_unread_after_delete = client.get("/api/notifications/unread-count")

    assert published.status_code == 201
    assert published.json()["voice"]["display_name"] == "广场音色"
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "PLAZA_POST_ALREADY_PUBLISHED"
    assert listed.status_code == 200 and listed.json()["total"] == 1
    assert item["like_count"] == 0 and item["liked_by_me"] is False
    assert item["author"]["display_name"] == "alice"
    assert liked.status_code == 200
    assert liked_again.status_code == 409
    assert commented.status_code == 201
    assert favorited.status_code == 200
    assert unfavorited.status_code == 200
    assert unread.json()["count"] == 2  # like + comment
    assert alice_item["like_count"] == 1 and alice_item["comment_count"] == 1
    assert alice_item["liked_by_me"] is False
    assert deleted.status_code == 200
    assert gone.json()["total"] == 0
    assert bob_unread_after_delete.json()["count"] == 0  # cascade removed


def test_comment_validation_and_sensitive_block(tmp_path):
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        alice_id = _register_and_login(client, "alice")
        profile_id = _seed_ready_profile(app, alice_id)
        post_id = client.post("/api/plaza/posts", json={"voice_profile_id": profile_id}).json()["id"]

        empty = client.post(f"/api/plaza/posts/{post_id}/comments", json={"content": "   "})
        too_long = client.post(
            f"/api/plaza/posts/{post_id}/comments", json={"content": "x" * 501}
        )
        app.state.plaza_service.sensitive_filter = SensitiveFilter(["违禁词"])
        sensitive = client.post(
            f"/api/plaza/posts/{post_id}/comments", json={"content": "含违禁词的评论"}
        )
        comments = client.get(f"/api/plaza/posts/{post_id}/comments")
        my_comments = client.get("/api/plaza/comments")

    assert empty.status_code == 400
    assert empty.json()["error"]["code"] == "PLAZA_COMMENT_INVALID"
    assert too_long.status_code == 400
    assert sensitive.status_code == 422
    assert sensitive.json()["error"]["code"] == "SENSITIVE_COMMENT_BLOCKED"
    assert comments.json()["total"] == 0
    assert my_comments.json()["total"] == 0


def test_publish_gates_and_delete_permissions(tmp_path):
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        alice_id = _register_and_login(client, "alice")
        profile_id = _seed_ready_profile(app, alice_id)
        post_id = client.post("/api/plaza/posts", json={"voice_profile_id": profile_id}).json()["id"]

        client.post("/api/auth/logout")
        _register_and_login(client, "bob")
        bob_publish = client.post("/api/plaza/posts", json={"voice_profile_id": profile_id})
        bob_delete = client.delete(f"/api/plaza/posts/{post_id}")
        missing = client.get("/api/plaza/posts/missing")
        missing_delete = client.delete("/api/plaza/posts/missing")

        client.post("/api/auth/logout")
        _login(client, "alice")
        alice_delete = client.delete(f"/api/plaza/posts/{post_id}")

    assert bob_publish.status_code == 404  # foreign profile
    assert bob_delete.status_code == 403
    assert bob_delete.json()["error"]["code"] == "PLAZA_NOT_POST_AUTHOR"
    assert missing.status_code == 404
    assert missing_delete.status_code == 404
    assert alice_delete.status_code == 200


def test_plaza_requires_login(tmp_path):
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        for response in [
            client.get("/api/plaza/posts"),
            client.post("/api/plaza/posts", json={"voice_profile_id": "x"}),
            client.get("/api/plaza/comments"),
        ]:
            assert response.status_code == 401


def test_list_filters_and_search(tmp_path):
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        alice_id = _register_and_login(client, "alice")
        p_alice = _seed_ready_profile(app, alice_id, name="清亮女声")
        post_a = client.post(
            "/api/plaza/posts", json={"voice_profile_id": p_alice, "description": "温柔"}
        ).json()["id"]

        bob_id = _register_and_login(client, "bob")
        p_bob = _seed_ready_profile(app, bob_id, name="低沉男声")
        client.post("/api/plaza/posts", json={"voice_profile_id": p_bob})

        mine = client.get("/api/plaza/posts", params={"mine": "true"})
        search = client.get("/api/plaza/posts", params={"q": "清亮"})
        search_desc = client.get("/api/plaza/posts", params={"q": "温柔"})

        client.post(f"/api/plaza/posts/{post_a}/likes")  # bob likes alice's
        sort_likes = client.get("/api/plaza/posts", params={"sort": "likes"})

    assert mine.json()["total"] == 1 and mine.json()["items"][0]["voice"]["display_name"] == "低沉男声"
    assert search.json()["total"] == 1
    assert search_desc.json()["total"] == 1
    assert sort_likes.json()["items"][0]["voice"]["display_name"] == "清亮女声"
