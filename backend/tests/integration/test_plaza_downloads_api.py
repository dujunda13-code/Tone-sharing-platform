"""Plaza downloads, preview, import fork and synthesis gate relaxation."""

import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.main import create_app

SEGMENT_BYTES = b"RIFF-segment-wav-bytes"


def _settings(tmp_path):
    settings = get_settings()
    return replace(settings, storage=replace(settings.storage, root=tmp_path / "data"))


def _register_and_login(client: TestClient, username: str) -> str:
    assert client.post("/api/auth/register", json={"username": username, "password": "Correct-Horse-42"}).status_code == 201
    assert client.post("/api/auth/login", json={"username": username, "password": "Correct-Horse-42"}).status_code == 200
    return client.get("/api/auth/me").json()["id"]


def _login(client: TestClient, username: str, password: str = "Correct-Horse-42") -> str:
    assert client.post("/api/auth/login", json={"username": username, "password": password}).status_code == 200
    return client.get("/api/auth/me").json()["id"]


def _seed_published_post(app, settings, owner_id: str) -> str:
    # register_dataset already persists the reviewed seg_0001 row (and its asset);
    # only the audio file itself is missing, so mirror it on disk here.
    app.state.dataset_service.register_dataset(
        "plaza-ds", effective_seconds=6.0, owner_user_id=owner_id, authorization_confirmed=True
    )
    segment_path = Path(settings.storage.root) / "datasets" / "plaza-ds" / "segments" / "seg_0001.wav"
    segment_path.parent.mkdir(parents=True, exist_ok=True)
    segment_path.write_bytes(SEGMENT_BYTES)
    profile = app.state.profiles.create_zero_shot(
        dataset_id="plaza-ds",
        owner_user_id=owner_id,
        reference_asset_id="asset-plaza",
        reference_segment_id="seg_0001",
        prompt_text="确认文本。",
        prompt_language="zh",
        emotion_label="neutral",
        emotion_confidence=0.9,
        base_model_id="gpt-sovits-v2proplus-official",
        display_name="可下载音色",
    )
    post = app.state.plaza_service.publish(owner_id, voice_profile_id=profile.id)
    return post.id, profile.id


def test_download_package_reference_and_preview(tmp_path):
    settings = _settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        alice_id = _register_and_login(client, "alice")
        post_id, _ = _seed_published_post(app, settings, alice_id)

        package = client.get(f"/api/plaza/posts/{post_id}/download")
        reference = client.get(f"/api/plaza/posts/{post_id}/download/reference")
        preview = client.get(f"/api/plaza/posts/{post_id}/preview")
        missing = client.get("/api/plaza/posts/missing/download")

    assert package.status_code == 200
    assert package.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(package.content)) as archive:
        names = archive.namelist()
        assert "metadata.json" in names
        assert archive.read("references/primary-seg_0001.wav") == SEGMENT_BYTES
        metadata = json.loads(archive.read("metadata.json"))
    assert metadata["voice"]["display_name"] == "可下载音色"
    assert reference.status_code == 200
    assert reference.headers["content-type"].startswith("audio/wav")
    assert reference.content == SEGMENT_BYTES
    assert "attachment" in reference.headers["content-disposition"]
    assert preview.status_code == 200
    assert preview.content == SEGMENT_BYTES
    assert "attachment" not in preview.headers.get("content-disposition", "")
    assert missing.status_code == 404


def test_import_fork_requires_confirmation_then_succeeds(tmp_path):
    settings = _settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        alice_id = _register_and_login(client, "alice")
        post_id, _ = _seed_published_post(app, settings, alice_id)
        client.post("/api/auth/logout")
        _register_and_login(client, "bob")

        rejected = client.post(
            f"/api/plaza/posts/{post_id}/import", json={"authorization_confirmed": False}
        )
        imported = client.post(
            f"/api/plaza/posts/{post_id}/import", json={"authorization_confirmed": True}
        )
        bob_voices = client.get("/api/voices")

    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "AUTHORIZATION_REQUIRED"
    assert imported.status_code == 201
    forked_id = imported.json()["voice_profile_id"]
    assert imported.json()["display_name"] == "可下载音色"
    assert any(v["id"] == forked_id for v in bob_voices.json()["items"])


def test_synthesis_gate_relaxed_for_published_plaza_profiles(tmp_path):
    settings = _settings(tmp_path)
    app = create_app(settings)
    payload = {
        "text": "safe local text",
        "text_lang": "zh",
        "emotion": {"mode": "manual", "label": "happy", "strength": 0.7},
        "consent_confirmed": True,
    }
    with TestClient(app) as client:
        alice_id = _register_and_login(client, "alice")
        post_id, profile_id = _seed_published_post(app, settings, alice_id)

        unpublished = app.state.profiles.create_zero_shot(
            dataset_id="plaza-ds",
            owner_user_id=alice_id,
            reference_asset_id="asset-2",
            reference_segment_id="seg_0001",
            prompt_text="确认文本。",
            prompt_language="zh",
            emotion_label="neutral",
            emotion_confidence=0.9,
            base_model_id="gpt-sovits-v2proplus-official",
            display_name="未发布音色",
        )
        client.post("/api/auth/logout")
        _register_and_login(client, "bob")

        published_ok = client.post(
            "/api/syntheses", json={"voice_profile_id": profile_id, **payload}
        )
        unpublished_rejected = client.post(
            "/api/syntheses", json={"voice_profile_id": unpublished.id, **payload}
        )

        # Deleting the post re-tightens the gate.
        client.post("/api/auth/logout")
        _login(client, "alice")
        client.delete(f"/api/plaza/posts/{post_id}")
        client.post("/api/auth/logout")
        _login(client, "bob")
        after_delete = client.post(
            "/api/syntheses", json={"voice_profile_id": profile_id, **payload}
        )

    assert published_ok.status_code == 202
    assert unpublished_rejected.status_code == 404
    assert unpublished_rejected.json()["error"]["code"] == "VOICE_PROFILE_NOT_FOUND"
    assert after_delete.status_code == 404
