from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.routes.voices import EmotionOverrideStore, router
from backend.tests.integration.auth_test_helpers import install_authenticated_user


def test_manual_override_requires_login():
    app = FastAPI()
    store = EmotionOverrideStore()
    store.register_segment("d1", "seg-1", auto_label="sad", auto_confidence=0.81)
    app.state.emotion_override_store = store
    app.include_router(router)

    response = TestClient(app).put(
        "/api/datasets/d1/segments/seg-1/emotion",
        json={"label": "happy", "reason": "speaker confirmed"},
    )

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "AUTH_REQUIRED"


def test_manual_override_keeps_auto_result():
    app = FastAPI()
    store = EmotionOverrideStore()
    store.register_segment(
        "d1", "seg-1", auto_label="sad", auto_confidence=0.81, owner_user_id="test-user"
    )
    app.state.emotion_override_store = store
    install_authenticated_user(app)
    app.include_router(router)

    response = TestClient(app).put(
        "/api/datasets/d1/segments/seg-1/emotion",
        json={"label": "happy", "reason": "speaker confirmed"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["auto_label"] == "sad"
    assert body["auto_confidence"] == 0.81
    assert body["effective_label"] == "happy"
    assert body["label_source"] == "manual"
    assert body["reason"] == "speaker confirmed"
