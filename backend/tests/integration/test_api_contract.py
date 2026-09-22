from dataclasses import replace

from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.main import create_app


def _settings(tmp_path):
    settings = get_settings()
    return replace(settings, storage=replace(settings.storage, root=tmp_path / "data"))


def test_openapi_exposes_local_and_emotion_api_language_values(tmp_path):
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()

    language = schema["components"]["schemas"]["SynthesisCreate"]
    assert all(code in str(language) for code in ("zh", "en", "ja", "es", "ar"))


def test_request_validation_uses_unified_error_shape(tmp_path):
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        assert client.post(
            "/api/auth/register",
            json={"username": "contract-user", "password": "Correct-Horse-42"},
        ).status_code == 201
        assert client.post(
            "/api/auth/login",
            json={"username": "contract-user", "password": "Correct-Horse-42"},
        ).status_code == 200
        response = client.post("/api/syntheses", json={})

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "REQUEST_VALIDATION_ERROR"
    assert body["error"]["request_id"] == response.headers["X-Request-ID"]
    assert body["error"]["details"]


def test_cors_is_limited_to_the_two_local_frontend_origins(tmp_path):
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        allowed = client.options(
            "/api/health/live",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        denied = client.get(
            "/api/health/live",
            headers={"Origin": "https://example.invalid"},
        )

    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-origin" not in denied.headers


def test_http_and_unexpected_errors_use_the_same_envelope(tmp_path):
    app = create_app(_settings(tmp_path))

    @app.get("/api/test-http-error")
    def test_http_error():
        raise HTTPException(
            status_code=409,
            detail={"code": "CONTRACT_ERROR", "message": "contract failed", "details": {"x": 1}},
        )

    @app.get("/api/test-unexpected-error")
    def test_unexpected_error():
        raise RuntimeError("private local detail")

    with TestClient(app, raise_server_exceptions=False) as client:
        http_error = client.get("/api/test-http-error")
        unexpected = client.get("/api/test-unexpected-error")

    assert http_error.status_code == 409
    assert http_error.json()["error"]["code"] == "CONTRACT_ERROR"
    assert http_error.json()["error"]["details"] == {"x": 1}
    assert unexpected.status_code == 500
    assert unexpected.json()["error"]["code"] == "INTERNAL_SERVER_ERROR"
    assert "private local detail" not in unexpected.text
