from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.routes.datasets import DatasetService, router
from backend.app.services.storage import LocalStorage
from backend.tests.integration.auth_test_helpers import install_authenticated_user


def test_preprocess_rejects_reference_duration_out_of_range(tmp_path):
    app = FastAPI()
    service = DatasetService(storage=LocalStorage(tmp_path / "data"))
    service.register_dataset(
        "d1",
        effective_seconds=2.99,
        owner_user_id="test-user",
        authorization_confirmed=True,
    )
    app.state.dataset_service = service
    install_authenticated_user(app)
    app.include_router(router)

    response = TestClient(app).post("/api/datasets/d1/preprocess")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "REFERENCE_DURATION_OUT_OF_RANGE"


def test_confirm_rejects_reference_duration_out_of_range(tmp_path):
    app = FastAPI()
    service = DatasetService(storage=LocalStorage(tmp_path / "data"))
    service.register_dataset(
        "d1",
        effective_seconds=10.01,
        owner_user_id="test-user",
        authorization_confirmed=True,
    )
    app.state.dataset_service = service
    install_authenticated_user(app)
    app.include_router(router)

    response = TestClient(app).post(
        "/api/datasets/d1/segments/confirm", json={"reason": "批量确认"}
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "REFERENCE_DURATION_OUT_OF_RANGE"
