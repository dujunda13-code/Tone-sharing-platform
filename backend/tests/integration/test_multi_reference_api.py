from datetime import datetime, timezone
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
import soundfile as sf
import numpy as np

from backend.app.api.routes.voices import VoiceService, router
from backend.app.api.routes.datasets import DatasetService
from backend.app.core.config import get_settings
from backend.app.db.models import AudioAsset, Dataset, DatasetSegment
from backend.app.db.session import create_database_engine, init_db, session_factory
from backend.app.services.storage import LocalStorage
from backend.app.services.voice_profiles import VoiceProfileStore
from backend.tests.integration.auth_test_helpers import install_authenticated_user


def _seed_ready_dataset(engine, storage, owner_id: str, dataset_id: str, duration: float = 5.0, emotion: str = "neutral"):
    with session_factory(engine)() as session:
        ds = Dataset(
            id=dataset_id,
            owner_user_id=owner_id,
            authorization_confirmed_at=datetime.now(timezone.utc),
            status="ready_for_profile",
            effective_seconds=duration,
        )
        session.add(ds)
        asset = AudioAsset(
            id=f"asset_{dataset_id}",
            dataset_id=dataset_id,
            owner_user_id=owner_id,
            path=f"datasets/{dataset_id}/raw.wav",
        )
        session.add(asset)
        seg = DatasetSegment(
            dataset_id=dataset_id,
            segment_id=f"seg_{dataset_id}",
            owner_user_id=owner_id,
            relative_path=f"datasets/{dataset_id}/segments/seg.wav",
            duration_seconds=duration,
            snr_db=25.0,
            clipping_ratio=0.0,
            split="reference",
            language="zh",
            auto_transcript="这是已确认的参考文本。",
            auto_emotion_label=emotion,
            auto_emotion_confidence=0.9,
            reviewed_at=datetime.now(timezone.utc),
        )
        session.add(seg)
        session.commit()

    seg_path = storage.resolve("datasets", f"{dataset_id}/segments/seg.wav")
    seg_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(seg_path, np.zeros(16_000, dtype=np.float32), 16_000)


@pytest.fixture
def api_client(tmp_path: Path):
    engine = create_database_engine("sqlite://")
    init_db(engine)
    storage = LocalStorage(tmp_path / "data")
    settings = get_settings()

    datasets = DatasetService(storage=storage, engine=engine)
    profiles = VoiceProfileStore(engine=engine)
    service = VoiceService(profiles=profiles, datasets=datasets, settings=settings)

    app = FastAPI()
    app.state.voice_service = service
    install_authenticated_user(app, user_id="user-1")
    app.include_router(router)
    return TestClient(app), engine, storage


def test_add_and_delete_reference_api(api_client):
    client, engine, storage = api_client

    # Register initial ready dataset
    _seed_ready_dataset(engine, storage, "user-1", "d1", duration=5.0)
    res = client.post("/api/voices", json={"dataset_id": "d1", "display_name": "多参考音色"})
    assert res.status_code == 201
    voice = res.json()
    voice_id = voice["id"]
    assert len(voice["references"]) == 1
    assert voice["references"][0]["is_primary"] is True
    primary_ref_id = voice["references"][0]["id"]

    # Register auxiliary dataset
    _seed_ready_dataset(engine, storage, "user-1", "d2", duration=6.0, emotion="happy")
    res2 = client.post(f"/api/voices/{voice_id}/references", json={"dataset_id": "d2"})
    assert res2.status_code == 201
    voice2 = res2.json()
    assert len(voice2["references"]) == 2
    aux_ref_id = [r["id"] for r in voice2["references"] if not r["is_primary"]][0]

    # Deleting primary reference is rejected with 400
    res_del_prim = client.delete(f"/api/voices/{voice_id}/references/{primary_ref_id}")
    assert res_del_prim.status_code == 400
    assert res_del_prim.json()["error"]["code"] == "PRIMARY_REFERENCE_CANNOT_BE_DELETED"

    # Deleting auxiliary reference succeeds with 204
    res_del_aux = client.delete(f"/api/voices/{voice_id}/references/{aux_ref_id}")
    assert res_del_aux.status_code == 204

    # Profile now has 1 reference again
    get_res = client.get(f"/api/voices/{voice_id}")
    assert get_res.status_code == 200
    assert len(get_res.json()["references"]) == 1


def test_add_reference_exceeding_five_returns_409(api_client):
    client, engine, storage = api_client

    _seed_ready_dataset(engine, storage, "user-1", "d1", duration=5.0)
    res = client.post("/api/voices", json={"dataset_id": "d1", "display_name": "上限测试音色"})
    assert res.status_code == 201
    voice_id = res.json()["id"]

    # Add 4 auxiliary references to reach 5
    for i in range(2, 6):
        _seed_ready_dataset(engine, storage, "user-1", f"d_{i}", duration=5.0)
        res_i = client.post(f"/api/voices/{voice_id}/references", json={"dataset_id": f"d_{i}"})
        assert res_i.status_code == 201

    # Add 6th reference -> 409 REFERENCE_LIMIT_EXCEEDED
    _seed_ready_dataset(engine, storage, "user-1", "d_overflow", duration=5.0)
    res_overflow = client.post(f"/api/voices/{voice_id}/references", json={"dataset_id": "d_overflow"})
    assert res_overflow.status_code == 409
    assert res_overflow.json()["error"]["code"] == "REFERENCE_LIMIT_EXCEEDED"

