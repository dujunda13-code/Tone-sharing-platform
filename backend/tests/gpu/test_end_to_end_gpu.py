from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select
import soundfile as sf

from backend.app.api.routes.datasets import DatasetService
from backend.app.api.routes.voices import VoiceService
from backend.app.core.config import get_settings
from backend.app.db.models import Job
from backend.app.db.session import session_factory
from backend.app.main import create_app
from backend.app.services.emotion import EmotionAnalyzer
from backend.app.services.gpt_sovits import GPTSoVITSAdapter
from backend.app.services.local_worker import assemble_local_worker
from backend.app.services.storage import LocalStorage
from backend.app.services.transcription import LocalFunASRTranscriber


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.gpu
def test_authorized_voice_end_to_end_requires_real_local_prerequisites(tmp_path: Path):
    """Run the real production assembly and authenticated zero-shot path."""
    reference_text = os.environ.get("TIMBRE_AUTHORIZED_REFERENCE", "").strip()
    if not reference_text:
        pytest.fail("TIMBRE_AUTHORIZED_REFERENCE is required for real zero-shot acceptance")
    reference = Path(reference_text).resolve()
    if not reference.is_file():
        pytest.fail("TIMBRE_AUTHORIZED_REFERENCE does not point to a local authorized reference")
    if reference.suffix.casefold() not in {".wav", ".flac", ".mp3"}:
        pytest.fail("authorized reference must be WAV, FLAC or MP3")

    raw_duration = float(sf.info(reference).duration)
    if not 3.0 <= raw_duration <= 10.0:
        pytest.fail(
            "authorized reference file duration must be between 3 and 10 seconds; "
            f"got {raw_duration:.3f}"
        )

    runtime = GPTSoVITSAdapter().probe()
    if not runtime.available:
        pytest.fail(f"real GPU E2E prerequisites are unavailable: {runtime.reason}")
    assert runtime.tag == "20250606v2pro"
    assert runtime.device == "cuda:0"
    assert runtime.available

    settings = get_settings(ROOT / "config" / "app.yaml")
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    test_settings = replace(settings, storage=replace(settings.storage, root=data_root))
    app = create_app(test_settings)
    storage = LocalStorage(data_root)
    dataset_service = DatasetService(
        storage=storage,
        engine=app.state.engine,
        transcriber=LocalFunASRTranscriber(
            model_dir=ROOT / "models" / "funasr" / "paraformer-zh",
            models_root=ROOT / "models",
        ),
        emotion_analyzer=EmotionAnalyzer(
            embedding_root=data_root / "features" / "emotion"
        ),
    )
    app.state.dataset_service = dataset_service
    app.state.voice_service = VoiceService(
        profiles=app.state.profiles,
        datasets=dataset_service,
        settings=test_settings,
    )

    reference_bytes = reference.read_bytes()
    with TestClient(app) as client:
        registered = client.post(
            "/api/auth/register",
            json={"username": "real-zero-shot", "password": "Correct-Horse-42"},
        )
        assert registered.status_code == 201, registered.text
        login = client.post(
            "/api/auth/login",
            json={"username": "real-zero-shot", "password": "Correct-Horse-42"},
        )
        assert login.status_code == 200, login.text
        owner_id = login.json()["id"]

        upload_response = client.post(
            "/api/datasets",
            files={"file": (reference.name, reference_bytes, "audio/wav")},
            data={"consent_confirmed": "true"},
        )
        assert upload_response.status_code == 200, upload_response.text
        dataset_id = upload_response.json()["dataset_id"]
        preprocessed = client.post(f"/api/datasets/{dataset_id}/preprocess")
        assert preprocessed.status_code == 200, preprocessed.text
        assert 3.0 <= float(preprocessed.json()["effective_seconds"]) <= 10.0
        confirmed = client.post(
            f"/api/datasets/{dataset_id}/segments/confirm",
            json={"reason": "真实授权零样本验收人工确认"},
        )
        assert confirmed.status_code == 200, confirmed.text
        assert 3.0 <= float(confirmed.json()["reviewed_effective_seconds"]) <= 10.0

        created = client.post(
            "/api/voices",
            json={"dataset_id": dataset_id, "display_name": "验收音色"},
        )
        assert created.status_code == 201, created.text
        profile_id = created.json()["id"]
        profile = app.state.profiles.get(profile_id, owner_user_id=owner_id)
        assert profile.status == "ready"
        assert profile.mode == "zero_shot"
        assert profile.public_weight_dir is None

        queued = client.post(
            "/api/syntheses",
            json={
                "voice_profile_id": profile_id,
                "text": "这是本地基础模型零样本合成验收文本。",
                "text_lang": "zh",
                "emotion": {"mode": "auto"},
                "consent_confirmed": True,
            },
        )
        assert queued.status_code == 202, queued.text
        job_id = queued.json()["job_id"]

        worker = assemble_local_worker(
            root=ROOT,
            python_executable=Path(sys.executable),
            database_url=f"sqlite+pysqlite:///{(data_root / 'app.db').as_posix()}",
            storage_root=data_root,
        )
        completed = worker.run_one()

        assert completed is not None
        assert completed.status.value == "succeeded", completed.error_message
        assert completed.result is not None
        result = completed.result
        assert float(result["speaker_similarity"]) >= 0.90
        assert float(result["watermark_probability"]) >= 0.80
        fingerprint = result["fingerprint"]
        assert isinstance(fingerprint, dict)
        assert "high_band_energy_ratio" in fingerprint
        assert "spectral_flatness" in fingerprint
        output = storage.resolve("outputs", str(result["public_audio_path"]))
        assert output.is_file()
        assert output.stat().st_size > 0

        download = client.get(f"/api/syntheses/{job_id}/audio")
        assert download.status_code == 200, download.text
        assert download.headers["content-type"].startswith("audio/wav")
        assert download.content == output.read_bytes()

        with TestClient(app) as foreign_client:
            foreign_register = foreign_client.post(
                "/api/auth/register",
                json={"username": "foreign-user", "password": "Correct-Horse-42"},
            )
            assert foreign_register.status_code == 201, foreign_register.text
            foreign_login = foreign_client.post(
                "/api/auth/login",
                json={"username": "foreign-user", "password": "Correct-Horse-42"},
            )
            assert foreign_login.status_code == 200, foreign_login.text
            foreign_download = foreign_client.get(f"/api/syntheses/{job_id}/audio")
        assert foreign_download.status_code == 404
        assert foreign_download.json()["error"]["code"] == "SYNTHESIS_NOT_FOUND"

    with session_factory(app.state.engine)() as session:
        assert session.scalar(select(Job).where(Job.kind == "train")) is None
    assert not (data_root / "features" / dataset_id).exists()
    assert not (data_root / "profiles").exists()
    assert not any(
        path.suffix.casefold() in {".ckpt", ".pth", ".pt", ".safetensors"}
        or path.name.casefold().startswith("checkpoint")
        for path in data_root.rglob("*")
        if path.is_file()
    )
