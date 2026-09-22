from dataclasses import replace

from fastapi.testclient import TestClient
from sqlalchemy import event

from backend.app.core.config import get_settings
from backend.app.main import create_app
from backend.app.schemas.common import JobKind
from backend.app.schemas.voice import VoiceProfileCreate


def test_synthesis_detail_exposes_live_candidate_progress(tmp_path):
    """The running synthesis summary must surface the worker's candidate progress message."""
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        alice_id = _login(client, "alice", "Correct-Horse-42")
        app.state.dataset_service.register_dataset(
            "alice-dataset",
            effective_seconds=6.5,
            owner_user_id=alice_id,
            authorization_confirmed=True,
        )
        voice = app.state.voice_service.create(
            VoiceProfileCreate(dataset_id="alice-dataset", display_name="爱丽丝音色"),
            owner_user_id=alice_id,
        )
        job = app.state.queue.enqueue(
            JobKind.SYNTHESIZE,
            {"voice_profile_id": voice.id, "text": "进度展示文本", "text_lang": "zh"},
            owner_user_id=alice_id,
        )
        app.state.queue.claim_next("gpu-0")
        app.state.queue.update_progress(job.id, "正在生成候选 1/3...")

        running = client.get(f"/api/syntheses/{job.id}")

        assert running.status_code == 200
        assert running.json()["progress_message"] == "正在生成候选 1/3..."

        app.state.queue.succeed(job.id, {"watermark_probability": 0.95})
        finished = client.get(f"/api/syntheses/{job.id}")

        assert finished.json()["progress_message"] is None


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


def test_workspace_lists_only_current_users_public_summaries(tmp_path):
    """Would fail if a query leaks another user's resources or raw job payload."""
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        alice_id = _login(client, "alice", "Correct-Horse-42")
        app.state.dataset_service.register_dataset(
            "alice-dataset",
            effective_seconds=6.5,
            owner_user_id=alice_id,
            authorization_confirmed=True,
        )
        app.state.dataset_service.register_dataset(
            "too-short",
            effective_seconds=2.5,
            owner_user_id=alice_id,
            authorization_confirmed=True,
        )
        app.state.dataset_service.register_dataset(
            "not-authorized",
            effective_seconds=600,
            owner_user_id=alice_id,
            authorization_confirmed=False,
        )
        alice_voice = app.state.voice_service.create(
            VoiceProfileCreate(dataset_id="alice-dataset", display_name="爱丽丝音色"),
            owner_user_id=alice_id,
        )
        train_job = app.state.queue.enqueue(
            JobKind.TRAIN,
            {"profile_id": alice_voice.id, "private_debug": "do-not-return"},
            owner_user_id=alice_id,
        )
        synthesis_job = app.state.queue.enqueue(
            JobKind.SYNTHESIZE,
            {
                "voice_profile_id": alice_voice.id,
                "text": "private source text",
                "text_lang": "zh",
            },
            owner_user_id=alice_id,
        )

        dashboard = client.get("/api/dashboard")
        datasets = client.get("/api/datasets")
        safe_job_detail = client.get(f"/api/jobs/{train_job.id}")
        safe_synthesis_detail = client.get(f"/api/syntheses/{synthesis_job.id}")

        assert client.post("/api/auth/logout").status_code == 204
        bob_id = _login(client, "bob", "Correct-Horse-43")
        bob_voices = client.get("/api/voices")
        foreign_create = client.post(
            "/api/voices",
            json={"dataset_id": "alice-dataset", "display_name": "越权音色"},
        )

        assert client.post("/api/auth/logout").status_code == 204
        assert client.post(
            "/api/auth/login",
            json={"username": "alice", "password": "Correct-Horse-42"},
        ).status_code == 200
        short_create = client.post(
            "/api/voices",
            json={"dataset_id": "too-short", "display_name": "过短音色"},
        )
        unauthorized_create = client.post(
            "/api/voices",
            json={"dataset_id": "not-authorized", "display_name": "未授权音色"},
        )

    assert bob_id != alice_id
    assert dashboard.status_code == 200
    assert dashboard.json()["counts"] == {
        "datasets": 3,
        "voices": 1,
        "jobs": 2,
        "active_jobs": 2,
        "syntheses": 1,
    }
    assert {item["id"] for item in datasets.json()["items"]} == {
        "alice-dataset",
        "too-short",
        "not-authorized",
    }
    assert "private source text" not in dashboard.text
    assert "do-not-return" not in dashboard.text
    assert str(tmp_path) not in dashboard.text
    assert safe_job_detail.status_code == 200
    assert "payload" not in safe_job_detail.text
    assert "worker_id" not in safe_job_detail.text
    assert "do-not-return" not in safe_job_detail.text
    assert safe_synthesis_detail.status_code == 200
    assert "private source text" not in safe_synthesis_detail.text
    assert "public_audio_path" not in safe_synthesis_detail.text
    assert bob_voices.status_code == 200
    assert bob_voices.json()["items"] == []
    assert foreign_create.status_code == 404
    assert short_create.status_code == 409
    assert short_create.json()["error"]["code"] == "REFERENCE_DURATION_OUT_OF_RANGE"
    assert unauthorized_create.status_code == 409
    assert unauthorized_create.json()["error"]["code"] == "DATASET_AUTHORIZATION_REQUIRED"


def test_workspace_filters_status_and_marks_only_verified_output_downloadable(tmp_path):
    """Would fail if status filtering or output publication bypasses local file validation."""
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        owner_id = _login(client, "owner", "Correct-Horse-42")
        published = app.state.queue.enqueue(
            JobKind.SYNTHESIZE,
            {"voice_profile_id": "voice-1", "text": "private", "text_lang": "en"},
            owner_user_id=owner_id,
        )
        assert app.state.queue.claim_next("local-worker").id == published.id
        audio_path = app.state.storage.resolve("outputs", "verified.wav")
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"verified local audio")
        app.state.queue.succeed(
            published.id,
            {
                "public_audio_path": "verified.wav",
                "watermark_probability": 0.92,
                "fingerprint": {"anomaly": False},
                "speaker_similarity": 0.74,
                "quality_warning_codes": ["SPEAKER_SIMILARITY_BELOW_RECOMMENDED"],
            },
        )
        running = app.state.queue.enqueue(
            JobKind.TRAIN,
            {"profile_id": "voice-1"},
            owner_user_id=owner_id,
        )
        assert app.state.queue.claim_next("local-worker").id == running.id
        queued = app.state.queue.enqueue(
            JobKind.TRAIN,
            {"profile_id": "voice-1"},
            owner_user_id=owner_id,
        )
        unpublished = app.state.queue.enqueue(
            JobKind.SYNTHESIZE,
            {"voice_profile_id": "voice-1", "text": "private", "text_lang": "zh"},
            owner_user_id=owner_id,
        )

        running_jobs = client.get("/api/jobs?status=running")
        syntheses = client.get("/api/syntheses")
        client.cookies.clear()
        anonymous_dashboard = client.get("/api/dashboard")

    assert running_jobs.status_code == 200
    assert [item["id"] for item in running_jobs.json()["items"]] == [running.id]
    synthesis_by_id = {item["job_id"]: item for item in syntheses.json()["items"]}
    assert synthesis_by_id[published.id]["download_ready"] is True
    assert synthesis_by_id[published.id]["watermark_probability"] == 0.92
    assert synthesis_by_id[published.id]["fingerprint_anomaly"] is False
    assert synthesis_by_id[published.id]["speaker_similarity"] == 0.74
    assert synthesis_by_id[published.id]["quality_warning_codes"] == [
        "SPEAKER_SIMILARITY_BELOW_RECOMMENDED"
    ]
    assert synthesis_by_id[unpublished.id]["download_ready"] is False
    assert queued.id not in synthesis_by_id
    assert anonymous_dashboard.status_code == 401


def test_zero_shot_voice_summary_reports_reference_and_base_model(tmp_path):
    """Zero-shot profiles synthesize from references plus base weights, not weight dirs."""
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        owner_id = _login(client, "owner", "Correct-Horse-42")
        app.state.dataset_service.register_dataset(
            "dataset-1",
            effective_seconds=6.5,
            owner_user_id=owner_id,
            authorization_confirmed=True,
        )
        with app.state.engine.begin() as connection:
            connection.exec_driver_sql(
                "UPDATE dataset_segments SET snr_db=11.361 "
                "WHERE dataset_id='dataset-1'"
            )
        app.state.voice_service.create(
            VoiceProfileCreate(dataset_id="dataset-1", display_name="温柔旁白"),
            owner_user_id=owner_id,
        )

        voices = client.get("/api/voices")

    assert voices.status_code == 200
    items = voices.json()["items"]
    assert len(items) == 1
    summary = items[0]
    assert summary["display_name"] == "温柔旁白"
    assert summary["status"] == "ready"
    assert summary["mode"] == "zero_shot"
    assert summary["base_model_id"] == "gpt-sovits-v2proplus-official"
    assert summary["reference_emotions"] == ["neutral"]
    assert summary["can_synthesize"] is True
    assert summary["reference_snr_db"] == 11.361
    assert summary["quality_warning_codes"] == ["SNR_BELOW_RECOMMENDED"]


def test_workspace_batches_reference_quality_queries(tmp_path):
    """Adding profiles must not add one reference-SNR query per voice."""
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        owner_id = _login(client, "owner", "Correct-Horse-42")
        for dataset_id, snr_db in (("dataset-1", 11.361), ("dataset-2", 27.0)):
            app.state.dataset_service.register_dataset(
                dataset_id,
                effective_seconds=6.5,
                owner_user_id=owner_id,
                authorization_confirmed=True,
            )
            with app.state.engine.begin() as connection:
                connection.exec_driver_sql(
                    "UPDATE dataset_segments SET snr_db=? WHERE dataset_id=?",
                    (snr_db, dataset_id),
                )
            app.state.voice_service.create(
                VoiceProfileCreate(
                    dataset_id=dataset_id, display_name=f"批量音色 {dataset_id}"
                ),
                owner_user_id=owner_id,
            )

        segment_queries = []

        def capture_segment_query(
            connection, cursor, statement, parameters, context, executemany
        ):
            del connection, cursor, parameters, context, executemany
            if "FROM dataset_segments" in statement:
                segment_queries.append(statement)

        event.listen(app.state.engine, "before_cursor_execute", capture_segment_query)
        try:
            voices = client.get("/api/voices")
        finally:
            event.remove(
                app.state.engine, "before_cursor_execute", capture_segment_query
            )

    assert voices.status_code == 200
    summaries = {item["dataset_id"]: item for item in voices.json()["items"]}
    assert summaries["dataset-1"]["quality_warning_codes"] == [
        "SNR_BELOW_RECOMMENDED"
    ]
    assert summaries["dataset-2"]["quality_warning_codes"] == []
    assert len(segment_queries) == 1
