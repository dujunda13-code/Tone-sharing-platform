from dataclasses import replace
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend.app.api.routes.health import HealthService
from backend.app.core.config import get_settings
from backend.app.main import create_app
from backend.app.services import emotion as emotion_module
from backend.app.services.emotion import EmotionAnalyzer
from backend.app.services.storage import LocalStorage


def _settings(tmp_path):
    settings = get_settings()
    return replace(settings, storage=replace(settings.storage, root=tmp_path / "data"))


def _health_service_with_counted_probe(monkeypatch, tmp_path):
    """Use a controlled local probe so the test observes cache behavior, not GPU speed."""
    app = create_app(_settings(tmp_path))
    models_root = tmp_path / "models"
    models_root.mkdir()
    model_file = models_root / "example.bin"
    model_file.write_bytes(b"original")
    (models_root / "checksums.sha256").write_text(
        f"{'0' * 64}  models/example.bin\n", encoding="utf-8"
    )
    calls: list[int] = []

    class CountedAdapter:
        def __init__(self, **_kwargs):
            pass

        def probe(self):
            calls.append(1)
            return SimpleNamespace(available=True, tag=f"probe-{len(calls)}", reason=None)

    monkeypatch.setattr("backend.app.api.routes.health.GPTSoVITSAdapter", CountedAdapter)
    service = HealthService(
        engine=app.state.engine,
        storage=app.state.storage,
        settings=app.state.settings,
        project_root=tmp_path,
    )
    return service, calls, model_file


def test_gpt_sovits_readiness_reuses_probe_for_unchanged_model_state(monkeypatch, tmp_path):
    """Would fail if repeated UI readiness reads rescan the unchanged model set."""
    service, calls, _ = _health_service_with_counted_probe(monkeypatch, tmp_path)

    first = service._gpt_sovits()
    second = service._gpt_sovits()

    assert first == (True, "GPT-SoVITS probe-1 is ready")
    assert second == first
    assert calls == [1]


def test_gpt_sovits_readiness_reprobes_after_registered_file_changes(monkeypatch, tmp_path):
    """Would fail if a changed registered model file keeps an old ready result."""
    service, calls, model_file = _health_service_with_counted_probe(monkeypatch, tmp_path)

    first = service._gpt_sovits()
    unchanged = service._gpt_sovits()
    model_file.write_bytes(b"changed-model-content")
    os.utime(model_file, None)
    changed = service._gpt_sovits()

    assert first == (True, "GPT-SoVITS probe-1 is ready")
    assert unchanged == first
    assert changed == (True, "GPT-SoVITS probe-2 is ready")
    assert calls == [1, 1]


def test_concurrent_gpt_sovits_readiness_requests_share_one_probe(monkeypatch, tmp_path):
    """Would fail if two simultaneous UI reads begin two expensive model scans."""
    app = create_app(_settings(tmp_path))
    models_root = tmp_path / "models"
    models_root.mkdir()
    (models_root / "checksums.sha256").write_text(
        f"{'0' * 64}  models/example.bin\n", encoding="utf-8"
    )
    (models_root / "example.bin").write_bytes(b"model")
    calls: list[int] = []
    probe_started = threading.Event()
    allow_probe_return = threading.Event()

    class BlockingAdapter:
        def __init__(self, **_kwargs):
            pass

        def probe(self):
            calls.append(1)
            probe_started.set()
            assert allow_probe_return.wait(timeout=2)
            return SimpleNamespace(available=True, tag="shared", reason=None)

    monkeypatch.setattr("backend.app.api.routes.health.GPTSoVITSAdapter", BlockingAdapter)
    service = HealthService(
        engine=app.state.engine,
        storage=app.state.storage,
        settings=app.state.settings,
        project_root=tmp_path,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(service._gpt_sovits)
        assert probe_started.wait(timeout=1)
        second = executor.submit(service._gpt_sovits)
        deadline = time.monotonic() + 0.2
        while len(calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        allow_probe_return.set()
        assert first.result(timeout=2) == (True, "GPT-SoVITS shared is ready")
        assert second.result(timeout=2) == (True, "GPT-SoVITS shared is ready")

    assert calls == [1]


def test_live_health_endpoint_is_process_local(tmp_path):
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        response = client.get("/api/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readiness_reports_every_required_dependency(tmp_path):
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        response = client.get("/api/health/ready")

    body = response.json()
    assert set(body["checks"]) == {
        "database",
        "storage",
        "gpu",
        "gpt_sovits",
        "emotion2vec",
        "audioseal",
    }
    assert response.status_code in {200, 503}
    assert all("ok" in check and "message" in check for check in body["checks"].values())


def test_create_app_injects_local_emotion_analyzer_and_readiness_uses_it(
    tmp_path, monkeypatch
):
    model_dir = tmp_path / "models" / "emotion2vec"
    monkeypatch.setattr(emotion_module, "DEFAULT_EMOTION_MODEL_DIR", model_dir)
    app = create_app(_settings(tmp_path))

    analyzer = app.state.dataset_service.emotion_analyzer
    assert isinstance(analyzer, EmotionAnalyzer)
    assert analyzer.available() is False
    assert app.state.health_service.checks()["emotion2vec"].ok is False

    model_dir.mkdir(parents=True)
    (model_dir / "config.yaml").write_text("config", encoding="utf-8")
    (model_dir / "model.pt").write_bytes(b"weights")
    (model_dir / "tokens.txt").write_text("tokens", encoding="utf-8")

    assert analyzer.available() is True
    assert app.state.health_service.checks()["emotion2vec"].ok is True


def test_ready_endpoint_uses_safe_public_messages(tmp_path):
    """Would fail if public readiness responses include paths, stack traces, or model internals."""
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        response = client.get("/api/health/ready")

    assert response.status_code in {200, 503}
    assert str(tmp_path) not in response.text
    assert "Traceback" not in response.text
    assert all(
        check["message"]
        in {
            "本地数据库可用",
            "本地数据库不可用",
            "本地存储可用",
            "本地存储不可用",
            "GPU cuda:0 可用",
            "GPU cuda:0 未就绪",
            "GPT-SoVITS 可用",
            "GPT-SoVITS 未就绪",
            "Emotion2Vec 可用",
            "Emotion2Vec 未就绪",
            "AudioSeal 可用",
            "AudioSeal 未就绪",
        }
        for check in response.json()["checks"].values()
    )


def test_startup_recovers_interrupted_jobs(monkeypatch, tmp_path):
    calls: list[str] = []

    def recover(self):
        calls.append("recover")
        return 3

    monkeypatch.setattr("backend.app.services.job_queue.JobQueue.recover_interrupted", recover)
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        assert client.app.state.recovered_jobs == 3

    assert calls == ["recover"]


def test_health_reports_local_storage_runtime_and_model_failures(monkeypatch, tmp_path):
    app = create_app(_settings(tmp_path))
    broken_root = tmp_path / "storage-file"
    broken_root.write_text("not a directory", encoding="utf-8")
    broken_storage = object.__new__(LocalStorage)
    broken_storage.root = broken_root
    settings = replace(
        app.state.settings,
        runtime=replace(app.state.settings.runtime, device="cpu"),
    )
    service = HealthService(
        engine=app.state.engine,
        storage=broken_storage,
        settings=settings,
        project_root=tmp_path,
    )

    checks = service.checks()

    assert checks["storage"].ok is False
    assert checks["gpu"].message == "runtime device must be cuda:0"
    assert checks["gpt_sovits"].ok is False
    assert checks["emotion2vec"].ok is False
    assert checks["audioseal"].ok is False

    emotion_dir = tmp_path / "models" / "emotion2vec"
    emotion_dir.mkdir(parents=True)
    (emotion_dir / "config.yaml").write_text("config", encoding="utf-8")
    (emotion_dir / "model.pt").write_bytes(b"weights")
    (emotion_dir / "tokens.txt").write_text("tokens", encoding="utf-8")
    ready_model_service = HealthService(
        engine=app.state.engine,
        storage=app.state.storage,
        settings=app.state.settings,
        project_root=tmp_path,
    )
    assert ready_model_service._emotion2vec()[0] is True

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert ready_model_service._gpu() == (False, "cuda:0 is unavailable")
