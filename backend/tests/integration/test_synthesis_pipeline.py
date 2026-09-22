from __future__ import annotations

import shutil
from hashlib import sha256
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import soundfile as sf

from backend.app.api.routes.synthesis import SynthesisService, router
from backend.app.api.routes.safety import SafetyService, router as safety_router
from backend.app.db.session import create_database_engine
from backend.app.schemas.common import JobKind, JobStatus
from backend.app.schemas.synthesis import EmotionControl, SynthesisCreate
from backend.app.services.fingerprint import FingerprintService, SpectralFingerprint
from backend.app.services.job_queue import JobQueue
from backend.app.services.sensitive_filter import SensitiveFilter
from backend.app.services.storage import LocalStorage
from backend.app.services.synthesis_pipeline import (
    ReferenceCondition,
    SynthesisPipeline,
    SynthesisRunResult,
)
from backend.app.services.voice_profiles import VoiceProfileStore
from backend.app.services.watermark import WatermarkDetection, WatermarkService
from backend.app.workers.gpu_worker import GPUWorker
from backend.tests.integration.auth_test_helpers import install_authenticated_user


class StaticReferenceResolver:
    def __init__(self, reference_audio: Path, prompt_lang: str) -> None:
        self.reference_audio = reference_audio
        self.prompt_lang = prompt_lang

    def resolve(self, profile, emotion: EmotionControl) -> ReferenceCondition:
        return ReferenceCondition(
            reference_audio=self.reference_audio,
            prompt_text="authorized reference",
            prompt_lang=self.prompt_lang,
        )


class PassingSimilarityGate:
    def __init__(self, score: float = 0.95) -> None:
        self.score = score

    def similarity(self, output_wav: Path, reference_wav: Path) -> float:
        return self.score


class WavSynthesizer:
    def synthesize(self, request, profile, reference, output_wav: Path) -> Path:
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_wav, np.zeros(16_000, dtype=np.float32), 16_000)
        return output_wav


class CopyingWatermarkBackend:
    def __init__(self, probability: float, *, payload_delta: int = 0) -> None:
        self.probability = probability
        self.payload_delta = payload_delta
        self.payload: int | None = None

    def embed(self, input_wav: Path, output_wav: Path, payload: int) -> None:
        self.payload = payload
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_wav, output_wav)

    def detect(self, wav: Path) -> WatermarkDetection:
        payload = None if self.payload is None else (self.payload + self.payload_delta) & 0xFFFF
        return WatermarkDetection(probability=self.probability, payload=payload)


def _ready_profile(tmp_path: Path):
    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    profiles = VoiceProfileStore(engine)
    profile = profiles.create("d1", owner_user_id="test-user")
    profiles.mark_queued(profile.id)
    profiles.mark_training(profile.id)
    public_weights = tmp_path / "data" / "profiles" / profile.id / "public"
    public_weights.mkdir(parents=True)
    return profiles, profiles.publish(profile.id, str(public_weights))


def _request(profile_id: str, text_lang: str = "en") -> SynthesisCreate:
    return SynthesisCreate(
        voice_profile_id=profile_id,
        text="safe text",
        text_lang=text_lang,
        emotion=EmotionControl(mode="manual", label="happy", strength=0.7),
        consent_confirmed=True,
    )


def test_blocked_text_creates_no_job_or_file(tmp_path: Path):
    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    queue = JobQueue(engine=engine)
    profiles = VoiceProfileStore(engine)
    storage = LocalStorage(tmp_path / "data")
    app = FastAPI()
    app.state.synthesis_service = SynthesisService(
        profiles=profiles,
        queue=queue,
        sensitive_filter=SensitiveFilter(["blocked term"]),
        storage=storage,
    )
    install_authenticated_user(app)
    app.include_router(router)
    client = TestClient(app)

    response = client.post(
        "/api/syntheses",
        json={
            "voice_profile_id": "does-not-matter",
            "text": "blocked-term",
            "text_lang": "en",
            "emotion": {"mode": "manual", "label": "happy", "strength": 0.7},
            "consent_confirmed": True,
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "SENSITIVE_TEXT_BLOCKED"
    assert queue.count_jobs() == 0
    assert list((tmp_path / "data").rglob("*.wav")) == []


def test_cloud_emotion_request_fails_before_queue_when_provider_is_not_configured(tmp_path: Path):
    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    queue = JobQueue(engine=engine)
    app = FastAPI()
    app.state.synthesis_service = SynthesisService(
        profiles=VoiceProfileStore(engine),
        queue=queue,
        sensitive_filter=SensitiveFilter([]),
        storage=LocalStorage(tmp_path / "data"),
        cloud_tts_available=False,
    )
    install_authenticated_user(app)
    app.include_router(router)

    response = TestClient(app).post(
        "/api/syntheses",
        json={
            "voice_profile_id": "voice-1",
            "text": "今天真开心。",
            "text_lang": "zh",
            "emotion": {"mode": "auto", "strength": 0.7},
            "consent_confirmed": True,
            "synthesis_mode": "emotion_api",
            "cloud_processing_confirmed": True,
            "cloud_options": {"control_mode": "label", "label": "happy"},
        },
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "CLOUD_TTS_NOT_CONFIGURED"
    assert queue.count_jobs() == 0


def test_audio_is_not_published_when_watermark_detection_fails(tmp_path: Path):
    profiles, profile = _ready_profile(tmp_path)
    reference_audio = tmp_path / "data" / "reference.wav"
    sf.write(reference_audio, np.zeros(16_000, dtype=np.float32), 16_000)
    storage = LocalStorage(tmp_path / "data")
    pipeline = SynthesisPipeline(
        profiles=profiles,
        sensitive_filter=SensitiveFilter([]),
        storage=storage,
        synthesizer=WavSynthesizer(),
        reference_resolver=StaticReferenceResolver(reference_audio, "zh"),
        watermark=WatermarkService(backend=CopyingWatermarkBackend(probability=0.79)),
        fingerprint=FingerprintService(),
        similarity_gate=PassingSimilarityGate(),
    )

    result = pipeline.run("job-watermark-fails", _request(profile.id))

    assert result.status == "failed"
    assert result.error_code == "WATERMARK_VERIFICATION_FAILED"
    assert result.public_audio_path is None
    assert list((tmp_path / "data" / "outputs").rglob("*.wav")) == []


def test_audio_is_not_published_when_watermark_payload_does_not_match(tmp_path: Path):
    profiles, profile = _ready_profile(tmp_path)
    reference_audio = tmp_path / "data" / "reference.wav"
    sf.write(reference_audio, np.zeros(16_000, dtype=np.float32), 16_000)
    storage = LocalStorage(tmp_path / "data")
    pipeline = SynthesisPipeline(
        profiles=profiles,
        sensitive_filter=SensitiveFilter([]),
        storage=storage,
        synthesizer=WavSynthesizer(),
        reference_resolver=StaticReferenceResolver(reference_audio, "zh"),
        watermark=WatermarkService(
            backend=CopyingWatermarkBackend(probability=0.91, payload_delta=1)
        ),
        fingerprint=FingerprintService(),
        similarity_gate=PassingSimilarityGate(),
    )

    result = pipeline.run("job-payload-mismatch", _request(profile.id))

    assert result.status == "failed"
    assert result.error_code == "WATERMARK_VERIFICATION_FAILED"
    assert result.public_audio_path is None
    assert list((tmp_path / "data" / "outputs").rglob("*.wav")) == []


def test_pipeline_blocks_sensitive_text_before_creating_temporary_audio(tmp_path: Path):
    profiles, profile = _ready_profile(tmp_path)
    storage = LocalStorage(tmp_path / "data")
    pipeline = SynthesisPipeline(
        profiles=profiles,
        sensitive_filter=SensitiveFilter(["blocked term"]),
        storage=storage,
        synthesizer=WavSynthesizer(),
        reference_resolver=StaticReferenceResolver(tmp_path / "data" / "missing.wav", "zh"),
        watermark=WatermarkService(backend=CopyingWatermarkBackend(probability=0.91)),
        fingerprint=FingerprintService(),
        similarity_gate=PassingSimilarityGate(),
    )
    request = _request(profile.id).model_copy(update={"text": "blocked-term"})

    result = pipeline.run("job-blocked-direct", request)

    assert result.status == "failed"
    assert result.error_code == "SENSITIVE_TEXT_BLOCKED"
    assert list((tmp_path / "data" / "temp").rglob("*.wav")) == []


def test_pipeline_does_not_expose_local_paths_when_synthesis_backend_crashes(tmp_path: Path):
    class RaisingSynthesizer:
        def synthesize(self, request, profile, reference, output_wav: Path) -> Path:
            raise RuntimeError(f"internal local path: {tmp_path}")

    profiles, profile = _ready_profile(tmp_path)
    reference_audio = tmp_path / "data" / "reference.wav"
    sf.write(reference_audio, np.zeros(16_000, dtype=np.float32), 16_000)
    storage = LocalStorage(tmp_path / "data")
    pipeline = SynthesisPipeline(
        profiles=profiles,
        sensitive_filter=SensitiveFilter([]),
        storage=storage,
        synthesizer=RaisingSynthesizer(),
        reference_resolver=StaticReferenceResolver(reference_audio, "zh"),
        watermark=WatermarkService(backend=CopyingWatermarkBackend(probability=0.91)),
        fingerprint=FingerprintService(),
        similarity_gate=PassingSimilarityGate(),
    )

    result = pipeline.run("job-backend-crash", _request(profile.id))

    assert result.status == "failed"
    assert result.error_code == "SYNTHESIS_PIPELINE_FAILED"
    assert result.error_message == "local synthesis pipeline failed"
    assert str(tmp_path) not in result.error_message


def test_verified_cross_language_reference_is_published_only_after_fingerprint(tmp_path: Path):
    profiles, profile = _ready_profile(tmp_path)
    reference_audio = tmp_path / "data" / "reference.wav"
    sf.write(reference_audio, np.zeros(16_000, dtype=np.float32), 16_000)
    storage = LocalStorage(tmp_path / "data")
    pipeline = SynthesisPipeline(
        profiles=profiles,
        sensitive_filter=SensitiveFilter([]),
        storage=storage,
        synthesizer=WavSynthesizer(),
        reference_resolver=StaticReferenceResolver(reference_audio, "zh"),
        watermark=WatermarkService(backend=CopyingWatermarkBackend(probability=0.81)),
        fingerprint=FingerprintService(),
        similarity_gate=PassingSimilarityGate(),
    )

    result = pipeline.run("job-cross-language", _request(profile.id, text_lang="en"))

    assert result.status == "succeeded"
    assert result.error_code is None
    assert result.public_audio_path is not None
    assert result.public_audio_path.is_file()
    assert result.watermark_probability == 0.81
    assert result.reference_language == "zh"
    assert result.fingerprint.sample_rate == 16_000


def test_gpu_worker_runs_synthesis_job_and_persists_verified_result(tmp_path: Path):
    class CompletedPipeline:
        def __init__(self) -> None:
            self.calls: list[tuple[str, SynthesisCreate]] = []
            self.storage = LocalStorage(tmp_path / "data")

        def run(self, job_id: str, request: SynthesisCreate, *, progress=None):
            self.calls.append((job_id, request))
            output = self.storage.resolve("outputs", "verified.wav")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"verified local audio")
            return SynthesisRunResult(
                status="succeeded",
                error_code=None,
                error_message=None,
                public_audio_path=output,
                watermark_probability=0.82,
                reference_language="zh",
                fingerprint=SpectralFingerprint(
                    sample_rate=16_000,
                    high_band_hz=(6_000.0, 7_600.0),
                    high_band_energy_ratio=0.1,
                    spectral_flatness=0.2,
                    zscore_vs_profile=0.0,
                    anomaly=False,
                ),
            )

    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    queue = JobQueue(engine=engine)
    profiles = VoiceProfileStore(engine)
    request = _request("profile-1")
    job = queue.enqueue(JobKind.SYNTHESIZE, request.model_dump(mode="json"))
    pipeline = CompletedPipeline()
    worker = GPUWorker(
        queue=queue,
        profiles=profiles,
        manifest_loader=lambda dataset_id: None,
        adapters=object(),
        synthesis_pipeline=pipeline,
    )

    worker.run_one()

    completed = queue.get(job.id)
    assert completed.status is JobStatus.SUCCEEDED
    assert pipeline.calls == [(job.id, request)]
    assert completed.result is not None
    assert completed.result["watermark_probability"] == 0.82
    assert completed.result["public_audio_path"].endswith("verified.wav")


def test_safety_route_rechecks_only_published_local_synthesis_audio(tmp_path: Path):
    class RecordingDetector:
        def __init__(self) -> None:
            self.detected_paths: list[Path] = []
            self.payload: int | None = None

        def embed(self, input_wav: Path, output_wav: Path, payload: int) -> None:
            raise AssertionError("safety detection must not embed a second watermark")

        def detect(self, wav: Path) -> WatermarkDetection:
            self.detected_paths.append(wav)
            return WatermarkDetection(probability=0.93, payload=self.payload)

    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    queue = JobQueue(engine=engine)
    storage = LocalStorage(tmp_path / "data")
    job = queue.enqueue(
        JobKind.SYNTHESIZE, {"safe": "payload"}, owner_user_id="test-user"
    )
    output = storage.resolve("outputs", "verified.wav")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"verified local audio")
    claimed = queue.claim_next("local-worker")
    assert claimed is not None
    queue.succeed(job.id, {"public_audio_path": "verified.wav"})
    detector = RecordingDetector()
    detector.payload = int.from_bytes(sha256(job.id.encode("utf-8")).digest()[:2], "big")
    app = FastAPI()
    app.state.safety_service = SafetyService(
        syntheses=SynthesisService(queue=queue, storage=storage),
        watermark=WatermarkService(backend=detector),
    )
    install_authenticated_user(app)
    app.include_router(safety_router)

    response = TestClient(app).post(
        "/api/safety/detect-watermark",
        json={"job_id": job.id},
    )

    assert response.status_code == 200
    assert response.json() == {
        "job_id": job.id,
        "probability": 0.93,
        "payload": detector.payload,
        "payload_matches_job": True,
    }
    assert detector.detected_paths == [output]


def test_safety_route_rejects_a_published_output_with_wrong_payload(tmp_path: Path):
    class WrongPayloadDetector:
        def embed(self, input_wav: Path, output_wav: Path, payload: int) -> None:
            raise AssertionError("safety detection must not embed a second watermark")

        def detect(self, wav: Path) -> WatermarkDetection:
            return WatermarkDetection(probability=0.93, payload=0)

    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    queue = JobQueue(engine=engine)
    storage = LocalStorage(tmp_path / "data")
    job = queue.enqueue(
        JobKind.SYNTHESIZE, {"safe": "payload"}, owner_user_id="test-user"
    )
    output = storage.resolve("outputs", "verified.wav")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"verified local audio")
    assert queue.claim_next("local-worker") is not None
    queue.succeed(job.id, {"public_audio_path": "verified.wav"})
    app = FastAPI()
    app.state.safety_service = SafetyService(
        syntheses=SynthesisService(queue=queue, storage=storage),
        watermark=WatermarkService(backend=WrongPayloadDetector()),
    )
    install_authenticated_user(app)
    app.include_router(safety_router)

    response = TestClient(app).post("/api/safety/detect-watermark", json={"job_id": job.id})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "WATERMARK_SOURCE_UNCONFIRMED"
    assert response.json()["error"]["message"] == "无法确认音频来源"
    assert str(tmp_path) not in response.text


def test_safety_route_marks_a_low_probability_published_output_as_unconfirmed(
    tmp_path: Path,
):
    class LowProbabilityDetector:
        def embed(self, input_wav: Path, output_wav: Path, payload: int) -> None:
            raise AssertionError("safety detection must not embed a second watermark")

        def detect(self, wav: Path) -> WatermarkDetection:
            return WatermarkDetection(probability=0.28, payload=None)

    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    queue = JobQueue(engine=engine)
    storage = LocalStorage(tmp_path / "data")
    job = queue.enqueue(
        JobKind.SYNTHESIZE, {"safe": "payload"}, owner_user_id="test-user"
    )
    output = storage.resolve("outputs", "verified.wav")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"verified local audio")
    assert queue.claim_next("local-worker") is not None
    queue.succeed(job.id, {"public_audio_path": "verified.wav"})
    app = FastAPI()
    app.state.safety_service = SafetyService(
        syntheses=SynthesisService(queue=queue, storage=storage),
        watermark=WatermarkService(backend=LowProbabilityDetector()),
    )
    install_authenticated_user(app)
    app.include_router(safety_router)

    response = TestClient(app).post("/api/safety/detect-watermark", json={"job_id": job.id})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "WATERMARK_SOURCE_UNCONFIRMED"
    assert response.json()["error"]["message"] == "无法确认音频来源"
    assert str(tmp_path) not in response.text


def test_safety_route_does_not_probe_audio_for_an_unknown_synthesis(tmp_path: Path):
    class FailingDetector:
        def embed(self, input_wav: Path, output_wav: Path, payload: int) -> None:
            raise AssertionError("safety detection must not embed a second watermark")

        def detect(self, wav: Path) -> WatermarkDetection:
            raise AssertionError("unknown synthesis must not resolve an audio path")

    queue = JobQueue(engine=create_database_engine("sqlite+pysqlite:///:memory:"))
    storage = LocalStorage(tmp_path / "data")
    app = FastAPI()
    app.state.safety_service = SafetyService(
        syntheses=SynthesisService(queue=queue, storage=storage),
        watermark=WatermarkService(backend=FailingDetector()),
    )
    install_authenticated_user(app)
    app.include_router(safety_router)

    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/safety/detect-watermark", json={"job_id": "not-a-synthesis"}
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "SYNTHESIS_NOT_FOUND"
    assert str(tmp_path) not in response.text


def test_safety_route_does_not_probe_audio_that_is_not_published(tmp_path: Path):
    class FailingDetector:
        def embed(self, input_wav: Path, output_wav: Path, payload: int) -> None:
            raise AssertionError("safety detection must not embed a second watermark")

        def detect(self, wav: Path) -> WatermarkDetection:
            raise AssertionError("unpublished synthesis must not resolve an audio path")

    queue = JobQueue(engine=create_database_engine("sqlite+pysqlite:///:memory:"))
    job = queue.enqueue(
        JobKind.SYNTHESIZE, {"safe": "payload"}, owner_user_id="test-user"
    )
    app = FastAPI()
    app.state.safety_service = SafetyService(
        syntheses=SynthesisService(queue=queue, storage=LocalStorage(tmp_path / "data")),
        watermark=WatermarkService(backend=FailingDetector()),
    )
    install_authenticated_user(app)
    app.include_router(safety_router)

    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/safety/detect-watermark", json={"job_id": job.id}
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "SYNTHESIS_AUDIO_NOT_READY"
    assert str(tmp_path) not in response.text


def test_safety_route_rejects_untrusted_file_path_fields_before_service_lookup():
    app = FastAPI()
    app.state.safety_service = object()
    install_authenticated_user(app)
    app.include_router(safety_router)

    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/safety/detect-watermark",
        json={"job_id": "local-job", "path": "C:/must-not-be-read.wav"},
    )

    assert response.status_code == 422
