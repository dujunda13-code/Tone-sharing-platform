"""Zero-shot synthesis: frozen base weights plus persisted local references.

The client worker must synthesize with checksum-verified base S1/S2 weights
and the profile's persisted reference audio, reject every training job with
CLIENT_TRAINING_DISABLED, and leave no training artifacts on disk.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from backend.app.core.config import get_settings
from backend.app.db.models import DatasetSegment
from backend.app.db.session import create_database_engine, session_factory
from backend.app.schemas.common import JobKind, JobStatus
from backend.app.schemas.synthesis import EmotionControl, SynthesisCreate
from backend.app.services.fingerprint import FingerprintService
from backend.app.services.gpt_sovits import RawSynthesis, SynthesisSpec
from backend.app.services.job_queue import JobQueue
from backend.app.services.sensitive_filter import SensitiveFilter
from backend.app.services.storage import LocalStorage
from backend.app.services.synthesis_pipeline import (
    GPTSoVITSLocalSynthesizer,
    SynthesisPipeline,
    VoiceProfileUnavailable,
    ZeroShotReferenceResolver,
)
from backend.app.services.voice_base_registry import VoiceBaseRegistry
from backend.app.services.voice_profiles import VoiceProfileStore
from backend.app.services.watermark import WatermarkDetection, WatermarkService
from backend.app.workers.gpu_worker import GPUWorker


def _write_weight(models_root: Path, relative: str, payload: bytes) -> None:
    weight_path = models_root / relative
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    weight_path.write_bytes(payload)


def _make_registry(tmp_path: Path) -> VoiceBaseRegistry:
    models_root = tmp_path / "models"
    gpt_payload = b"zero-shot-gpt-weights"
    sovits_payload = b"zero-shot-sovits-weights"
    _write_weight(models_root, "gpt-sovits/v2Pro/s1v3.ckpt", gpt_payload)
    _write_weight(models_root, "gpt-sovits/v2Pro/s2Gv2ProPlus.pth", sovits_payload)
    manifest = models_root / "checksums.sha256"
    manifest.write_text(
        "\n".join(
            [
                f"{hashlib.sha256(gpt_payload).hexdigest()}  models/gpt-sovits/v2Pro/s1v3.ckpt",
                f"{hashlib.sha256(sovits_payload).hexdigest()}  models/gpt-sovits/v2Pro/s2Gv2ProPlus.pth",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    settings = get_settings()
    return VoiceBaseRegistry(settings, models_root, manifest)


def _zero_shot_profile(tmp_path: Path, registry: VoiceBaseRegistry):
    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    profiles = VoiceProfileStore(engine)
    storage = LocalStorage(tmp_path / "data")
    profile = profiles.create_zero_shot(
        dataset_id="dataset-1",
        display_name="确认参考音色",
        owner_user_id="user-1",
        reference_asset_id="asset-1",
        reference_segment_id="seg_0001",
        prompt_text="这是确认后的参考文本。",
        prompt_language="zh",
        emotion_label="neutral",
        emotion_confidence=0.91,
        base_model_id="gpt-sovits-v2proplus-official",
    )
    with session_factory(engine)() as session:
        session.add(
            DatasetSegment(
                dataset_id="dataset-1",
                segment_id="seg_0001",
                owner_user_id="user-1",
                relative_path="datasets/dataset-1/segments/seg_0001.wav",
                duration_seconds=6.5,
                language="zh",
                auto_transcript="这是确认后的参考文本。",
                auto_emotion_label="neutral",
            )
        )
        session.commit()
    segment_dir = storage.resolve("datasets", Path("dataset-1") / "segments")
    segment_dir.mkdir(parents=True, exist_ok=True)
    sf.write(
        segment_dir / "seg_0001.wav",
        np.zeros(16_000, dtype=np.float32),
        16_000,
    )
    return profiles, storage, profile


class RecordingAdapter:
    def __init__(self) -> None:
        self.specs: list[SynthesisSpec] = []

    def synthesize(self, spec: SynthesisSpec) -> RawSynthesis:
        self.specs.append(spec)
        output_wav = Path(spec.output_wav)
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_wav, np.zeros(16_000, dtype=np.float32), 16_000)
        return RawSynthesis(
            output_wav=output_wav,
            gpt_weight=spec.gpt_weight,
            sovits_weight=spec.sovits_weight,
            profile_dir=output_wav.parent,
        )


class StubSimilarityGate:
    def __init__(self, score: float) -> None:
        self.score = score
        self.calls: list[tuple[Path, Path]] = []

    def similarity(self, output_wav: Path, reference_wav: Path) -> float:
        self.calls.append((output_wav, reference_wav))
        return self.score


class CopyingWatermarkBackend:
    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.payload: int | None = None

    def embed(self, input_wav: Path, output_wav: Path, payload: int) -> None:
        self.payload = payload
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(input_wav, output_wav)

    def detect(self, wav: Path) -> WatermarkDetection:
        return WatermarkDetection(probability=self.probability, payload=self.payload)


def _resolver(profiles: VoiceProfileStore, storage: LocalStorage, registry) -> ZeroShotReferenceResolver:
    return ZeroShotReferenceResolver(profiles=profiles, storage=storage, registry=registry)


def _pipeline(
    profiles: VoiceProfileStore,
    storage: LocalStorage,
    registry,
    adapter: RecordingAdapter,
    gate: StubSimilarityGate,
    watermark_probability: float = 0.92,
) -> SynthesisPipeline:
    return SynthesisPipeline(
        profiles=profiles,
        sensitive_filter=SensitiveFilter([]),
        storage=storage,
        synthesizer=GPTSoVITSLocalSynthesizer(adapter),
        reference_resolver=_resolver(profiles, storage, registry),
        watermark=WatermarkService(
            backend=CopyingWatermarkBackend(probability=watermark_probability)
        ),
        fingerprint=FingerprintService(),
        similarity_gate=gate,
    )


def _request(profile_id: str) -> SynthesisCreate:
    return SynthesisCreate(
        voice_profile_id=profile_id,
        text="safe local text",
        text_lang="en",
        emotion=EmotionControl(mode="auto"),
        consent_confirmed=True,
    )


def test_zero_shot_synthesis_uses_base_weights_and_persisted_reference(tmp_path):
    registry = _make_registry(tmp_path)
    profiles, storage, profile = _zero_shot_profile(tmp_path, registry)
    adapter = RecordingAdapter()
    gate = StubSimilarityGate(score=0.95)
    pipeline = _pipeline(profiles, storage, registry, adapter, gate)

    result = pipeline.run("job-zero-shot", _request(profile.id))

    assert result.status == "succeeded", result.error_message
    assert len(adapter.specs) == 3
    spec = adapter.specs[0]
    assert spec.gpt_weight == (tmp_path / "models/gpt-sovits/v2Pro/s1v3.ckpt").resolve()
    assert spec.sovits_weight == (
        tmp_path / "models/gpt-sovits/v2Pro/s2Gv2ProPlus.pth"
    ).resolve()
    assert spec.reference_audio == storage.resolve(
        "datasets", Path("dataset-1/segments/seg_0001.wav")
    )
    assert spec.prompt_text == "这是确认后的参考文本。"
    assert spec.prompt_lang == "zh"
    assert profile.public_weight_dir is None
    assert len(gate.calls) == 3
    assert result.speaker_similarity == 0.95
    assert result.candidate_count == 3


def test_low_similarity_result_publishes_with_quality_warning(tmp_path):
    registry = _make_registry(tmp_path)
    profiles, storage, profile = _zero_shot_profile(tmp_path, registry)
    adapter = RecordingAdapter()
    gate = StubSimilarityGate(score=0.89)
    pipeline = _pipeline(profiles, storage, registry, adapter, gate)

    result = pipeline.run("job-low-similarity", _request(profile.id))

    # Below the recommended value the result stays usable: it still runs the
    # mandatory watermark/fingerprint gates, publishes, and carries the real
    # score plus an explicit quality warning instead of failing.
    assert result.status == "succeeded", result.error_message
    assert result.speaker_similarity == 0.89
    assert result.quality_warning_codes == ("SPEAKER_SIMILARITY_BELOW_RECOMMENDED",)
    assert result.watermark_probability == 0.92
    assert result.fingerprint is not None
    published = storage.resolve("outputs", str(result.public_audio_path))
    assert published.is_file()
    assert published.stat().st_size > 0


def test_similarity_at_recommended_value_carries_no_warning(tmp_path):
    registry = _make_registry(tmp_path)
    profiles, storage, profile = _zero_shot_profile(tmp_path, registry)
    adapter = RecordingAdapter()
    gate = StubSimilarityGate(score=0.90)
    pipeline = _pipeline(profiles, storage, registry, adapter, gate)

    result = pipeline.run("job-boundary-similarity", _request(profile.id))

    assert result.status == "succeeded", result.error_message
    assert result.speaker_similarity == 0.90
    assert result.quality_warning_codes == ()


def test_watermark_recheck_failure_still_blocks_low_similarity_publication(tmp_path):
    registry = _make_registry(tmp_path)
    profiles, storage, profile = _zero_shot_profile(tmp_path, registry)
    adapter = RecordingAdapter()
    gate = StubSimilarityGate(score=0.89)
    pipeline = _pipeline(
        profiles, storage, registry, adapter, gate, watermark_probability=0.5
    )

    result = pipeline.run("job-low-similarity-bad-watermark", _request(profile.id))

    # The advisory similarity change never weakens the mandatory safety gates.
    assert result.status == "failed"
    assert result.error_code == "WATERMARK_VERIFICATION_FAILED"
    assert result.public_audio_path is None
    assert not list((tmp_path / "data" / "outputs").rglob("*.wav"))


def test_missing_emotion_reference_fails_without_synthesis(tmp_path):
    registry = _make_registry(tmp_path)
    profiles, storage, profile = _zero_shot_profile(tmp_path, registry)
    adapter = RecordingAdapter()
    gate = StubSimilarityGate(score=0.95)
    pipeline = _pipeline(profiles, storage, registry, adapter, gate)
    request = SynthesisCreate(
        voice_profile_id=profile.id,
        text="safe local text",
        text_lang="zh",
        emotion=EmotionControl(mode="manual", label="angry"),
        consent_confirmed=True,
    )

    result = pipeline.run("job-missing-emotion", request)

    assert result.status == "failed"
    assert result.error_code == "EMOTION_REFERENCE_UNAVAILABLE"
    assert adapter.specs == []


def test_base_model_hash_mismatch_fails_before_launching_synthesis(tmp_path):
    registry = _make_registry(tmp_path)
    (tmp_path / "models/gpt-sovits/v2Pro/s1v3.ckpt").write_bytes(b"tampered")
    profiles, storage, profile = _zero_shot_profile(tmp_path, registry)
    adapter = RecordingAdapter()
    gate = StubSimilarityGate(score=0.95)
    pipeline = _pipeline(profiles, storage, registry, adapter, gate)

    result = pipeline.run("job-hash-mismatch", _request(profile.id))

    assert result.status == "failed"
    assert result.error_code == "BASE_MODEL_UNAVAILABLE"
    assert adapter.specs == []


def test_foreign_owner_reference_is_never_used(tmp_path):
    registry = _make_registry(tmp_path)
    profiles, storage, profile = _zero_shot_profile(tmp_path, registry)
    # Owner-scoped reference lookup is the only path to reference audio, so a
    # foreign owner can neither list nor synthesize with user-1's references.
    import pytest

    with pytest.raises(KeyError):
        profiles.references(profile.id, "user-2")


def test_reference_resolution_is_bound_to_the_profile_dataset(tmp_path):
    registry = _make_registry(tmp_path)
    profiles, storage, profile = _zero_shot_profile(tmp_path, registry)
    with session_factory(profiles.engine)() as session:
        session.delete(
            session.get(
                DatasetSegment,
                {"dataset_id": "dataset-1", "segment_id": "seg_0001"},
            )
        )
        session.add(
            DatasetSegment(
                dataset_id="dataset-2",
                segment_id="seg_0001",
                owner_user_id="user-2",
                relative_path="datasets/dataset-2/segments/seg_0001.wav",
                duration_seconds=6.5,
                language="zh",
                auto_transcript="另一数据集的文本。",
                auto_emotion_label="neutral",
            )
        )
        session.commit()

    with pytest.raises(VoiceProfileUnavailable):
        _resolver(profiles, storage, registry).resolve(
            profile, EmotionControl(mode="auto")
        )


def test_worker_rejects_training_jobs_with_client_training_disabled(tmp_path):
    registry = _make_registry(tmp_path)
    profiles, storage, profile = _zero_shot_profile(tmp_path, registry)
    queue = JobQueue(engine=profiles.engine)
    queue.enqueue(
        JobKind.TRAIN,
        {"profile_id": profile.id, "dataset_id": profile.dataset_id, "consent_confirmed": True},
        owner_user_id="user-1",
    )
    adapter = RecordingAdapter()
    gate = StubSimilarityGate(score=0.95)
    worker = GPUWorker(
        queue=queue,
        profiles=profiles,
        manifest_loader=lambda dataset_id: (_ for _ in ()).throw(AssertionError("manifest must not load")),
        adapters=SimpleNamespace(
            preflight=lambda *_: (_ for _ in ()).throw(AssertionError("training adapter called")),
            feature_extract=lambda *_: (_ for _ in ()).throw(AssertionError("training adapter called")),
            gpt_sovits_train=lambda *_: (_ for _ in ()).throw(AssertionError("training adapter called")),
            disentangler_train=lambda *_: (_ for _ in ()).throw(AssertionError("training adapter called")),
            evaluate=lambda *_: (_ for _ in ()).throw(AssertionError("training adapter called")),
        ),
        synthesis_pipeline=_pipeline(profiles, storage, registry, adapter, gate),
        dataset_preparer=SimpleNamespace(
            prepare=lambda *_: (_ for _ in ()).throw(AssertionError("preparer called"))
        ),
        gpt_sovits_adapter=SimpleNamespace(
            train=lambda *_: (_ for _ in ()).throw(AssertionError("S1/S2 trainer called"))
        ),
        disentangler_trainer=SimpleNamespace(
            fit=lambda *_: (_ for _ in ()).throw(AssertionError("disentangler called"))
        ),
        feature_batch_loader=lambda *_: (_ for _ in ()).throw(AssertionError("features loaded")),
    )

    completed = worker.run_one()

    assert completed is not None
    assert completed.status is JobStatus.FAILED
    assert completed.error_code == "CLIENT_TRAINING_DISABLED"
    assert queue.count_jobs() == 1
    stored = profiles.get(profile.id, "user-1")
    assert stored.status == "ready"


def test_worker_persists_candidate_progress_during_synthesis(tmp_path):
    registry = _make_registry(tmp_path)
    profiles, storage, profile = _zero_shot_profile(tmp_path, registry)
    queue = JobQueue(engine=profiles.engine)
    job = queue.enqueue(
        JobKind.SYNTHESIZE,
        _request(profile.id).model_dump(mode="json"),
        owner_user_id="user-1",
    )

    class ProgressObservingAdapter(RecordingAdapter):
        def __init__(self, queue: JobQueue, job_id: str) -> None:
            super().__init__()
            self._queue = queue
            self._job_id = job_id
            self.observed: list[str | None] = []

        def synthesize(self, spec: SynthesisSpec) -> RawSynthesis:
            self.observed.append(self._queue.get(self._job_id).progress_message)
            return super().synthesize(spec)

    adapter = ProgressObservingAdapter(queue, job.id)
    gate = StubSimilarityGate(score=0.96)
    data_root = tmp_path / "data"
    features_root = data_root / "features"
    features_root.mkdir(parents=True)
    worker = GPUWorker(
        queue=queue,
        profiles=profiles,
        manifest_loader=lambda dataset_id: SimpleNamespace(dataset_id=dataset_id),
        adapters=object(),
        profiles_root=data_root / "profiles",
        features_root=features_root,
        synthesis_pipeline=_pipeline(profiles, storage, registry, adapter, gate),
    )

    worker.run_one()

    # While each candidate was being generated the job record exposed the
    # live progress message written through the worker's progress wiring.
    assert adapter.observed == [
        "正在生成候选 1/3...",
        "正在生成候选 2/3...",
        "正在生成候选 3/3...",
    ]
    assert queue.get(job.id).status is JobStatus.SUCCEEDED


def test_worker_runs_zero_shot_synthesis_and_leaves_no_training_artifacts(tmp_path):
    registry = _make_registry(tmp_path)
    profiles, storage, profile = _zero_shot_profile(tmp_path, registry)
    queue = JobQueue(engine=profiles.engine)
    job = queue.enqueue(
        JobKind.SYNTHESIZE,
        _request(profile.id).model_dump(mode="json"),
        owner_user_id="user-1",
    )
    adapter = RecordingAdapter()
    gate = StubSimilarityGate(score=0.96)
    data_root = tmp_path / "data"
    features_root = data_root / "features"
    features_root.mkdir(parents=True)
    worker = GPUWorker(
        queue=queue,
        profiles=profiles,
        manifest_loader=lambda dataset_id: SimpleNamespace(dataset_id=dataset_id),
        adapters=object(),
        profiles_root=data_root / "profiles",
        features_root=features_root,
        synthesis_pipeline=_pipeline(profiles, storage, registry, adapter, gate),
    )

    worker.run_one()

    completed = queue.get(job.id)
    assert completed.status is JobStatus.SUCCEEDED
    assert completed.result is not None
    assert completed.result["watermark_probability"] >= 0.80
    assert completed.result["fingerprint"]["anomaly"] is False
    # No training disk side effects anywhere.
    assert list(features_root.rglob("*")) == []
    assert not (data_root / "profiles" / profile.id / "weights").exists()
    assert not (data_root / "profiles" / profile.id / "public" / "weights").exists()
    assert not (data_root / "profiles").exists()
    assert not any(
        path.suffix.casefold() in {".ckpt", ".pth", ".pt", ".safetensors"}
        or path.name.casefold().startswith("checkpoint")
        for path in data_root.rglob("*")
        if path.is_file()
    )
    assert queue.count_jobs() == 1


def test_worker_publishes_low_similarity_result_with_warning(tmp_path):
    registry = _make_registry(tmp_path)
    profiles, storage, profile = _zero_shot_profile(tmp_path, registry)
    queue = JobQueue(engine=profiles.engine)
    job = queue.enqueue(
        JobKind.SYNTHESIZE,
        _request(profile.id).model_dump(mode="json"),
        owner_user_id="user-1",
    )
    adapter = RecordingAdapter()
    gate = StubSimilarityGate(score=0.89)
    data_root = tmp_path / "data"
    features_root = data_root / "features"
    features_root.mkdir(parents=True)
    worker = GPUWorker(
        queue=queue,
        profiles=profiles,
        manifest_loader=lambda dataset_id: SimpleNamespace(dataset_id=dataset_id),
        adapters=object(),
        profiles_root=data_root / "profiles",
        features_root=features_root,
        synthesis_pipeline=_pipeline(profiles, storage, registry, adapter, gate),
    )

    worker.run_one()

    completed = queue.get(job.id)
    assert completed.status is JobStatus.SUCCEEDED
    assert completed.result is not None
    assert completed.result["speaker_similarity"] == 0.89
    assert completed.result["quality_warning_codes"] == [
        "SPEAKER_SIMILARITY_BELOW_RECOMMENDED"
    ]
    assert storage.resolve(
        "outputs", str(completed.result["public_audio_path"])
    ).is_file()


def test_synthesis_request_schema_exposes_no_reference_path_field():
    field_names = set(SynthesisCreate.model_fields)
    assert "reference_audio" not in field_names
    assert "audio_path" not in field_names
    assert "reference" not in field_names
