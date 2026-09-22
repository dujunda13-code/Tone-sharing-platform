from pathlib import Path
import numpy as np
import soundfile as sf

from backend.app.schemas.synthesis import EmotionControl, SynthesisCreate
from backend.app.services.fingerprint import FingerprintService
from backend.app.services.storage import LocalStorage
from backend.app.services.synthesis_pipeline import (
    ReferenceCondition,
    SynthesisPipeline,
)
from backend.app.services.voice_profiles import VoiceProfileStore
from backend.app.services.watermark import WatermarkDetection, WatermarkService
from backend.app.db.session import create_database_engine


class MultiCandidateSynthesizer:
    def __init__(self) -> None:
        self.synthesized_seeds: list[int] = []
        self.synthesized_paths: list[Path] = []

    def synthesize(self, request, profile, reference, output_wav: Path, *, seed: int = 0) -> Path:
        self.synthesized_seeds.append(seed)
        self.synthesized_paths.append(output_wav)
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_wav, np.zeros(16_000, dtype=np.float32), 16_000)
        return output_wav


class SequenceSimilarityGate:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.index = 0

    def similarity(self, output_wav: Path, reference_wav: Path) -> float:
        if self.index < len(self.scores):
            score = self.scores[self.index]
            self.index += 1
            return score
        return 0.85


class SequenceWatermarkBackend:
    def __init__(self, detections: list[tuple[float, int]]) -> None:
        self.detections = detections
        self.detect_index = 0

    def embed(self, input_wav: Path, output_wav: Path, payload: int) -> None:
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_wav, np.zeros(16_000, dtype=np.float32), 16_000)

    def detect(self, wav: Path) -> WatermarkDetection:
        if self.detect_index < len(self.detections):
            prob, payload = self.detections[self.detect_index]
            self.detect_index += 1
            return WatermarkDetection(probability=prob, payload=payload)
        return WatermarkDetection(probability=0.9, payload=123)


class StaticReferenceResolver:
    def __init__(self, reference_audio: Path) -> None:
        self.reference_audio = reference_audio
        self.emotions: list[EmotionControl] = []

    def resolve(self, profile, emotion: EmotionControl) -> ReferenceCondition:
        self.emotions.append(emotion)
        return ReferenceCondition(
            reference_audio=self.reference_audio,
            prompt_text="authorized reference",
            prompt_lang="zh",
        )


def _setup_pipeline(tmp_path: Path, scores: list[float], detections: list[tuple[float, int]], max_candidates: int = 3):
    engine = create_database_engine("sqlite://")
    profiles = VoiceProfileStore(engine)
    profile = profiles.create("d1", owner_user_id="test-user")
    profiles.mark_queued(profile.id)
    profiles.mark_training(profile.id)
    public_weights = tmp_path / "data" / "profiles" / profile.id / "public"
    public_weights.mkdir(parents=True)
    published_profile = profiles.publish(profile.id, str(public_weights))

    ref_audio = tmp_path / "data" / "ref.wav"
    ref_audio.parent.mkdir(parents=True, exist_ok=True)
    sf.write(ref_audio, np.zeros(16_000, dtype=np.float32), 16_000)

    synthesizer = MultiCandidateSynthesizer()
    gate = SequenceSimilarityGate(scores)
    backend = SequenceWatermarkBackend(detections)
    storage = LocalStorage(tmp_path / "data")

    pipeline = SynthesisPipeline(
        profiles=profiles,
        sensitive_filter=type("Filter", (), {"check": lambda s, t: type("C", (), {"blocked": False})()})(),
        storage=storage,
        synthesizer=synthesizer,
        reference_resolver=StaticReferenceResolver(ref_audio),
        watermark=WatermarkService(backend=backend),
        fingerprint=FingerprintService(),
        similarity_gate=gate,
        max_candidates=max_candidates,
    )
    return pipeline, published_profile, synthesizer, storage


def test_pipeline_generates_three_candidates_and_selects_highest_similarity(tmp_path: Path):
    from hashlib import sha256
    job_id = "job_test_123"
    payload = int.from_bytes(sha256(job_id.encode("utf-8")).digest()[:2], "big")

    scores = [0.82, 0.93, 0.88]
    detections = [(0.95, payload), (0.95, payload), (0.95, payload)]

    pipeline, profile, synthesizer, storage = _setup_pipeline(tmp_path, scores, detections, max_candidates=3)

    req = SynthesisCreate(
        voice_profile_id=profile.id,
        text="合成测试文本。",
        text_lang="zh",
        emotion=EmotionControl(mode="auto"),
        consent_confirmed=True,
    )

    result = pipeline.run(job_id, req)
    assert result.status == "succeeded"
    assert result.candidate_count == 3
    assert result.selected_candidate_index == 1
    assert result.speaker_similarity == 0.93
    assert result.public_audio_path is not None
    assert result.public_audio_path.is_file()

    # Verify all temporary candidate files were cleaned up
    temp_dir = tmp_path / "data" / "temp" / "synthesis" / profile.id / job_id
    assert not temp_dir.exists() or list(temp_dir.glob("*.wav")) == []


def test_pipeline_skips_watermark_failed_candidate_when_selecting_best(tmp_path: Path):
    from hashlib import sha256
    job_id = "job_test_wm_fail"
    payload = int.from_bytes(sha256(job_id.encode("utf-8")).digest()[:2], "big")

    scores = [0.96, 0.89, 0.85]
    detections = [(0.70, payload), (0.92, payload), (0.91, payload)]

    pipeline, profile, synthesizer, storage = _setup_pipeline(tmp_path, scores, detections, max_candidates=3)

    req = SynthesisCreate(
        voice_profile_id=profile.id,
        text="合成测试文本。",
        text_lang="zh",
        emotion=EmotionControl(mode="auto"),
        consent_confirmed=True,
    )

    result = pipeline.run(job_id, req)
    assert result.status == "succeeded"
    assert result.candidate_count == 3
    assert result.selected_candidate_index == 1
    assert result.speaker_similarity == 0.89


def test_pipeline_reports_real_candidate_progress_stages(tmp_path: Path):
    from hashlib import sha256
    job_id = "job_test_progress"
    payload = int.from_bytes(sha256(job_id.encode("utf-8")).digest()[:2], "big")

    scores = [0.82, 0.93, 0.88]
    detections = [(0.95, payload), (0.95, payload), (0.95, payload)]

    pipeline, profile, synthesizer, storage = _setup_pipeline(tmp_path, scores, detections, max_candidates=3)

    req = SynthesisCreate(
        voice_profile_id=profile.id,
        text="合成测试文本。",
        text_lang="zh",
        emotion=EmotionControl(mode="auto"),
        consent_confirmed=True,
    )

    stages: list[str] = []
    result = pipeline.run(job_id, req, progress=stages.append)

    assert result.status == "succeeded"
    assert stages == [
        "正在生成候选 1/3...",
        "正在验证候选 1/3...",
        "正在生成候选 2/3...",
        "正在验证候选 2/3...",
        "正在生成候选 3/3...",
        "正在验证候选 3/3...",
        "正在择优选择最佳结果...",
    ]


def test_pipeline_fails_closed_when_all_candidates_fail_watermark(tmp_path: Path):
    job_id = "job_test_all_fail"
    payload = int.from_bytes(job_id.encode("utf-8")[:2], "big")

    # all 3 candidates fail watermark
    scores = [0.95, 0.94, 0.93]
    detections = [(0.60, payload), (0.65, payload), (0.70, payload)]

    pipeline, profile, synthesizer, storage = _setup_pipeline(tmp_path, scores, detections, max_candidates=3)

    req = SynthesisCreate(
        voice_profile_id=profile.id,
        text="合成测试文本。",
        text_lang="zh",
        emotion=EmotionControl(mode="auto"),
        consent_confirmed=True,
    )

    result = pipeline.run(job_id, req)
    assert result.status == "failed"
    assert result.error_code == "WATERMARK_VERIFICATION_FAILED"
    assert result.public_audio_path is None


def test_cloud_emotion_mode_uses_primary_reference_and_one_paid_candidate(tmp_path: Path):
    from hashlib import sha256

    job_id = "job_cloud_single"
    payload = int.from_bytes(sha256(job_id.encode("utf-8")).digest()[:2], "big")
    pipeline, profile, synthesizer, _storage = _setup_pipeline(
        tmp_path, [0.91], [(0.95, payload)], max_candidates=3
    )
    resolver = pipeline.reference_resolver
    request = SynthesisCreate.model_validate(
        {
            "voice_profile_id": profile.id,
            "text": "今天真开心。",
            "text_lang": "zh",
            "emotion": {"mode": "manual", "label": "happy"},
            "consent_confirmed": True,
            "synthesis_mode": "emotion_api",
            "cloud_processing_confirmed": True,
            "cloud_options": {"control_mode": "label", "label": "happy"},
        }
    )

    result = pipeline.run(job_id, request)

    assert result.status == "succeeded"
    assert result.candidate_count == 1
    assert len(synthesizer.synthesized_paths) == 1
    assert resolver.emotions[0].mode == "auto"
