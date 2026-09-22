from pathlib import Path
import pytest

from backend.app.db.models import DatasetSegment
from backend.app.db.session import create_database_engine, session_factory
from backend.app.schemas.common import EmotionLabel
from backend.app.schemas.synthesis import EmotionControl
from backend.app.services.storage import LocalStorage
from backend.app.services.voice_base_registry import ResolvedVoiceBase
from backend.app.services.voice_profiles import (
    VoiceProfileRecord,
    VoiceProfileStore,
)
from backend.app.services.synthesis_pipeline import (
    EmotionReferenceUnavailable,
    ZeroShotReferenceResolver,
    ZeroShotSynthesisReference,
)


ACTIVE_BASE_ID = "gpt-sovits-v2proplus-official"


def _store(engine=None) -> VoiceProfileStore:
    return VoiceProfileStore(engine=engine or create_database_engine("sqlite://"))


def _create_zero_shot(store: VoiceProfileStore, **overrides) -> VoiceProfileRecord:
    payload = {
        "display_name": "多参考音色",
        "dataset_id": "dataset-1",
        "owner_user_id": "user-1",
        "reference_asset_id": "asset-1",
        "reference_segment_id": "seg-1",
        "prompt_text": "主参考文本内容。",
        "prompt_language": "zh",
        "emotion_label": "neutral",
        "emotion_confidence": 0.95,
        "base_model_id": ACTIVE_BASE_ID,
    }
    payload.update(overrides)
    return store.create_zero_shot(**payload)


def test_add_reference_rejects_exceeding_five_references():
    store = _store()
    profile = _create_zero_shot(store)

    # 1 primary exists; add 4 auxiliary references -> total 5 references
    for i in range(2, 6):
        store.add_reference(
            profile.id,
            "user-1",
            asset_id=f"asset-{i}",
            segment_id=f"seg-{i}",
            prompt_text=f"辅助参考文本 {i}",
            prompt_language="zh",
            emotion_label="neutral",
            emotion_confidence=0.9,
        )

    refs = store.references(profile.id, "user-1")
    assert len(refs) == 5

    # 6th reference must fail
    with pytest.raises(ValueError, match="音色参考音频最多保存 5 段"):
        store.add_reference(
            profile.id,
            "user-1",
            asset_id="asset-6",
            segment_id="seg-6",
            prompt_text="超出上限参考文本",
            prompt_language="zh",
            emotion_label="neutral",
            emotion_confidence=0.9,
        )


def test_delete_auxiliary_reference_succeeds_and_prevents_deleting_primary():
    store = _store()
    profile = _create_zero_shot(store)

    aux_ref = store.add_reference(
        profile.id,
        "user-1",
        asset_id="asset-aux",
        segment_id="seg-aux",
        prompt_text="辅助参考文本",
        prompt_language="zh",
        emotion_label="happy",
        emotion_confidence=0.88,
    )

    refs = store.references(profile.id, "user-1")
    assert len(refs) == 2

    # Foreign user cannot delete
    with pytest.raises(KeyError):
        store.delete_reference(profile.id, aux_ref.id, "user-2")

    # Deleting primary reference is rejected
    primary_ref = next(r for r in refs if r.is_primary)
    with pytest.raises(ValueError, match="主参考音频不可删除"):
        store.delete_reference(profile.id, primary_ref.id, "user-1")

    # Deleting auxiliary reference succeeds
    store.delete_reference(profile.id, aux_ref.id, "user-1")
    remaining = store.references(profile.id, "user-1")
    assert len(remaining) == 1
    assert remaining[0].id == primary_ref.id


class FakeRegistry:
    def resolve(self, base_model_id: str) -> ResolvedVoiceBase:
        return ResolvedVoiceBase(
            id=base_model_id,
            source_tag="test",
            gpt_weight=Path("fake_gpt.ckpt"),
            sovits_weight=Path("fake_sovits.pth"),
        )


def _seed_segment(
    engine,
    dataset_id: str,
    segment_id: str,
    owner_user_id: str,
    snr_db: float,
    clipping_ratio: float = 0.0,
):
    with session_factory(engine)() as session:
        segment = DatasetSegment(
            dataset_id=dataset_id,
            segment_id=segment_id,
            owner_user_id=owner_user_id,
            order_index=0,
            relative_path=f"datasets/{dataset_id}/segments/{segment_id}.wav",
            duration_seconds=5.0,
            snr_db=snr_db,
            clipping_ratio=clipping_ratio,
        )
        session.add(segment)
        session.commit()


def test_zero_shot_reference_resolver_auto_emotion_uses_primary_and_up_to_three_auxiliary(tmp_path):
    engine = create_database_engine("sqlite://")
    store = _store(engine)
    profile = _create_zero_shot(store)

    storage = LocalStorage(tmp_path)
    storage.root.mkdir(parents=True, exist_ok=True)
    # create dummy audio files in storage
    seg_dir = storage.root / "datasets" / profile.dataset_id / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / "seg-1.wav").write_bytes(b"dummy1")
    _seed_segment(engine, profile.dataset_id, "seg-1", "user-1", snr_db=15.0)

    # Add 4 auxiliary references with varying SNR
    # seg-2: SNR 25.0 (best)
    # seg-3: SNR 20.0
    # seg-4: SNR 18.0
    # seg-5: SNR 12.0 (lowest)
    for idx, snr in [(2, 25.0), (3, 20.0), (4, 18.0), (5, 12.0)]:
        (seg_dir / f"seg-{idx}.wav").write_bytes(f"dummy{idx}".encode())
        _seed_segment(engine, profile.dataset_id, f"seg-{idx}", "user-1", snr_db=snr)
        store.add_reference(
            profile.id,
            "user-1",
            asset_id=f"asset-{idx}",
            segment_id=f"seg-{idx}",
            prompt_text=f"文本 {idx}",
            prompt_language="zh",
            emotion_label="neutral",
            emotion_confidence=0.9,
        )

    resolver = ZeroShotReferenceResolver(
        profiles=store,
        storage=storage,
        registry=FakeRegistry(),
    )

    resolved = resolver.resolve(profile, EmotionControl(mode="auto"))
    assert isinstance(resolved, ZeroShotSynthesisReference)
    # Primary audio must be the profile's primary reference
    assert resolved.reference_audio.name == "seg-1.wav"
    assert resolved.primary_audio_path.name == "seg-1.wav"
    # Auxiliary references must take top 3 by quality: seg-2 (25.0), seg-3 (20.0), seg-4 (18.0)
    aux_names = [p.name for p in resolved.auxiliary_audios]
    assert aux_names == ["seg-2.wav", "seg-3.wav", "seg-4.wav"]
    assert len(resolved.auxiliary_audios) == 3


def test_zero_shot_reference_resolver_manual_emotion_filters_same_emotion_and_ranks_best(tmp_path):
    engine = create_database_engine("sqlite://")
    store = _store(engine)
    profile = _create_zero_shot(store)  # primary is neutral

    storage = LocalStorage(tmp_path)
    seg_dir = storage.root / "datasets" / profile.dataset_id / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / "seg-1.wav").write_bytes(b"dummy1")
    _seed_segment(engine, profile.dataset_id, "seg-1", "user-1", snr_db=15.0)

    # Add happy references
    # happy-1: SNR 14.0
    # happy-2: SNR 22.0 (best)
    # happy-3: SNR 19.0
    # and one sad reference: SNR 30.0 (should not be selected for happy)
    (seg_dir / "seg-h1.wav").write_bytes(b"dummyh1")
    _seed_segment(engine, profile.dataset_id, "seg-h1", "user-1", snr_db=14.0)
    store.add_reference(
        profile.id,
        "user-1",
        asset_id="asset-h1",
        segment_id="seg-h1",
        prompt_text="开心文本1",
        prompt_language="zh",
        emotion_label="happy",
        emotion_confidence=0.9,
    )

    (seg_dir / "seg-h2.wav").write_bytes(b"dummyh2")
    _seed_segment(engine, profile.dataset_id, "seg-h2", "user-1", snr_db=22.0)
    store.add_reference(
        profile.id,
        "user-1",
        asset_id="asset-h2",
        segment_id="seg-h2",
        prompt_text="开心文本2",
        prompt_language="zh",
        emotion_label="happy",
        emotion_confidence=0.9,
    )

    (seg_dir / "seg-h3.wav").write_bytes(b"dummyh3")
    _seed_segment(engine, profile.dataset_id, "seg-h3", "user-1", snr_db=19.0)
    store.add_reference(
        profile.id,
        "user-1",
        asset_id="asset-h3",
        segment_id="seg-h3",
        prompt_text="开心文本3",
        prompt_language="zh",
        emotion_label="happy",
        emotion_confidence=0.9,
    )

    (seg_dir / "seg-sad.wav").write_bytes(b"dummysad")
    _seed_segment(engine, profile.dataset_id, "seg-sad", "user-1", snr_db=30.0)
    store.add_reference(
        profile.id,
        "user-1",
        asset_id="asset-sad",
        segment_id="seg-sad",
        prompt_text="难过文本",
        prompt_language="zh",
        emotion_label="sad",
        emotion_confidence=0.9,
    )

    resolver = ZeroShotReferenceResolver(
        profiles=store,
        storage=storage,
        registry=FakeRegistry(),
    )

    # When requesting happy emotion:
    resolved = resolver.resolve(profile, EmotionControl(mode="manual", label=EmotionLabel.HAPPY))
    # Best quality happy reference is seg-h2 (SNR 22.0)
    assert resolved.reference_audio.name == "seg-h2.wav"
    assert resolved.prompt_text == "开心文本2"
    assert resolved.emotion_label == "happy"
    # Primary audio path remains the original profile primary (seg-1.wav) for CAM++ similarity check
    assert resolved.primary_audio_path.name == "seg-1.wav"
    # Auxiliary references must only be other happy references: seg-h3 (19.0), seg-h1 (14.0)
    aux_names = [p.name for p in resolved.auxiliary_audios]
    assert aux_names == ["seg-h3.wav", "seg-h1.wav"]
    assert "seg-sad.wav" not in aux_names

    # Requesting angry emotion has no references -> EmotionReferenceUnavailable
    with pytest.raises(EmotionReferenceUnavailable):
        resolver.resolve(profile, EmotionControl(mode="manual", label=EmotionLabel.ANGRY))
