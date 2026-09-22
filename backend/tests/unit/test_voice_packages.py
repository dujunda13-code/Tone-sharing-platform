"""VoicePackageService: ZIP contents and decoupled import forks."""

import io
import json
import zipfile
from pathlib import Path

from sqlalchemy import create_engine

from backend.app.db.models import (
    AudioAsset,
    Dataset,
    DatasetSegment,
    PlazaPost,
    VoiceProfile,
    VoiceReference,
    utc_now,
)
from backend.app.db.session import init_db, session_factory
from backend.app.services.plaza import PlazaError
from backend.app.services.storage import LocalStorage
from backend.app.services.voice_packages import VoicePackageService
from backend.app.services.voice_profiles import VoiceProfileStore

PRIMARY_BYTES = b"RIFF-primary-wav-bytes"
AUX_BYTES = b"RIFF-auxiliary-wav-bytes"


def _setup(tmp_path: Path):
    engine = create_engine(f"sqlite+pysqlite:///{(tmp_path / 'pkg.db').as_posix()}")
    init_db(engine)
    storage = LocalStorage(tmp_path)
    profiles = VoiceProfileStore(engine=engine)
    service = VoicePackageService(engine=engine, storage=storage, profiles=profiles)
    with session_factory(engine)() as session:
        session.add(
            Dataset(
                id="ds1",
                owner_user_id="alice",
                authorization_confirmed_at=utc_now(),
                effective_seconds=6.0,
                status="ready_for_profile",
            )
        )
        session.add(
            VoiceProfile(
                id="vp1",
                owner_user_id="alice",
                dataset_id="ds1",
                status="ready",
                mode="zero_shot",
                base_model_id="gpt-sovits-v2proplus-official",
                display_name="可下载音色",
            )
        )
        for segment_id, content, is_primary, emotion in [
            ("seg_0001", PRIMARY_BYTES, True, "neutral"),
            ("seg_0002", AUX_BYTES, False, "happy"),
        ]:
            relative = f"datasets/ds1/segments/{segment_id}.wav"
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            session.add(
                DatasetSegment(
                    dataset_id="ds1",
                    segment_id=segment_id,
                    owner_user_id="alice",
                    order_index=0 if is_primary else 1,
                    relative_path=relative,
                    duration_seconds=3.0,
                    snr_db=27.0,
                    clipping_ratio=0.0,
                    split="reference",
                    language="zh",
                    auto_transcript=f"转写{segment_id}",
                    auto_emotion_label=emotion,
                    auto_emotion_confidence=0.9,
                    reviewed_at=utc_now(),
                )
            )
            session.add(
                AudioAsset(
                    id=f"asset-{segment_id}",
                    owner_user_id="alice",
                    dataset_id="ds1",
                    path=relative,
                    duration_seconds=3.0,
                )
            )
            session.add(
                VoiceReference(
                    id=f"ref-{segment_id}",
                    profile_id="vp1",
                    owner_user_id="alice",
                    asset_id=f"asset-{segment_id}",
                    segment_id=segment_id,
                    prompt_text=f"转写{segment_id}",
                    prompt_language="zh",
                    emotion_label=emotion,
                    emotion_confidence=0.9,
                    is_primary=is_primary,
                )
            )
        session.add(PlazaPost(id="p1", author_user_id="alice", voice_profile_id="vp1"))
        session.commit()
    return engine, tmp_path, service, profiles


def test_build_package_zip_contents(tmp_path):
    engine, root, service, _ = _setup(tmp_path)

    package = service.build_package("p1")

    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        names = archive.namelist()
        assert "metadata.json" in names
        assert "references/primary-seg_0001.wav" in names
        assert "references/auxiliary-seg_0002.wav" in names
        assert archive.read("references/primary-seg_0001.wav") == PRIMARY_BYTES
        metadata = json.loads(archive.read("metadata.json"))
    assert metadata["voice"]["display_name"] == "可下载音色"
    assert metadata["voice"]["base_model_id"] == "gpt-sovits-v2proplus-official"
    assert metadata["post"]["post_id"] == "p1"
    assert len(metadata["references"]) == 2
    assert metadata["references"][0]["is_primary"] is True
    assert metadata["references"][0]["prompt_text"] == "转写seg_0001"


def test_build_package_fails_when_reference_file_missing(tmp_path):
    engine, root, service, _ = _setup(tmp_path)
    (root / "datasets/ds1/segments/seg_0001.wav").unlink()

    try:
        service.build_package("p1")
    except PlazaError as exc:
        assert exc.code == "PLAZA_REFERENCE_AUDIO_MISSING"
    else:
        raise AssertionError("missing audio must fail closed")

    try:
        service.reference_wav_path("p1")
    except PlazaError as exc:
        assert exc.code == "PLAZA_REFERENCE_AUDIO_MISSING"
    else:
        raise AssertionError("missing audio must fail closed")


def test_reference_wav_path_returns_primary(tmp_path):
    engine, root, service, _ = _setup(tmp_path)

    assert service.reference_wav_path("p1").read_bytes() == PRIMARY_BYTES


def test_import_fork_requires_authorization(tmp_path):
    engine, root, service, _ = _setup(tmp_path)

    try:
        service.import_fork(
            "p1", recipient_user_id="bob", authorization_confirmed=False, fallback_username="bob"
        )
    except PlazaError as exc:
        assert exc.code == "AUTHORIZATION_REQUIRED"
    else:
        raise AssertionError("import without confirmation must be rejected")


def test_import_fork_copies_chain_and_decouples_from_post(tmp_path):
    engine, root, service, profiles = _setup(tmp_path)

    forked = service.import_fork(
        "p1", recipient_user_id="bob", authorization_confirmed=True, fallback_username="bob"
    )

    assert forked.owner_user_id == "bob"
    assert forked.status == "ready"
    references = profiles.references(forked.id, "bob")
    assert len(references) == 2
    assert sum(1 for r in references if r.is_primary) == 1
    forked_dataset_id = forked.dataset_id
    primary = next(r for r in references if r.is_primary)
    copied = (
        root / "datasets" / forked_dataset_id / "segments" / f"{primary.segment_id}.wav"
    )
    assert copied.read_bytes() == PRIMARY_BYTES
    with session_factory(engine)() as session:
        segment = session.get(
            DatasetSegment, {"dataset_id": forked_dataset_id, "segment_id": primary.segment_id}
        )
        assert segment is not None and segment.owner_user_id == "bob"
        assert segment.auto_transcript == "转写seg_0001"

    # Decoupling: deleting the original post leaves the fork fully usable.
    with session_factory(engine)() as session:
        post = session.get(PlazaPost, "p1")
        session.delete(post)
        session.commit()
    assert len(profiles.references(forked.id, "bob")) == 2
    assert copied.read_bytes() == PRIMARY_BYTES


def test_import_fork_rejects_not_ready_profile(tmp_path):
    engine, root, service, _ = _setup(tmp_path)
    with session_factory(engine)() as session:
        profile = session.get(VoiceProfile, "vp1")
        profile.status = "created"
        session.commit()

    try:
        service.import_fork(
            "p1", recipient_user_id="bob", authorization_confirmed=True, fallback_username="bob"
        )
    except PlazaError as exc:
        assert exc.code == "PLAZA_VOICE_PROFILE_NOT_READY"
    else:
        raise AssertionError("not-ready profile must be rejected")
