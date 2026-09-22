import sqlite3

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy import text

from backend.app.db.session import create_database_engine, init_db, migrate_voice_base_id
from backend.app.services.voice_profiles import (
    VoiceProfileRecord,
    VoiceProfileStore,
)


LEGACY_BASE_ID = "gpt-sovits-v2pro-official"
ACTIVE_BASE_ID = "gpt-sovits-v2proplus-official"


def _store() -> VoiceProfileStore:
    return VoiceProfileStore(engine=create_database_engine("sqlite://"))


def _create_zero_shot(store: VoiceProfileStore, **overrides) -> VoiceProfileRecord:
    payload = {
        "display_name": "讲解音色",
        "dataset_id": "dataset-1",
        "owner_user_id": "user-1",
        "reference_asset_id": "asset-1",
        "reference_segment_id": "seg-1",
        "prompt_text": "这是确认后的参考文本。",
        "prompt_language": "zh",
        "emotion_label": "neutral",
        "emotion_confidence": 0.91,
        "base_model_id": ACTIVE_BASE_ID,
    }
    payload.update(overrides)
    return store.create_zero_shot(**payload)


def test_create_zero_shot_persists_ready_profile_and_primary_reference():
    store = _store()

    profile = _create_zero_shot(store)

    assert profile.status == "ready"
    assert profile.mode == "zero_shot"
    assert profile.public_weight_dir is None
    assert profile.reference_asset_id == "asset-1"
    assert profile.reference_segment_id == "seg-1"
    assert profile.prompt_text == "这是确认后的参考文本。"
    assert profile.prompt_language == "zh"
    assert profile.base_model_id == ACTIVE_BASE_ID

    references = store.references(profile.id, "user-1")
    assert len(references) == 1
    primary = references[0]
    assert primary.is_primary is True
    assert primary.asset_id == "asset-1"
    assert primary.segment_id == "seg-1"
    assert primary.prompt_text == "这是确认后的参考文本。"
    assert primary.prompt_language == "zh"
    assert primary.emotion_label == "neutral"
    assert primary.emotion_confidence == 0.91


def test_create_zero_shot_persists_trimmed_display_name():
    store = _store()

    profile = _create_zero_shot(store, display_name="  讲解音色  ")

    assert profile.display_name == "讲解音色"
    assert store.get(profile.id, "user-1").display_name == "讲解音色"


def test_legacy_profile_rows_keep_null_display_name():
    store = _store()

    profile = store.create("dataset-legacy", owner_user_id="user-1")

    assert profile.display_name is None


def test_zero_shot_profile_is_invisible_to_other_owners():
    store = _store()
    profile = _create_zero_shot(store)

    with pytest.raises(KeyError):
        store.get(profile.id, "user-2")
    with pytest.raises(KeyError):
        store.references(profile.id, "user-2")


def test_create_zero_shot_rejects_empty_prompt_and_unknown_language():
    store = _store()

    with pytest.raises(ValueError):
        _create_zero_shot(store, prompt_text="   ")
    with pytest.raises(ValueError):
        _create_zero_shot(store, prompt_language="jp")


def test_create_zero_shot_rejects_unknown_emotion_label():
    store = _store()

    with pytest.raises(ValueError):
        _create_zero_shot(store, emotion_label="excited")


def test_added_emotion_reference_is_non_primary_and_keeps_single_primary():
    store = _store()
    profile = _create_zero_shot(store)

    added = store.add_reference(
        profile.id,
        "user-1",
        asset_id="asset-2",
        segment_id="seg-2",
        prompt_text="生气的参考文本。",
        prompt_language="zh",
        emotion_label="angry",
        emotion_confidence=None,
    )

    references = store.references(profile.id, "user-1")
    assert len(references) == 2
    assert added.is_primary is False
    assert added.emotion_label == "angry"
    assert added.emotion_confidence is None
    assert sum(1 for reference in references if reference.is_primary) == 1

    with pytest.raises(IntegrityError):
        with store.engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO voice_references "
                    "(id, profile_id, owner_user_id, asset_id, segment_id, prompt_text, "
                    "prompt_language, emotion_label, is_primary, created_at) "
                    "VALUES ('ref-x', :profile_id, 'user-1', 'asset-3', 'seg-3', 't', "
                    "'zh', 'sad', 1, CURRENT_TIMESTAMP)"
                ),
                {"profile_id": profile.id},
            )


def test_add_reference_rejects_foreign_or_missing_profile():
    store = _store()
    profile = _create_zero_shot(store)

    with pytest.raises(KeyError):
        store.add_reference(
            profile.id,
            "user-2",
            asset_id="asset-2",
            segment_id="seg-2",
            prompt_text="其他用户的参考。",
            prompt_language="zh",
            emotion_label="sad",
            emotion_confidence=0.7,
        )
    with pytest.raises(KeyError):
        store.add_reference(
            "missing-profile",
            "user-1",
            asset_id="asset-2",
            segment_id="seg-2",
            prompt_text="不存在的档案。",
            prompt_language="zh",
            emotion_label="sad",
            emotion_confidence=0.7,
        )


def test_add_reference_never_changes_base_model_id():
    store = _store()
    profile = _create_zero_shot(store)

    store.add_reference(
        profile.id,
        "user-1",
        asset_id="asset-2",
        segment_id="seg-2",
        prompt_text="英文参考。",
        prompt_language="en",
        emotion_label="happy",
        emotion_confidence=0.66,
    )

    assert store.get(profile.id, "user-1").base_model_id == ACTIVE_BASE_ID


def test_migrate_voice_base_id_updates_only_legacy_zero_shot_profiles_idempotently():
    store = _store()
    profile = _create_zero_shot(store, base_model_id=LEGACY_BASE_ID)
    legacy_training_profile = store.create("dataset-legacy", owner_user_id="user-1")

    assert migrate_voice_base_id(store.engine, LEGACY_BASE_ID, ACTIVE_BASE_ID) == 1
    assert migrate_voice_base_id(store.engine, LEGACY_BASE_ID, ACTIVE_BASE_ID) == 0
    assert store.get(profile.id, "user-1").base_model_id == ACTIVE_BASE_ID
    assert store.get(legacy_training_profile.id, "user-1").base_model_id is None


def test_legacy_sqlite_database_gains_zero_shot_columns_without_losing_rows(tmp_path):
    db_path = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE users (
            id VARCHAR(64) PRIMARY KEY,
            username VARCHAR(64),
            password_hash TEXT,
            password_salt VARCHAR(64)
        );
        CREATE TABLE sessions (
            id VARCHAR(64) PRIMARY KEY,
            token_hash VARCHAR(64),
            user_id VARCHAR(64),
            expires_at DATETIME
        );
        CREATE TABLE voice_profiles (
            id VARCHAR(64) PRIMARY KEY,
            status VARCHAR(32),
            public_weight_dir TEXT,
            created_at DATETIME
        );
        INSERT INTO users (id, username, password_hash, password_salt)
            VALUES ('user-1', 'alice', 'hash', 'salt');
        INSERT INTO sessions (id, token_hash, user_id, expires_at)
            VALUES ('session-1', 'token', 'user-1', '2026-12-31 00:00:00');
        INSERT INTO voice_profiles (id, status, public_weight_dir, created_at)
            VALUES ('profile-1', 'ready', NULL, '2026-09-01 00:00:00');
        """
    )
    connection.commit()
    connection.close()

    engine = create_database_engine(f"sqlite+pysqlite:///{db_path.as_posix()}")
    init_db(engine)

    with engine.connect() as raw:
        assert raw.execute(text("SELECT COUNT(*) FROM users")).scalar_one() == 1
        assert raw.execute(text("SELECT COUNT(*) FROM sessions")).scalar_one() == 1
        assert raw.execute(text("SELECT COUNT(*) FROM voice_profiles")).scalar_one() == 1
        profile_columns = {
            row[1] for row in raw.execute(text("PRAGMA table_info(voice_profiles)"))
        }
    assert {
        "mode",
        "reference_asset_id",
        "reference_segment_id",
        "prompt_text",
        "prompt_language",
        "base_model_id",
    } <= profile_columns

    store = VoiceProfileStore(engine=engine)
    profile = _create_zero_shot(store)
    assert profile.status == "ready"
    assert len(store.references(profile.id, "user-1")) == 1
