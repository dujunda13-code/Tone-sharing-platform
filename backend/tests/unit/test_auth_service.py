from datetime import timedelta

import pytest
from sqlalchemy import select

from backend.app.db.models import AudioAsset, Dataset, Job, Synthesis, VoiceProfile
from backend.app.db.session import create_database_engine, session_factory
from backend.app.services.auth import AdminAlreadyInitialized, AuthService, UsernameTaken, WeakPassword


@pytest.fixture
def auth_service():
    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    return AuthService(engine=engine, session_ttl=timedelta(hours=2))


def test_password_hash_is_not_plaintext_and_verifies(auth_service):
    record = auth_service.hasher.hash("Correct-Horse-42")

    assert record.password_hash != "Correct-Horse-42"
    assert auth_service.hasher.verify("Correct-Horse-42", record) is True
    assert auth_service.hasher.verify("wrong", record) is False


def test_local_password_policy_accepts_eight_and_rejects_seven_characters(auth_service):
    accepted = auth_service.hasher.hash("eight888")

    assert auth_service.hasher.verify("eight888", accepted) is True
    with pytest.raises(WeakPassword):
        auth_service.hasher.hash("seven77")


def test_register_normalizes_username_and_rejects_duplicate(auth_service):
    first = auth_service.register(" Alice ", "Correct-Horse-42")

    assert first.username == "alice"
    with pytest.raises(UsernameTaken):
        auth_service.register("ALICE", "Another-Password-42")


def test_session_token_is_opaque_and_revocation_is_effective(auth_service):
    user = auth_service.register("alice", "Correct-Horse-42")
    raw_token, _ = auth_service.create_session(user.id)

    resolved = auth_service.resolve_session(raw_token)
    assert resolved is not None
    assert resolved.id == user.id
    auth_service.revoke_session(raw_token)
    assert auth_service.resolve_session(raw_token) is None


def test_bootstrap_admin_claims_unowned_legacy_rows_once(auth_service):
    auth_service.register("existing-user", "Correct-Horse-42")
    with session_factory(auth_service.engine)() as session:
        session.add_all(
            [
                Dataset(id="legacy-dataset"),
                AudioAsset(id="legacy-asset", dataset_id="legacy-dataset", path="uploads/a.wav"),
                VoiceProfile(id="legacy-profile", dataset_id="legacy-dataset"),
                Job(
                    id="legacy-job",
                    kind="train",
                    status="queued",
                    queue_order=1,
                    payload_json="{}",
                ),
                Synthesis(id="legacy-synthesis", job_id="legacy-job"),
            ]
        )
        session.commit()

    admin = auth_service.bootstrap_admin("admin", "Admin-Horse-42")

    with session_factory(auth_service.engine)() as session:
        for model in (Dataset, AudioAsset, VoiceProfile, Job, Synthesis):
            rows = session.scalars(select(model)).all()
            assert rows
            assert {row.owner_user_id for row in rows} == {admin.id}

    with pytest.raises(AdminAlreadyInitialized):
        auth_service.bootstrap_admin("other-admin", "Other-Horse-42")
