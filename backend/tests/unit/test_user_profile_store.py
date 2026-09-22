"""UserProfileStore: display-name/bio rules and validated local avatar storage."""

from pathlib import Path

from sqlalchemy import create_engine

from backend.app.db.session import init_db
from backend.app.services.storage import LocalStorage
from backend.app.services.user_profiles import AvatarRejected, UserProfileStore

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 64
JPEG_BYTES = b"\xff\xd8\xff" + b"0" * 64
WEBP_BYTES = b"RIFF\x24\x00\x00\x00WEBP" + b"0" * 64


def _store(tmp_path: Path) -> UserProfileStore:
    engine = create_engine(f"sqlite+pysqlite:///{(tmp_path / 'profiles.db').as_posix()}")
    init_db(engine)
    return UserProfileStore(engine=engine, storage=LocalStorage(tmp_path))


def test_get_returns_username_fallback_when_profile_missing(tmp_path):
    store = _store(tmp_path)
    record = store.get("u1", fallback_username="alice")

    assert record.display_name == "alice"
    assert record.bio is None
    assert record.has_avatar is False


def test_update_persists_and_validates_lengths(tmp_path):
    store = _store(tmp_path)
    updated = store.update("u1", display_name="昵称", bio="一句话简介")

    assert store.get("u1", fallback_username="alice").display_name == "昵称"
    assert updated.bio == "一句话简介"

    for bad in [("x" * 33, None), ("", None), ("ok", "y" * 201)]:
        try:
            store.update("u1", display_name=bad[0], bio=bad[1])
        except ValueError:
            pass
        else:
            raise AssertionError(f"must reject {bad!r}")


def test_set_avatar_accepts_png_jpeg_webp_and_writes_file(tmp_path):
    store = _store(tmp_path)
    for user, content, suffix in [
        ("u1", PNG_BYTES, ".png"),
        ("u2", JPEG_BYTES, ".jpg"),
        ("u3", WEBP_BYTES, ".webp"),
    ]:
        record = store.set_avatar(user, content, fallback_username=user)

        assert record.has_avatar is True
        assert (tmp_path / "avatars" / f"{user}{suffix}").read_bytes() == content
        assert store.avatar_file(user) is not None
        assert store.avatar_file(user).read_bytes() == content


def test_set_avatar_rejects_oversized_and_unknown_formats(tmp_path):
    store = _store(tmp_path)
    try:
        store.set_avatar("u1", b"x" * (2 * 1024 * 1024 + 1), fallback_username="u1")
    except AvatarRejected as exc:
        assert exc.code == "AVATAR_TOO_LARGE"
    else:
        raise AssertionError("oversized avatar must be rejected")

    try:
        store.set_avatar("u1", b"not-an-image-at-all", fallback_username="u1")
    except AvatarRejected as exc:
        assert exc.code == "AVATAR_FORMAT_UNSUPPORTED"
    else:
        raise AssertionError("unknown format must be rejected")


def test_set_avatar_replaces_previous_file(tmp_path):
    store = _store(tmp_path)
    store.set_avatar("u1", PNG_BYTES, fallback_username="u1")
    store.set_avatar("u1", JPEG_BYTES, fallback_username="u1")

    assert store.avatar_file("u1").name == "u1.jpg"
    assert not (tmp_path / "avatars" / "u1.png").exists()


def test_avatar_file_returns_none_without_avatar(tmp_path):
    store = _store(tmp_path)
    assert store.avatar_file("nope") is None
