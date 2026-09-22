from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.db.models import UserProfile, utc_now
from backend.app.db.session import create_database_engine, init_db, session_factory
from backend.app.services.storage import LocalStorage

MAX_DISPLAY_NAME_LENGTH = 32
MAX_BIO_LENGTH = 200
MAX_AVATAR_BYTES = 2 * 1024 * 1024


class AvatarRejected(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class UserProfileRecord:
    user_id: str
    display_name: str
    bio: str | None
    avatar_path: str | None
    has_avatar: bool
    updated_at: datetime | None


def _avatar_suffix(content: bytes) -> str:
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if content.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return ".webp"
    raise AvatarRejected("AVATAR_FORMAT_UNSUPPORTED", "头像仅支持 PNG、JPEG 或 WebP 图片")


class UserProfileStore:
    """Persist per-user public profile data and validated local avatar files."""

    def __init__(
        self,
        engine: Engine | None = None,
        storage: LocalStorage | None = None,
    ) -> None:
        self.engine = engine or create_database_engine("sqlite:///data/app.db")
        init_db(self.engine)
        self._session_factory: sessionmaker[Session] = session_factory(self.engine)
        self.storage = storage or LocalStorage("data")

    @staticmethod
    def _record(row: UserProfile | None, user_id: str, fallback_username: str) -> UserProfileRecord:
        if row is None:
            return UserProfileRecord(
                user_id=user_id,
                display_name=fallback_username,
                bio=None,
                avatar_path=None,
                has_avatar=False,
                updated_at=None,
            )
        return UserProfileRecord(
            user_id=row.user_id,
            display_name=row.display_name,
            bio=row.bio,
            avatar_path=row.avatar_path,
            has_avatar=bool(row.avatar_path),
            updated_at=row.updated_at,
        )

    def get(self, user_id: str, *, fallback_username: str) -> UserProfileRecord:
        with self._session_factory() as session:
            row = session.get(UserProfile, user_id)
            return self._record(row, user_id, fallback_username)

    def update(self, user_id: str, *, display_name: str, bio: str | None) -> UserProfileRecord:
        normalized_name = (display_name or "").strip()
        if not 1 <= len(normalized_name) <= MAX_DISPLAY_NAME_LENGTH:
            raise ValueError(f"昵称需在 1–{MAX_DISPLAY_NAME_LENGTH} 个字符以内")
        normalized_bio = (bio or "").strip() or None
        if normalized_bio is not None and len(normalized_bio) > MAX_BIO_LENGTH:
            raise ValueError(f"简介最长 {MAX_BIO_LENGTH} 个字符")
        with self._session_factory() as session:
            row = session.get(UserProfile, user_id)
            if row is None:
                row = UserProfile(user_id=user_id, display_name=normalized_name)
                session.add(row)
            row.display_name = normalized_name
            row.bio = normalized_bio
            row.updated_at = utc_now()
            session.commit()
            return self._record(row, user_id, normalized_name)

    def set_avatar(self, user_id: str, content: bytes, *, fallback_username: str) -> UserProfileRecord:
        if len(content) > MAX_AVATAR_BYTES:
            raise AvatarRejected("AVATAR_TOO_LARGE", "头像文件需在 2MB 以内")
        suffix = _avatar_suffix(content)
        filename = f"{user_id}{suffix}"
        destination = self.storage.resolve("avatars", filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        with self._session_factory() as session:
            row = session.get(UserProfile, user_id)
            previous_path = row.avatar_path if row is not None else None
            if row is None:
                row = UserProfile(user_id=user_id, display_name=fallback_username)
                session.add(row)
            row.avatar_path = filename
            row.updated_at = utc_now()
            session.commit()
            if previous_path and previous_path != filename:
                try:
                    self.storage.resolve("avatars", previous_path).unlink()
                except FileNotFoundError:
                    pass
            return self._record(row, user_id, fallback_username)

    def avatar_file(self, user_id: str) -> Path | None:
        with self._session_factory() as session:
            row = session.get(UserProfile, user_id)
            if row is None or not row.avatar_path:
                return None
        path = self.storage.resolve("avatars", row.avatar_path)
        return path if path.is_file() else None
