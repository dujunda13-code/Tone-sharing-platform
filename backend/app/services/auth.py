from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import re
import secrets
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.engine import Engine

from backend.app.db.models import AudioAsset, Dataset, Job, Synthesis, User, UserSession, VoiceProfile
from backend.app.db.session import create_database_engine, init_db, session_factory
from backend.app.schemas.auth import PasswordRecord, SessionRecord, UserRecord


class AuthError(ValueError):
    """Base class for local authentication errors."""


class UsernameTaken(AuthError):
    """Raised when a normalized username already exists."""


class InvalidCredentials(AuthError):
    """Raised without revealing whether a username exists."""


class InvalidUsername(AuthError):
    """Raised when a username violates the local account contract."""


class WeakPassword(AuthError):
    """Raised when a password is too short for local account creation."""


class AdminAlreadyInitialized(AuthError):
    """Raised when the one-time local administrator already exists."""


class PasswordHasher:
    _PARAMS = {"n": 2**14, "r": 8, "p": 1, "dklen": 32}

    @classmethod
    def hash(cls, password: str) -> PasswordRecord:
        if len(password) < 8:
            raise WeakPassword("password must contain at least 8 characters")
        salt = secrets.token_bytes(16)
        digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, **cls._PARAMS)
        return PasswordRecord(
            password_hash=digest.hex(),
            password_salt=salt.hex(),
            password_params_json=json.dumps(cls._PARAMS, sort_keys=True),
        )

    @staticmethod
    def verify(password: str, record: PasswordRecord) -> bool:
        try:
            params = json.loads(record.password_params_json)
            salt = bytes.fromhex(record.password_salt)
            digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, **params)
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        return hmac.compare_digest(digest.hex(), record.password_hash)


class AuthService:
    def __init__(self, engine: Engine | None = None, session_ttl: timedelta = timedelta(hours=8)) -> None:
        engine = engine or create_database_engine("sqlite:///data/app.db")
        init_db(engine)
        self.engine = engine
        self.session_ttl = session_ttl
        self.hasher = PasswordHasher()
        self._session_factory = session_factory(engine)

    @staticmethod
    def normalize_username(username: str) -> str:
        normalized = username.strip().casefold()
        if not re.fullmatch(r"[a-z0-9_\-\.]{3,64}", normalized):
            raise InvalidUsername("username must contain 3-64 ASCII letters, digits, _, -, or .")
        return normalized

    @staticmethod
    def _user_record(user: User) -> UserRecord:
        return UserRecord(
            id=user.id,
            username=user.username,
            role=user.role,
            status=user.status,
            created_at=user.created_at,
        )

    @staticmethod
    def _session_record(session: UserSession) -> SessionRecord:
        return SessionRecord(
            id=session.id,
            user_id=session.user_id,
            expires_at=session.expires_at,
            created_at=session.created_at,
            last_seen_at=session.last_seen_at,
        )

    def register(self, username: str, password: str, role: str = "user") -> UserRecord:
        normalized = self.normalize_username(username)
        if role not in {"user", "admin"}:
            raise ValueError("unsupported local account role")
        password_record = self.hasher.hash(password)
        with self._session_factory() as db_session:
            existing = db_session.scalar(select(User).where(User.username == normalized))
            if existing is not None:
                raise UsernameTaken(normalized)
            user = User(
                id=uuid4().hex,
                username=normalized,
                password_hash=password_record.password_hash,
                password_salt=password_record.password_salt,
                password_params_json=password_record.password_params_json,
                role=role,
                status="active",
            )
            db_session.add(user)
            db_session.commit()
            return self._user_record(user)

    def bootstrap_admin(self, username: str, password: str) -> UserRecord:
        normalized = self.normalize_username(username)
        password_record = self.hasher.hash(password)
        with self._session_factory() as db_session:
            if db_session.scalar(select(User.id).where(User.role == "admin").limit(1)) is not None:
                raise AdminAlreadyInitialized("local admin is already initialized")
            if db_session.scalar(select(User.id).where(User.username == normalized)) is not None:
                raise UsernameTaken(normalized)
            user = User(
                id=uuid4().hex,
                username=normalized,
                password_hash=password_record.password_hash,
                password_salt=password_record.password_salt,
                password_params_json=password_record.password_params_json,
                role="admin",
                status="active",
            )
            db_session.add(user)
            db_session.flush()
            for model in (Dataset, AudioAsset, VoiceProfile, Job, Synthesis):
                db_session.execute(
                    update(model)
                    .where(model.owner_user_id.is_(None))
                    .values(owner_user_id=user.id)
                )
            db_session.commit()
            return self._user_record(user)

    def authenticate(self, username: str, password: str) -> UserRecord:
        normalized = self.normalize_username(username)
        with self._session_factory() as db_session:
            user = db_session.scalar(select(User).where(User.username == normalized))
            if user is None or user.status != "active":
                raise InvalidCredentials("invalid username or password")
            record = PasswordRecord(user.password_hash, user.password_salt, user.password_params_json)
            if not self.hasher.verify(password, record):
                raise InvalidCredentials("invalid username or password")
            return self._user_record(user)

    def create_session(self, user_id: str) -> tuple[str, SessionRecord]:
        now = datetime.now(timezone.utc)
        raw_token = secrets.token_urlsafe(32)
        session = UserSession(
            id=uuid4().hex,
            token_hash=hashlib.sha256(raw_token.encode("utf-8")).hexdigest(),
            user_id=user_id,
            expires_at=now + self.session_ttl,
            created_at=now,
            last_seen_at=now,
        )
        with self._session_factory() as db_session:
            user = db_session.get(User, user_id)
            if user is None or user.status != "active":
                raise InvalidCredentials("user is not active")
            db_session.add(session)
            db_session.commit()
            return raw_token, self._session_record(session)

    def resolve_session(self, raw_token: str) -> UserRecord | None:
        token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        now = datetime.now(timezone.utc)
        with self._session_factory() as db_session:
            session = db_session.scalar(select(UserSession).where(UserSession.token_hash == token_hash))
            if session is None or session.revoked_at is not None:
                return None
            expires_at = session.expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at <= now:
                return None
            user = db_session.get(User, session.user_id)
            if user is None or user.status != "active":
                return None
            session.last_seen_at = now
            db_session.commit()
            return self._user_record(user)

    def revoke_session(self, raw_token: str) -> None:
        token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        with self._session_factory() as db_session:
            session = db_session.scalar(select(UserSession).where(UserSession.token_hash == token_hash))
            if session is not None and session.revoked_at is None:
                session.revoked_at = datetime.now(timezone.utc)
                db_session.commit()
