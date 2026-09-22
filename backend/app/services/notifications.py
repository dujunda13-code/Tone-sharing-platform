from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.db.models import Notification, PlazaComment, PlazaPost, User, UserProfile, VoiceProfile
from backend.app.db.session import create_database_engine, init_db, session_factory
from backend.app.services.voice_profiles import public_voice_name

VALID_TYPES = {"like", "comment"}
EXCERPT_LENGTH = 50


@dataclass(frozen=True)
class NotificationRecord:
    id: str
    recipient_user_id: str
    actor_user_id: str
    type: str
    post_id: str
    comment_id: str | None
    is_read: bool
    created_at: datetime


@dataclass(frozen=True)
class NotificationView(NotificationRecord):
    actor_display_name: str = ""
    voice_name: str = ""
    comment_excerpt: str | None = None


class NotificationStore:
    """Persist recipient-scoped plaza notifications in the local database."""

    def __init__(self, engine: Engine | None = None) -> None:
        self.engine = engine or create_database_engine("sqlite:///data/app.db")
        init_db(self.engine)
        self._session_factory: sessionmaker[Session] = session_factory(self.engine)

    @staticmethod
    def _record(row: Notification) -> NotificationRecord:
        return NotificationRecord(
            id=row.id,
            recipient_user_id=row.recipient_user_id,
            actor_user_id=row.actor_user_id,
            type=row.type,
            post_id=row.post_id,
            comment_id=row.comment_id,
            is_read=bool(row.is_read),
            created_at=row.created_at,
        )

    def create(
        self,
        recipient_user_id: str,
        actor_user_id: str,
        *,
        type: str,
        post_id: str,
        comment_id: str | None = None,
    ) -> NotificationRecord | None:
        if type not in VALID_TYPES:
            raise ValueError(f"通知类型只允许 {'、'.join(sorted(VALID_TYPES))}")
        if recipient_user_id == actor_user_id:
            return None
        with self._session_factory() as session:
            row = Notification(
                id=uuid4().hex,
                recipient_user_id=recipient_user_id,
                actor_user_id=actor_user_id,
                type=type,
                post_id=post_id,
                comment_id=comment_id,
                is_read=False,
            )
            session.add(row)
            session.commit()
            return self._record(row)

    def list_views(
        self, recipient_user_id: str, *, limit: int = 20, offset: int = 0
    ) -> tuple[list[NotificationView], int]:
        with self._session_factory() as session:
            total = int(
                session.scalar(
                    select(func.count(Notification.id)).where(
                        Notification.recipient_user_id == recipient_user_id
                    )
                )
                or 0
            )
            rows = session.scalars(
                select(Notification)
                .where(Notification.recipient_user_id == recipient_user_id)
                .order_by(Notification.created_at.desc(), Notification.id.desc())
                .limit(limit)
                .offset(offset)
            ).all()
            views: list[NotificationView] = []
            for row in rows:
                actor_name = session.scalar(
                    select(User.username).where(User.id == row.actor_user_id)
                )
                actor_profile = session.get(UserProfile, row.actor_user_id)
                if actor_profile is not None:
                    actor_name = actor_profile.display_name
                post = session.get(PlazaPost, row.post_id)
                voice_name = ""
                if post is not None:
                    profile = session.get(VoiceProfile, post.voice_profile_id)
                    if profile is not None:
                        voice_name = public_voice_name(profile.id, profile.display_name)
                excerpt: str | None = None
                if row.comment_id:
                    comment = session.get(PlazaComment, row.comment_id)
                    if comment is not None:
                        excerpt = comment.content[:EXCERPT_LENGTH]
                views.append(
                    NotificationView(
                        id=row.id,
                        recipient_user_id=row.recipient_user_id,
                        actor_user_id=row.actor_user_id,
                        type=row.type,
                        post_id=row.post_id,
                        comment_id=row.comment_id,
                        is_read=bool(row.is_read),
                        created_at=row.created_at,
                        actor_display_name=actor_name or row.actor_user_id,
                        voice_name=voice_name,
                        comment_excerpt=excerpt,
                    )
                )
            return views, total

    def unread_count(self, recipient_user_id: str) -> int:
        with self._session_factory() as session:
            return int(
                session.scalar(
                    select(func.count(Notification.id)).where(
                        Notification.recipient_user_id == recipient_user_id,
                        Notification.is_read.is_(False),
                    )
                )
                or 0
            )

    def mark_read(
        self,
        recipient_user_id: str,
        *,
        notification_ids: list[str] | None = None,
        all: bool = False,
    ) -> int:
        with self._session_factory() as session:
            stmt = select(Notification).where(
                Notification.recipient_user_id == recipient_user_id
            )
            if not all:
                if not notification_ids:
                    return 0
                stmt = stmt.where(Notification.id.in_(notification_ids))
            rows = session.scalars(stmt).all()
            for row in rows:
                row.is_read = True
            session.commit()
            return len(rows)

    def delete_for_post(self, post_id: str) -> None:
        with self._session_factory() as session:
            rows = session.scalars(
                select(Notification).where(Notification.post_id == post_id)
            ).all()
            for row in rows:
                session.delete(row)
            session.commit()
