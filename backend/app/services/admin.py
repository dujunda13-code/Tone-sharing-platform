from __future__ import annotations

import json
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from backend.app.db.models import AuditEvent, Dataset, Job, User, VoiceProfile, utc_now
from backend.app.db.session import session_factory
from backend.app.schemas.admin import (
    AdminUserListResponse,
    AdminUserSummary,
    AuditEventListResponse,
    AuditEventSummary,
)


class AdminUserStatusError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class LocalAdminService:
    """Manage local account state and display only sanitized local audit summaries."""

    def __init__(self, *, engine: Engine) -> None:
        self._session_factory = session_factory(engine)

    def list_users(self) -> AdminUserListResponse:
        with self._session_factory() as session:
            users = session.scalars(
                select(User).order_by(User.created_at.desc(), User.id.desc())
            ).all()
            return AdminUserListResponse(
                items=self._user_summaries(session, users)
            )

    def set_user_status(
        self, actor_id: str, user_id: str, status: str
    ) -> AdminUserSummary:
        if actor_id == user_id:
            raise AdminUserStatusError(
                "ADMIN_SELF_STATUS_CHANGE_FORBIDDEN", "管理员不能修改自己的状态"
            )
        with self._session_factory() as session:
            user = session.get(User, user_id)
            if user is None:
                raise AdminUserStatusError("USER_NOT_FOUND", "本地用户不存在")
            old_status = user.status
            if old_status != status:
                user.status = status
                user.updated_at = utc_now()
                session.add(
                    AuditEvent(
                        event_type="admin.user_status_changed",
                        subject_id=user.id,
                        metadata_json=json.dumps(
                            {
                                "old_status": old_status,
                                "new_status": status,
                                "actor_id": actor_id,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    )
                )
                session.commit()
            return self._user_summaries(session, [user])[0]

    def list_audit_events(
        self, limit: int, before_id: int | None = None
    ) -> AuditEventListResponse:
        bounded_limit = min(max(int(limit), 1), 100)
        with self._session_factory() as session:
            statement = select(AuditEvent).order_by(AuditEvent.id.desc())
            if before_id is not None:
                statement = statement.where(AuditEvent.id < before_id)
            events = session.scalars(statement.limit(bounded_limit + 1)).all()
            visible_events = events[:bounded_limit]
            next_before_id = (
                visible_events[-1].id if len(events) > bounded_limit else None
            )
            return AuditEventListResponse(
                items=[self._audit_summary(event) for event in visible_events],
                next_before_id=next_before_id,
            )

    @staticmethod
    def _counts_by_owner(
        session: Session, model: Any, user_ids: list[str]
    ) -> dict[str, int]:
        if not user_ids:
            return {}
        rows = session.execute(
            select(model.owner_user_id, func.count(model.id))
            .where(model.owner_user_id.in_(user_ids))
            .group_by(model.owner_user_id)
        ).all()
        return {str(owner_id): int(count) for owner_id, count in rows if owner_id is not None}

    def _user_summaries(
        self, session: Session, users: list[User]
    ) -> list[AdminUserSummary]:
        user_ids = [user.id for user in users]
        dataset_counts = self._counts_by_owner(session, Dataset, user_ids)
        voice_counts = self._counts_by_owner(session, VoiceProfile, user_ids)
        job_counts = self._counts_by_owner(session, Job, user_ids)
        return [
            AdminUserSummary(
                id=user.id,
                username=user.username,
                role=user.role,
                status=user.status,
                created_at=user.created_at,
                dataset_count=dataset_counts.get(user.id, 0),
                voice_count=voice_counts.get(user.id, 0),
                job_count=job_counts.get(user.id, 0),
            )
            for user in users
        ]

    @staticmethod
    def _audit_summary(event: AuditEvent) -> AuditEventSummary:
        metadata = LocalAdminService._sanitized_metadata(event.metadata_json)
        return AuditEventSummary(
            id=event.id,
            event_type=event.event_type,
            subject_id=event.subject_id,
            metadata=metadata,
            created_at=event.created_at,
        )

    @staticmethod
    def _sanitized_metadata(raw_json: str) -> dict[str, str]:
        try:
            parsed = json.loads(raw_json)
        except (TypeError, json.JSONDecodeError):
            return {}
        if not isinstance(parsed, dict):
            return {}
        allowed = ("old_status", "new_status", "actor_id")
        return {
            key: str(parsed[key])
            for key in allowed
            if isinstance(parsed.get(key), (str, int))
        }
