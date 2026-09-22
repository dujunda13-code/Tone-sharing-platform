from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

from backend.app.api.dependencies import get_current_user
from backend.app.schemas.auth import UserRecord
from backend.app.schemas.notification import (
    MarkReadRequest,
    MarkReadResponse,
    NotificationListResponse,
    NotificationSummary,
    UnreadCountResponse,
)
from backend.app.services.notifications import NotificationStore

router = APIRouter(prefix="/api/notifications", tags=["notifications"])


def _store(request: Request) -> NotificationStore:
    return request.app.state.notification_store


@router.get("", response_model=NotificationListResponse)
def list_notifications(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: UserRecord = Depends(get_current_user),
):
    views, total = _store(request).list_views(current_user.id, limit=limit, offset=offset)
    return NotificationListResponse(
        items=[
            NotificationSummary(
                id=view.id,
                type=view.type,
                actor_display_name=view.actor_display_name,
                post_id=view.post_id,
                voice_name=view.voice_name,
                comment_excerpt=view.comment_excerpt,
                is_read=view.is_read,
                created_at=view.created_at,
            )
            for view in views
        ],
        total=total,
    )


@router.get("/unread-count", response_model=UnreadCountResponse)
def unread_count(request: Request, current_user: UserRecord = Depends(get_current_user)):
    return UnreadCountResponse(count=_store(request).unread_count(current_user.id))


@router.post("/mark-read", response_model=MarkReadResponse)
def mark_read(
    body: MarkReadRequest,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    updated = _store(request).mark_read(
        current_user.id, notification_ids=body.notification_ids, all=body.all
    )
    return MarkReadResponse(updated=updated)
