from datetime import datetime

from pydantic import BaseModel


class NotificationSummary(BaseModel):
    id: str
    type: str
    actor_display_name: str
    post_id: str
    voice_name: str
    comment_excerpt: str | None
    is_read: bool
    created_at: datetime


class NotificationListResponse(BaseModel):
    items: list[NotificationSummary]
    total: int


class UnreadCountResponse(BaseModel):
    count: int


class MarkReadRequest(BaseModel):
    notification_ids: list[str] | None = None
    all: bool = False


class MarkReadResponse(BaseModel):
    updated: int
