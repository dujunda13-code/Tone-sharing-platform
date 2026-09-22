from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class UserStatusUpdate(BaseModel):
    status: Literal["active", "disabled"]


class AdminUserSummary(BaseModel):
    id: str
    username: str
    role: Literal["user", "admin"]
    status: Literal["active", "disabled"]
    created_at: datetime
    dataset_count: int
    voice_count: int
    job_count: int


class AuditEventSummary(BaseModel):
    id: int
    event_type: str
    subject_id: str | None
    metadata: dict[str, str]
    created_at: datetime


class AdminUserListResponse(BaseModel):
    items: list[AdminUserSummary]


class AuditEventListResponse(BaseModel):
    items: list[AuditEventSummary]
    next_before_id: int | None
