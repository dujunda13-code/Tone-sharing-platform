from datetime import datetime

from pydantic import BaseModel, Field


class PlazaPublishRequest(BaseModel):
    voice_profile_id: str = Field(min_length=1, max_length=64)
    description: str | None = Field(default=None, max_length=500)


class PlazaCommentCreate(BaseModel):
    content: str = Field(min_length=1)


class PlazaVoiceInfo(BaseModel):
    voice_profile_id: str
    display_name: str
    status: str
    reference_count: int
    reference_emotions: list[str]


class PlazaAuthorInfo(BaseModel):
    user_id: str
    display_name: str
    has_avatar: bool


class PlazaPostSummary(BaseModel):
    id: str
    voice: PlazaVoiceInfo
    author: PlazaAuthorInfo
    description: str | None
    like_count: int
    comment_count: int
    favorite_count: int
    liked_by_me: bool
    favorited_by_me: bool
    created_at: datetime


class PlazaPostListResponse(BaseModel):
    items: list[PlazaPostSummary]
    total: int


class PlazaCommentSummary(BaseModel):
    id: str
    post_id: str
    author: PlazaAuthorInfo
    content: str
    voice_name: str | None = None
    created_at: datetime


class PlazaCommentListResponse(BaseModel):
    items: list[PlazaCommentSummary]
    total: int


class PlazaActionResponse(BaseModel):
    ok: bool


class PlazaDeleteResponse(BaseModel):
    deleted: bool


class PlazaImportRequest(BaseModel):
    authorization_confirmed: bool = False


class PlazaImportResponse(BaseModel):
    voice_profile_id: str
    display_name: str
