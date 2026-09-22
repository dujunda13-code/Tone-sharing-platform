from pydantic import BaseModel, Field


class PlazaStats(BaseModel):
    published: int
    likes_received: int


class UserProfileResponse(BaseModel):
    user_id: str
    username: str
    display_name: str
    bio: str | None
    has_avatar: bool
    stats: PlazaStats


class UserProfileUpdate(BaseModel):
    display_name: str = Field(min_length=1, max_length=32)
    bio: str | None = Field(default=None, max_length=200)
