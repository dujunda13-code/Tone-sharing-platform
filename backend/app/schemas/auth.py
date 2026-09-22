from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class PasswordRecord:
    password_hash: str
    password_salt: str
    password_params_json: str


@dataclass(frozen=True)
class UserRecord:
    id: str
    username: str
    role: str
    status: str
    created_at: datetime


@dataclass(frozen=True)
class SessionRecord:
    id: str
    user_id: str
    expires_at: datetime
    created_at: datetime
    last_seen_at: datetime


class CredentialsRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=256)


class UserResponse(BaseModel):
    id: str
    username: str
    role: str
    status: str


class AuthStatusResponse(BaseModel):
    authenticated: bool
    user: UserResponse | None = None
