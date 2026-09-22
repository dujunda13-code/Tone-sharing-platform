from datetime import datetime, timezone

from fastapi import FastAPI

from backend.app.api.dependencies import get_current_user
from backend.app.schemas.auth import UserRecord


def install_authenticated_user(app: FastAPI, user_id: str = "test-user") -> None:
    app.dependency_overrides[get_current_user] = lambda: UserRecord(
        id=user_id,
        username=user_id,
        role="user",
        status="active",
        created_at=datetime.now(timezone.utc),
    )
