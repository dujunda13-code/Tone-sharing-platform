from __future__ import annotations

from fastapi import HTTPException, Request

from backend.app.schemas.auth import UserRecord
from backend.app.services.auth import AuthService


SESSION_COOKIE_NAME = "timbre_session"


def _auth_service(request: Request) -> AuthService:
    service = getattr(request.app.state, "auth_service", None)
    if service is None:
        service = AuthService()
        request.app.state.auth_service = service
    return service


def get_current_user(request: Request) -> UserRecord:
    raw_token = request.cookies.get(SESSION_COOKIE_NAME)
    if not raw_token:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "AUTH_REQUIRED",
                "message": "请先登录",
                "details": {},
            },
        )
    user = _auth_service(request).resolve_session(raw_token)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "AUTH_REQUIRED",
                "message": "登录已失效，请重新登录",
                "details": {},
            },
        )
    return user


def require_admin(request: Request) -> UserRecord:
    user = get_current_user(request)
    if user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail={
                "code": "FORBIDDEN",
                "message": "需要管理员权限",
                "details": {},
            },
        )
    return user
