from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, status

from backend.app.api.dependencies import SESSION_COOKIE_NAME, get_current_user
from backend.app.schemas.auth import CredentialsRequest, UserResponse
from backend.app.services.auth import (
    AdminAlreadyInitialized,
    AuthService,
    InvalidCredentials,
    InvalidUsername,
    UsernameTaken,
    WeakPassword,
)


router = APIRouter(prefix="/api", tags=["auth"])


def _service(request: Request) -> AuthService:
    service = getattr(request.app.state, "auth_service", None)
    if service is None:
        service = AuthService()
        request.app.state.auth_service = service
    return service


def _user_response(user) -> UserResponse:
    return UserResponse(id=user.id, username=user.username, role=user.role, status=user.status)


def _auth_error(code: str, message: str, status_code: int) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message, "details": {}},
    )


@router.post("/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(body: CredentialsRequest, request: Request):
    try:
        return _user_response(_service(request).register(body.username, body.password))
    except UsernameTaken as exc:
        raise _auth_error("USERNAME_TAKEN", "用户名已存在", 409) from exc
    except (InvalidUsername, WeakPassword) as exc:
        raise _auth_error("INVALID_REGISTRATION", str(exc), 422) from exc


@router.post("/auth/login", response_model=UserResponse)
def login(body: CredentialsRequest, request: Request, response: Response):
    try:
        service = _service(request)
        user = service.authenticate(body.username, body.password)
        raw_token, _ = service.create_session(user.id)
    except (InvalidCredentials, InvalidUsername) as exc:
        raise _auth_error("INVALID_CREDENTIALS", "用户名或密码错误", 401) from exc
    response.set_cookie(
        SESSION_COOKIE_NAME,
        raw_token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=int(service.session_ttl.total_seconds()),
        path="/",
    )
    return _user_response(user)


@router.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: Request, response: Response):
    raw_token = request.cookies.get(SESSION_COOKIE_NAME)
    if raw_token:
        _service(request).revoke_session(raw_token)
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")


@router.get("/auth/me", response_model=UserResponse)
def me(request: Request):
    return _user_response(get_current_user(request))


@router.post("/admin/bootstrap", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def bootstrap_admin(body: CredentialsRequest, request: Request):
    try:
        return _user_response(_service(request).bootstrap_admin(body.username, body.password))
    except AdminAlreadyInitialized as exc:
        raise _auth_error("ADMIN_ALREADY_INITIALIZED", "管理员已经初始化", 409) from exc
    except UsernameTaken as exc:
        raise _auth_error("USERNAME_TAKEN", "用户名已存在", 409) from exc
    except (InvalidUsername, WeakPassword) as exc:
        raise _auth_error("INVALID_ADMIN_BOOTSTRAP", str(exc), 422) from exc
