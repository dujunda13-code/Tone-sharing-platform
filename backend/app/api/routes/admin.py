from fastapi import APIRouter, Depends, HTTPException, Query, Request

from backend.app.api.dependencies import require_admin
from backend.app.schemas.admin import (
    AdminUserListResponse,
    AdminUserSummary,
    AuditEventListResponse,
    UserStatusUpdate,
)
from backend.app.schemas.auth import UserRecord
from backend.app.services.admin import AdminUserStatusError, LocalAdminService


router = APIRouter(prefix="/api/admin", tags=["admin"])


def _service(request: Request) -> LocalAdminService:
    return request.app.state.admin_service


def _status_error(error: AdminUserStatusError) -> HTTPException:
    status_code = 409 if error.code == "ADMIN_SELF_STATUS_CHANGE_FORBIDDEN" else 404
    return HTTPException(
        status_code=status_code,
        detail={"code": error.code, "message": str(error), "details": {}},
    )


@router.get("/users", response_model=AdminUserListResponse)
def list_users(
    request: Request,
    current_admin: UserRecord = Depends(require_admin),
) -> AdminUserListResponse:
    del current_admin
    return _service(request).list_users()


@router.patch("/users/{user_id}/status", response_model=AdminUserSummary)
def update_user_status(
    user_id: str,
    body: UserStatusUpdate,
    request: Request,
    current_admin: UserRecord = Depends(require_admin),
) -> AdminUserSummary:
    try:
        return _service(request).set_user_status(current_admin.id, user_id, body.status)
    except AdminUserStatusError as exc:
        raise _status_error(exc) from exc


@router.get("/audit-events", response_model=AuditEventListResponse)
def list_audit_events(
    request: Request,
    current_admin: UserRecord = Depends(require_admin),
    limit: int = Query(default=20, ge=1, le=100),
    before_id: int | None = Query(default=None, ge=1),
) -> AuditEventListResponse:
    del current_admin
    return _service(request).list_audit_events(limit, before_id)
