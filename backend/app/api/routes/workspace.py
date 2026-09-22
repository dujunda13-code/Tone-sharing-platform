from fastapi import APIRouter, Depends, Request

from backend.app.api.dependencies import get_current_user
from backend.app.schemas.auth import UserRecord
from backend.app.schemas.workspace import DashboardResponse, ReadinessSummary
from backend.app.services.workspace import WorkspaceService


router = APIRouter(prefix="/api", tags=["workspace"])


def _service(request: Request) -> WorkspaceService:
    return request.app.state.workspace_service


@router.get("/dashboard", response_model=DashboardResponse)
def dashboard(
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
) -> DashboardResponse:
    readiness = ReadinessSummary.model_validate(
        request.app.state.health_service.ready().model_dump(mode="json")
    )
    return _service(request).dashboard(current_user.id, readiness)
