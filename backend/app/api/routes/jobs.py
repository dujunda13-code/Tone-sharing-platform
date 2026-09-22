from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.app.api.dependencies import get_current_user
from backend.app.schemas.auth import UserRecord
from backend.app.schemas.common import JobKind, JobStatus
from backend.app.schemas.workspace import JobListResponse, JobSummary
from backend.app.services.job_queue import JobQueue, JobRecord
from backend.app.services.metrics import EVALUATION_SEED, load_evaluation_texts
from backend.app.services.voice_profiles import VoiceProfileStore
from backend.app.services.workspace import WorkspaceService


class EvaluationQueuedResponse(BaseModel):
    job_id: str
    status: str


class EvaluationRequestError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class EvaluationService:
    def __init__(
        self,
        *,
        profiles: VoiceProfileStore | None = None,
        queue: JobQueue | None = None,
        texts_path: Path | str = Path("config/evaluation_texts.zh-en.json"),
    ) -> None:
        self.profiles = profiles or VoiceProfileStore()
        self.queue = queue or JobQueue()
        self.texts_path = Path(texts_path).resolve()

    def enqueue(self, profile_id: str, owner_user_id: str | None = None) -> EvaluationQueuedResponse:
        try:
            profile = self.profiles.get(profile_id, owner_user_id=owner_user_id)
        except KeyError as exc:
            raise EvaluationRequestError("VOICE_PROFILE_NOT_FOUND", "音色档案不存在") from exc
        if profile.status != "ready" or profile.public_weight_dir is None:
            raise EvaluationRequestError("VOICE_PROFILE_NOT_READY", "音色档案尚不可评测")
        if not Path(profile.public_weight_dir).resolve().is_dir():
            raise EvaluationRequestError("VOICE_PROFILE_WEIGHTS_UNAVAILABLE", "音色档案权重不可用")
        try:
            load_evaluation_texts(self.texts_path)
        except ValueError as exc:
            raise EvaluationRequestError("EVALUATION_CONTRACT_INVALID", str(exc)) from exc
        job = self.queue.enqueue(
            JobKind.EVALUATE,
            {
                "profile_id": profile.id,
                "dataset_id": profile.dataset_id,
                "evaluation_texts_path": "config/evaluation_texts.zh-en.json",
                "seed": EVALUATION_SEED,
            },
            owner_user_id=owner_user_id,
        )
        return EvaluationQueuedResponse(job_id=job.id, status=job.status.value)

    def get(self, job_id: str, owner_user_id: str | None = None) -> JobRecord:
        return self.queue.get(job_id, owner_user_id=owner_user_id)


router = APIRouter(prefix="/api", tags=["jobs"])


def _service(request: Request) -> EvaluationService:
    service = getattr(request.app.state, "evaluation_service", None)
    if service is None:
        service = EvaluationService()
        request.app.state.evaluation_service = service
    return service


def _workspace(request: Request) -> WorkspaceService:
    return request.app.state.workspace_service


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": {},
                "request_id": uuid4().hex,
            }
        },
    )


@router.post(
    "/voices/{profile_id}/evaluate",
    response_model=EvaluationQueuedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def evaluate_voice(
    profile_id: str,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        return _service(request).enqueue(profile_id, owner_user_id=current_user.id)
    except EvaluationRequestError as exc:
        status_code = 404 if exc.code == "VOICE_PROFILE_NOT_FOUND" else 409
        return _error(status_code, exc.code, str(exc))


@router.get("/jobs", response_model=JobListResponse)
def list_jobs(
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
    job_status: JobStatus | None = Query(default=None, alias="status"),
) -> JobListResponse:
    return _workspace(request).list_jobs(current_user.id, status=job_status)


@router.get("/jobs/{job_id}", response_model=JobSummary)
def get_job(
    job_id: str,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        return _workspace(request).get_job_summary(current_user.id, job_id)
    except KeyError:
        return _error(404, "JOB_NOT_FOUND", "任务不存在")
