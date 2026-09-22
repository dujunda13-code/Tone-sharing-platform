from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import FileResponse, JSONResponse

from backend.app.api.dependencies import get_current_user
from backend.app.schemas.auth import UserRecord
from backend.app.schemas.common import JobKind
from backend.app.schemas.synthesis import (
    SynthesisCreate,
    SynthesisQueuedResponse,
)
from backend.app.schemas.workspace import SynthesisListResponse, SynthesisSummary
from backend.app.services.job_queue import JobQueue, JobRecord
from backend.app.services.plaza import PlazaStore
from backend.app.services.sensitive_filter import SensitiveFilter
from backend.app.services.storage import LocalStorage
from backend.app.services.voice_profiles import VoiceProfileStore
from backend.app.services.workspace import WorkspaceService


class SynthesisRequestError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class SynthesisService:
    """Apply request-time safety gates before synthesis reaches the GPU worker."""

    def __init__(
        self,
        *,
        profiles: VoiceProfileStore | None = None,
        queue: JobQueue | None = None,
        sensitive_filter: SensitiveFilter | None = None,
        storage: LocalStorage | None = None,
        plaza: PlazaStore | None = None,
        cloud_tts_available: bool = False,
    ) -> None:
        self.profiles = profiles or VoiceProfileStore()
        self.queue = queue or JobQueue()
        self.sensitive_filter = sensitive_filter or SensitiveFilter.from_file(
            "config/sensitive_words.zh-en.txt"
        )
        self.storage = storage or LocalStorage("data")
        self.plaza = plaza
        self.cloud_tts_available = cloud_tts_available

    def enqueue(
        self,
        request: SynthesisCreate,
        owner_user_id: str | None = None,
    ) -> SynthesisQueuedResponse:
        if request.consent_confirmed is not True:
            raise SynthesisRequestError("CONSENT_REQUIRED", "需要字面确认合成授权")
        if request.synthesis_mode == "emotion_api" and not self.cloud_tts_available:
            raise SynthesisRequestError(
                "CLOUD_TTS_NOT_CONFIGURED", "情绪合成服务尚未配置，请联系管理员"
            )
        if self.sensitive_filter.check(request.text).blocked:
            raise SynthesisRequestError("SENSITIVE_TEXT_BLOCKED", "文本包含禁止合成的内容")
        try:
            profile = self.profiles.get(request.voice_profile_id, owner_user_id=owner_user_id)
        except KeyError as exc:
            # Published plaza profiles may be synthesized by any signed-in user;
            # authorization is re-confirmed per request via consent_confirmed.
            if self.plaza is None or not self.plaza.is_published_ready(request.voice_profile_id):
                raise SynthesisRequestError("VOICE_PROFILE_NOT_FOUND", "音色档案不存在") from exc
            profile = self.profiles.get(request.voice_profile_id)
        if profile.status != "ready":
            raise SynthesisRequestError("VOICE_PROFILE_NOT_READY", "音色档案尚不可用于合成")
        if profile.mode != "zero_shot" and profile.public_weight_dir is None:
            # Only zero-shot profiles synthesize without personal weights.
            raise SynthesisRequestError("VOICE_PROFILE_NOT_READY", "音色档案尚不可用于合成")
        job = self.queue.enqueue(
            JobKind.SYNTHESIZE,
            request.model_dump(mode="json"),
            owner_user_id=owner_user_id,
        )
        return SynthesisQueuedResponse(job_id=job.id, status="queued")

    def get(self, job_id: str, owner_user_id: str | None = None) -> JobRecord:
        job = self.queue.get(job_id, owner_user_id=owner_user_id)
        if job.kind is not JobKind.SYNTHESIZE:
            raise KeyError(job_id)
        return job

    def output_path(self, job_id: str, owner_user_id: str | None = None):
        job = self.get(job_id, owner_user_id=owner_user_id)
        if job.result is None or not job.result.get("public_audio_path"):
            raise FileNotFoundError(job_id)
        output_path = self.storage.resolve("outputs", str(job.result["public_audio_path"]))
        if not output_path.is_file():
            raise FileNotFoundError(output_path)
        return output_path


router = APIRouter(prefix="/api", tags=["synthesis"])


def _service(request: Request) -> SynthesisService:
    service = getattr(request.app.state, "synthesis_service", None)
    if service is None:
        service = SynthesisService()
        request.app.state.synthesis_service = service
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


@router.post("/syntheses", response_model=SynthesisQueuedResponse, status_code=status.HTTP_202_ACCEPTED)
def create_synthesis(
    body: SynthesisCreate,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        return _service(request).enqueue(body, owner_user_id=current_user.id)
    except SynthesisRequestError as exc:
        status_code = 422 if exc.code in {"CONSENT_REQUIRED", "SENSITIVE_TEXT_BLOCKED"} else 409
        if exc.code == "VOICE_PROFILE_NOT_FOUND":
            status_code = 404
        if exc.code == "CLOUD_TTS_NOT_CONFIGURED":
            status_code = 503
        return _error(status_code, exc.code, str(exc))


@router.get("/syntheses", response_model=SynthesisListResponse)
def list_syntheses(
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
) -> SynthesisListResponse:
    return _workspace(request).list_syntheses(current_user.id)


@router.get("/syntheses/{job_id}", response_model=SynthesisSummary)
def get_synthesis(
    job_id: str,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        return _workspace(request).get_synthesis_summary(current_user.id, job_id)
    except KeyError:
        return _error(404, "SYNTHESIS_NOT_FOUND", "合成任务不存在")


@router.get("/syntheses/{job_id}/audio")
def get_synthesis_audio(
    job_id: str,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        return FileResponse(
            _service(request).output_path(job_id, owner_user_id=current_user.id),
            media_type="audio/wav",
            filename=f"{job_id}.wav",
        )
    except KeyError:
        return _error(404, "SYNTHESIS_NOT_FOUND", "合成任务不存在")
    except FileNotFoundError:
        return _error(404, "SYNTHESIS_AUDIO_NOT_READY", "已验证音频尚不可用")
