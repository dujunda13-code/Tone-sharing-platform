from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import os
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.api.routes import admin, auth, datasets, health, jobs, safety, synthesis, voices, workspace
from backend.app.api.routes import notifications as notifications_routes
from backend.app.api.routes import plaza as plaza_routes
from backend.app.api.routes import users as users_routes
from backend.app.api.routes.datasets import DatasetService
from backend.app.api.routes.health import HealthDependencies, HealthService
from backend.app.api.routes.jobs import EvaluationService
from backend.app.api.routes.safety import SafetyService
from backend.app.api.routes.synthesis import SynthesisService
from backend.app.api.routes.voices import EmotionOverrideStore, VoiceService
from backend.app.core.config import AppSettings, get_settings
from backend.app.db.session import create_database_engine, init_db, migrate_voice_base_id
from backend.app.services.job_queue import JobQueue
from backend.app.services.auth import AuthService
from backend.app.services.admin import LocalAdminService
from backend.app.services.emotion import EmotionAnalyzer
from backend.app.services.notifications import NotificationStore
from backend.app.services.plaza import PlazaStore
from backend.app.services.sensitive_filter import SensitiveFilter
from backend.app.services.storage import LocalStorage
from backend.app.services.user_profiles import UserProfileStore
from backend.app.services.voice_packages import VoicePackageService
from backend.app.services.voice_profiles import VoiceProfileStore
from backend.app.services.workspace import WorkspaceService


LEGACY_V2PRO_BASE_ID = "gpt-sovits-v2pro-official"
V2PROPLUS_BASE_ID = "gpt-sovits-v2proplus-official"


@dataclass(frozen=True)
class ErrorPayload:
    code: str
    message: str
    details: Any


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", uuid4().hex)


def _error_response(request: Request, status_code: int, payload: ErrorPayload) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": payload.code,
                "message": payload.message,
                "details": payload.details if payload.details is not None else {},
                "request_id": _request_id(request),
            }
        },
    )


def _install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return _error_response(
            request,
            422,
            ErrorPayload(
                code="REQUEST_VALIDATION_ERROR",
                message="请求参数校验失败",
                details=jsonable_encoder(exc.errors()),
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        detail = exc.detail
        if isinstance(detail, dict):
            payload = ErrorPayload(
                code=str(detail.get("code", "HTTP_ERROR")),
                message=str(detail.get("message", "请求失败")),
                details=detail.get("details", {}),
            )
        else:
            payload = ErrorPayload(code="HTTP_ERROR", message=str(detail), details={})
        return _error_response(request, exc.status_code, payload)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        del exc
        return _error_response(
            request,
            500,
            ErrorPayload(
                code="INTERNAL_SERVER_ERROR",
                message="本地服务发生未处理错误",
                details={},
            ),
        )


def _database_url(storage: LocalStorage) -> str:
    database_path = (storage.root / "app.db").resolve()
    return f"sqlite+pysqlite:///{database_path.as_posix()}"


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Create routes, exception handlers, request IDs, DB lifecycle and recovery."""
    app_settings = settings or get_settings()
    storage = LocalStorage(app_settings.storage.root)
    engine = create_database_engine(_database_url(storage))
    init_db(engine)
    if app_settings.models.active_voice_base == V2PROPLUS_BASE_ID:
        migrate_voice_base_id(engine, LEGACY_V2PRO_BASE_ID, V2PROPLUS_BASE_ID)
    queue = JobQueue(engine=engine)
    profiles = VoiceProfileStore(engine=engine)
    notification_store = NotificationStore(engine=engine)
    plaza_service = PlazaStore(
        engine=engine,
        sensitive_filter=SensitiveFilter.from_file("config/sensitive_words.zh-en.txt"),
        notifications=notification_store,
    )
    voice_package_service = VoicePackageService(engine=engine, storage=storage, profiles=profiles)
    user_profile_store = UserProfileStore(engine=engine, storage=storage)
    emotion_analyzer = EmotionAnalyzer()

    def emotion2vec_readiness() -> tuple[bool, str]:
        available = emotion_analyzer.available()
        return (
            available,
            "local Emotion2Vec preprocessing model is available"
            if available
            else "local Emotion2Vec preprocessing model is unavailable",
        )

    dataset_service = DatasetService(
        storage=storage,
        engine=engine,
        emotion_analyzer=emotion_analyzer,
        snr_warning_db=app_settings.audio.snr_warning_db,
    )
    synthesis_service = SynthesisService(
        profiles=profiles,
        queue=queue,
        storage=storage,
        plaza=plaza_service,
        cloud_tts_available=bool(os.environ.get("COMPSHARE_API_KEY", "").strip()),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db(app.state.engine)
        app.state.storage.root.mkdir(parents=True, exist_ok=True)
        app.state.recovered_jobs = app.state.queue.recover_interrupted()
        yield

    app = FastAPI(
        title="本地音色共享平台 API",
        version="1.0.0-local",
        lifespan=lifespan,
    )
    app.state.settings = app_settings
    app.state.auth_service = AuthService(engine=engine)
    app.state.admin_service = LocalAdminService(engine=engine)
    app.state.engine = engine
    app.state.storage = storage
    app.state.queue = queue
    app.state.profiles = profiles
    app.state.dataset_service = dataset_service
    app.state.emotion_analyzer = emotion_analyzer
    app.state.voice_service = VoiceService(
        profiles=profiles, datasets=dataset_service, settings=app_settings
    )
    app.state.synthesis_service = synthesis_service
    app.state.safety_service = SafetyService(syntheses=synthesis_service)
    app.state.evaluation_service = EvaluationService(profiles=profiles, queue=queue)
    app.state.emotion_override_store = EmotionOverrideStore()
    app.state.health_service = HealthService(
        engine=engine,
        storage=storage,
        settings=app_settings,
        dependencies=HealthDependencies(emotion2vec=emotion2vec_readiness),
    )
    app.state.workspace_service = WorkspaceService(
        engine=engine,
        storage=storage,
        configured_base_model_ids=frozenset(app_settings.models.voice_bases),
        snr_warning_db=app_settings.audio.snr_warning_db,
    )
    app.state.notification_store = notification_store
    app.state.plaza_service = plaza_service
    app.state.voice_package_service = voice_package_service
    app.state.user_profile_store = user_profile_store

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid4().hex
        request.state.request_id = request_id[:128]
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )
    _install_exception_handlers(app)
    app.include_router(auth.router)
    app.include_router(admin.router)
    app.include_router(health.router)
    app.include_router(workspace.router)
    app.include_router(datasets.router)
    app.include_router(voices.router)
    app.include_router(synthesis.router)
    app.include_router(safety.router)
    app.include_router(jobs.router)
    app.include_router(users_routes.router)
    app.include_router(notifications_routes.router)
    app.include_router(plaza_routes.router)
    return app


app = create_app()
