from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from backend.app.api.dependencies import get_current_user
from backend.app.schemas.auth import UserRecord
from backend.app.schemas.plaza import (
    PlazaActionResponse,
    PlazaCommentCreate,
    PlazaCommentListResponse,
    PlazaCommentSummary,
    PlazaDeleteResponse,
    PlazaImportRequest,
    PlazaImportResponse,
    PlazaPostListResponse,
    PlazaPostSummary,
    PlazaPublishRequest,
)
from backend.app.services.plaza import PlazaError, PlazaStore
from backend.app.services.voice_packages import VoicePackageService

router = APIRouter(prefix="/api/plaza", tags=["plaza"])

_STATUS_BY_CODE = {
    "PLAZA_POST_ALREADY_PUBLISHED": 409,
    "PLAZA_ALREADY_LIKED": 409,
    "PLAZA_ALREADY_FAVORITED": 409,
    "PLAZA_NOT_POST_AUTHOR": 403,
    "PLAZA_VOICE_PROFILE_NOT_READY": 400,
    "PLAZA_DESCRIPTION_TOO_LONG": 400,
    "PLAZA_COMMENT_INVALID": 400,
    "SENSITIVE_COMMENT_BLOCKED": 422,
    "AUTHORIZATION_REQUIRED": 400,
    "PLAZA_REFERENCE_AUDIO_MISSING": 409,
    "REFERENCE_COPY_INTEGRITY_FAILED": 500,
}


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "details": {}, "request_id": uuid4().hex}},
    )


def _plaza_error(exc: PlazaError) -> JSONResponse:
    return _error(_STATUS_BY_CODE.get(exc.code, 400), exc.code, str(exc))


def _service(request: Request) -> PlazaStore:
    return request.app.state.plaza_service


def _summary(view) -> PlazaPostSummary:
    return PlazaPostSummary(
        id=view.post.id,
        voice={
            "voice_profile_id": view.post.voice_profile_id,
            "display_name": view.voice_name,
            "status": view.voice_status,
            "reference_count": view.reference_count,
            "reference_emotions": list(view.reference_emotions),
        },
        author={
            "user_id": view.post.author_user_id,
            "display_name": view.author_display_name,
            "has_avatar": view.author_has_avatar,
        },
        description=view.post.description,
        like_count=view.like_count,
        comment_count=view.comment_count,
        favorite_count=view.favorite_count,
        liked_by_me=view.liked_by_viewer,
        favorited_by_me=view.favorited_by_viewer,
        created_at=view.post.created_at,
    )


def _comment_summary(item) -> PlazaCommentSummary:
    return PlazaCommentSummary(
        id=item.id,
        post_id=item.post_id,
        author={
            "user_id": item.author_user_id,
            "display_name": item.author_display_name,
            "has_avatar": item.author_has_avatar,
        },
        content=item.content,
        voice_name=item.voice_name,
        created_at=item.created_at,
    )


@router.get("/posts", response_model=PlazaPostListResponse)
def list_posts(
    request: Request,
    mine: bool = Query(default=False),
    favorited: bool = Query(default=False),
    q: str | None = Query(default=None, max_length=100),
    sort: str = Query(default="newest", pattern="^(newest|likes)$"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: UserRecord = Depends(get_current_user),
):
    views, total = _service(request).list_posts(
        viewer_user_id=current_user.id,
        mine=mine,
        favorited=favorited,
        q=q,
        sort=sort,
        limit=limit,
        offset=offset,
    )
    return PlazaPostListResponse(items=[_summary(v) for v in views], total=total)


@router.get("/comments", response_model=PlazaCommentListResponse)
def my_comments(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: UserRecord = Depends(get_current_user),
):
    items, total = _service(request).my_comments(current_user.id, limit=limit, offset=offset)
    return PlazaCommentListResponse(items=[_comment_summary(i) for i in items], total=total)


@router.post("/posts", response_model=PlazaPostSummary, status_code=201)
def publish_post(
    body: PlazaPublishRequest,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    service = _service(request)
    try:
        post = service.publish(
            current_user.id,
            voice_profile_id=body.voice_profile_id,
            description=body.description,
        )
    except KeyError:
        return _error(404, "PLAZA_VOICE_PROFILE_NOT_FOUND", "音色档案不存在")
    except PlazaError as exc:
        return _plaza_error(exc)
    return _summary(service.get_post_view(post.id, viewer_user_id=current_user.id))


@router.get("/posts/{post_id}", response_model=PlazaPostSummary)
def get_post(
    post_id: str,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        view = _service(request).get_post_view(post_id, viewer_user_id=current_user.id)
    except KeyError:
        return _error(404, "PLAZA_POST_NOT_FOUND", "帖子不存在")
    return _summary(view)


@router.delete("/posts/{post_id}", response_model=PlazaDeleteResponse)
def delete_post(
    post_id: str,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        _service(request).delete_post(post_id, current_user.id)
    except KeyError:
        return _error(404, "PLAZA_POST_NOT_FOUND", "帖子不存在")
    except PlazaError as exc:
        return _plaza_error(exc)
    return PlazaDeleteResponse(deleted=True)


@router.get("/posts/{post_id}/comments", response_model=PlazaCommentListResponse)
def list_comments(
    post_id: str,
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        items, total = _service(request).list_comments(post_id, limit=limit, offset=offset)
    except KeyError:
        return _error(404, "PLAZA_POST_NOT_FOUND", "帖子不存在")
    return PlazaCommentListResponse(items=[_comment_summary(i) for i in items], total=total)


@router.post("/posts/{post_id}/comments", response_model=PlazaCommentSummary, status_code=201)
def add_comment(
    post_id: str,
    body: PlazaCommentCreate,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        item = _service(request).add_comment(post_id, current_user.id, body.content)
    except KeyError:
        return _error(404, "PLAZA_POST_NOT_FOUND", "帖子不存在")
    except PlazaError as exc:
        return _plaza_error(exc)
    return _comment_summary(item)


@router.post("/posts/{post_id}/likes", response_model=PlazaActionResponse)
def like_post(
    post_id: str,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        _service(request).like(post_id, current_user.id)
    except KeyError:
        return _error(404, "PLAZA_POST_NOT_FOUND", "帖子不存在")
    except PlazaError as exc:
        return _plaza_error(exc)
    return PlazaActionResponse(ok=True)


@router.delete("/posts/{post_id}/likes", response_model=PlazaActionResponse)
def unlike_post(
    post_id: str,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        _service(request).unlike(post_id, current_user.id)
    except KeyError:
        return _error(404, "PLAZA_POST_NOT_FOUND", "帖子不存在")
    return PlazaActionResponse(ok=True)


@router.post("/posts/{post_id}/favorites", response_model=PlazaActionResponse)
def favorite_post(
    post_id: str,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        _service(request).favorite(post_id, current_user.id)
    except KeyError:
        return _error(404, "PLAZA_POST_NOT_FOUND", "帖子不存在")
    except PlazaError as exc:
        return _plaza_error(exc)
    return PlazaActionResponse(ok=True)


@router.delete("/posts/{post_id}/favorites", response_model=PlazaActionResponse)
def unfavorite_post(
    post_id: str,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        _service(request).unfavorite(post_id, current_user.id)
    except KeyError:
        return _error(404, "PLAZA_POST_NOT_FOUND", "帖子不存在")
    return PlazaActionResponse(ok=True)


def _packages(request: Request) -> VoicePackageService:
    return request.app.state.voice_package_service


@router.get("/posts/{post_id}/download")
def download_package(
    post_id: str,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        package = _packages(request).build_package(post_id)
    except KeyError:
        return _error(404, "PLAZA_POST_NOT_FOUND", "帖子不存在")
    except PlazaError as exc:
        return _plaza_error(exc)
    return Response(
        content=package,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="voice-package-{post_id[:8]}.zip"'
        },
    )


@router.get("/posts/{post_id}/download/reference")
def download_reference(
    post_id: str,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        path = _packages(request).reference_wav_path(post_id)
    except KeyError:
        return _error(404, "PLAZA_POST_NOT_FOUND", "帖子不存在")
    except PlazaError as exc:
        return _plaza_error(exc)
    return FileResponse(
        path,
        media_type="audio/wav",
        filename=f"reference-{post_id[:8]}.wav",
    )


@router.get("/posts/{post_id}/preview")
def preview_reference(
    post_id: str,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        path = _packages(request).reference_wav_path(post_id)
    except KeyError:
        return _error(404, "PLAZA_POST_NOT_FOUND", "帖子不存在")
    except PlazaError as exc:
        return _plaza_error(exc)
    return FileResponse(path, media_type="audio/wav")


@router.post("/posts/{post_id}/import", response_model=PlazaImportResponse, status_code=201)
def import_post(
    post_id: str,
    body: PlazaImportRequest,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        forked = _packages(request).import_fork(
            post_id,
            recipient_user_id=current_user.id,
            authorization_confirmed=body.authorization_confirmed,
            fallback_username=current_user.username,
        )
    except KeyError:
        return _error(404, "PLAZA_POST_NOT_FOUND", "帖子不存在")
    except PlazaError as exc:
        return _plaza_error(exc)
    return PlazaImportResponse(voice_profile_id=forked.id, display_name=forked.display_name or "")
