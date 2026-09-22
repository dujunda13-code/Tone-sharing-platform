from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from backend.app.api.dependencies import get_current_user
from backend.app.schemas.auth import UserRecord
from backend.app.schemas.user_profile import PlazaStats, UserProfileResponse, UserProfileUpdate
from backend.app.services.plaza import PlazaStore
from backend.app.services.user_profiles import AvatarRejected, UserProfileStore

router = APIRouter(prefix="/api", tags=["users"])

_AVATAR_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".webp": "image/webp",
}


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "details": {}, "request_id": uuid4().hex}},
    )


def _profiles(request: Request) -> UserProfileStore:
    return request.app.state.user_profile_store


def _plaza(request: Request) -> PlazaStore:
    return request.app.state.plaza_service


def _response(record, current_user: UserRecord, plaza: PlazaStore) -> UserProfileResponse:
    published, likes_received = plaza.count_stats(current_user.id)
    return UserProfileResponse(
        user_id=record.user_id,
        username=current_user.username,
        display_name=record.display_name,
        bio=record.bio,
        has_avatar=record.has_avatar,
        stats=PlazaStats(published=published, likes_received=likes_received),
    )


@router.get("/users/me/profile", response_model=UserProfileResponse)
def get_my_profile(request: Request, current_user: UserRecord = Depends(get_current_user)):
    record = _profiles(request).get(current_user.id, fallback_username=current_user.username)
    return _response(record, current_user, _plaza(request))


@router.patch("/users/me/profile", response_model=UserProfileResponse)
def update_my_profile(
    body: UserProfileUpdate,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        record = _profiles(request).update(
            current_user.id, display_name=body.display_name, bio=body.bio
        )
    except ValueError as exc:
        return _error(400, "PROFILE_UPDATE_INVALID", str(exc))
    return _response(record, current_user, _plaza(request))


@router.post("/users/me/avatar", response_model=UserProfileResponse)
async def upload_my_avatar(
    file: UploadFile,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    content = await file.read()
    try:
        record = _profiles(request).set_avatar(
            current_user.id, content, fallback_username=current_user.username
        )
    except AvatarRejected as exc:
        return _error(400, exc.code, str(exc))
    return _response(record, current_user, _plaza(request))


@router.get("/avatars/{user_id}")
def get_avatar(
    user_id: str,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    path = _profiles(request).avatar_file(user_id)
    if path is None:
        return _error(404, "AVATAR_NOT_FOUND", "该用户尚未上传头像")
    media_type = _AVATAR_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media_type)
