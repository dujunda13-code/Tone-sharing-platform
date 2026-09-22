from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse

from backend.app.api.dependencies import get_current_user
from backend.app.api.routes.datasets import DatasetRecord, DatasetService
from backend.app.schemas.auth import UserRecord
from backend.app.schemas.common import EmotionLabel
from backend.app.schemas.workspace import VoiceListResponse
from backend.app.schemas.voice import (
    VoiceProfileCreate,
    VoiceProfileResponse,
    VoiceReferenceCreate,
    VoiceReferenceItemResponse,
    VoiceTrainRequest,
)
from backend.app.core.config import AppSettings
from backend.app.services.audio_validation import snr_warning_codes
from backend.app.services.voice_profiles import (
    VoiceProfileRecord,
    VoiceProfileStore,
    public_voice_name,
)
from backend.app.services.workspace import WorkspaceService


router = APIRouter(prefix="/api", tags=["voices"])


class VoiceDatasetStateError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class VoiceService:
    """Create ready zero-shot voice profiles from reviewed local references."""

    def __init__(
        self,
        *,
        profiles: VoiceProfileStore | None = None,
        datasets: DatasetService | None = None,
        settings: AppSettings,
    ) -> None:
        self.profiles = profiles or VoiceProfileStore()
        self.datasets = datasets
        self.settings = settings

    def create(self, request: VoiceProfileCreate, owner_user_id: str | None = None) -> VoiceProfileRecord:
        dataset = self._reference_ready_dataset(request.dataset_id, owner_user_id)
        base_model_id = self._active_base_model_id()
        (
            asset_id,
            segment_id,
            prompt_text,
            prompt_language,
            emotion_label,
            emotion_confidence,
        ) = self._reference_payload(dataset.id, owner_user_id)
        return self.profiles.create_zero_shot(
            dataset_id=dataset.id,
            owner_user_id=owner_user_id or "",
            reference_asset_id=asset_id,
            reference_segment_id=segment_id,
            prompt_text=prompt_text,
            prompt_language=prompt_language,
            emotion_label=emotion_label,
            emotion_confidence=emotion_confidence,
            base_model_id=base_model_id,
            display_name=request.display_name,
            reference_name=request.reference_name,
        )

    def add_reference(
        self,
        profile_id: str,
        request: VoiceReferenceCreate,
        owner_user_id: str | None = None,
    ) -> VoiceProfileRecord:
        profile = self.profiles.get(profile_id, owner_user_id=owner_user_id)
        dataset = self._reference_ready_dataset(request.dataset_id, owner_user_id)
        (
            asset_id,
            segment_id,
            prompt_text,
            prompt_language,
            emotion_label,
            emotion_confidence,
        ) = self._reference_payload(dataset.id, owner_user_id)
        self.profiles.add_reference(
            profile.id,
            owner_user_id or "",
            asset_id=asset_id,
            segment_id=segment_id,
            prompt_text=prompt_text,
            prompt_language=prompt_language,
            emotion_label=emotion_label,
            emotion_confidence=emotion_confidence,
            reference_name=request.reference_name,
        )
        return self.profiles.get(profile.id, owner_user_id=owner_user_id)

    def get(self, profile_id: str, owner_user_id: str | None = None) -> VoiceProfileRecord:
        return self.profiles.get(profile_id, owner_user_id=owner_user_id)

    def reference_emotions(self, profile_id: str, owner_user_id: str | None) -> list[EmotionLabel]:
        return [
            EmotionLabel(reference.emotion_label)
            for reference in self.profiles.references(profile_id, owner_user_id or "")
        ]

    def reference_quality(
        self,
        profile_id: str,
        owner_user_id: str,
    ) -> tuple[float | None, tuple[str, ...]]:
        profile = self.profiles.get(profile_id, owner_user_id=owner_user_id)
        if not profile.dataset_id or not profile.reference_segment_id:
            raise KeyError(profile_id)
        if self.datasets is None:
            raise KeyError(profile_id)
        segment = self.datasets.reference_segment(
            profile.dataset_id,
            profile.reference_segment_id,
            owner_user_id,
        )
        return segment.snr_db, snr_warning_codes(
            segment.snr_db,
            self.settings.audio.snr_warning_db,
        )

    def references(
        self, profile_id: str, owner_user_id: str
    ) -> list[VoiceReferenceItemResponse]:
        raw_refs = self.profiles.references(profile_id, owner_user_id)
        results = []
        for ref in raw_refs:
            snr_db = None
            if self.datasets is not None:
                profile = self.profiles.get(profile_id, owner_user_id)
                try:
                    seg = self.datasets.reference_segment(
                        profile.dataset_id, ref.segment_id, owner_user_id
                    )
                    snr_db = seg.snr_db
                except Exception:
                    pass
            results.append(
                VoiceReferenceItemResponse(
                    id=ref.id,
                    asset_id=ref.asset_id,
                    segment_id=ref.segment_id,
                    prompt_text=ref.prompt_text,
                    prompt_language=ref.prompt_language,
                    emotion_label=ref.emotion_label,
                    emotion_confidence=ref.emotion_confidence,
                    is_primary=ref.is_primary,
                    reference_name=ref.reference_name,
                    snr_db=snr_db,
                    created_at=ref.created_at.isoformat(),
                )
            )
        return results

    def delete_reference(
        self, profile_id: str, reference_id: str, owner_user_id: str
    ) -> None:
        self.profiles.delete_reference(profile_id, reference_id, owner_user_id)

    def _active_base_model_id(self) -> str:
        base_model_id = self.settings.models.active_voice_base
        if base_model_id not in self.settings.models.voice_bases:
            raise VoiceDatasetStateError(
                "BASE_MODEL_UNAVAILABLE",
                "基础模型缺失、哈希不符或版本未注册",
            )
        return base_model_id

    def _reference_ready_dataset(
        self, dataset_id: str, owner_user_id: str | None
    ) -> DatasetRecord:
        if self.datasets is None:
            raise VoiceDatasetStateError(
                "DATASET_SERVICE_UNAVAILABLE", "本地数据集服务不可用"
            )
        try:
            dataset = self.datasets.get_for_owner(dataset_id, owner_user_id)
        except KeyError as exc:
            raise VoiceDatasetStateError("DATASET_NOT_FOUND", "数据集不存在") from exc
        if not dataset.authorization_confirmed:
            raise VoiceDatasetStateError(
                "DATASET_AUTHORIZATION_REQUIRED", "需要确认拥有参考音频授权"
            )
        if dataset.status != "ready_for_profile":
            raise VoiceDatasetStateError(
                "REFERENCE_REVIEW_REQUIRED",
                "参考音频尚未完成质检、转写与人工确认",
            )
        # Re-derive the gate from persisted segments so a stale browser
        # state cannot bypass the review loop.
        summary = self.datasets.segment_summary(dataset.id, owner_user_id)
        if (
            not summary["ready_for_profile"]
            or summary["reviewed_segments"] == 0
            or summary["reviewed_effective_seconds"] <= 0
        ):
            raise VoiceDatasetStateError(
                "REFERENCE_REVIEW_REQUIRED",
                "参考音频尚未完成质检、转写与人工确认",
            )
        if (
            dataset.effective_seconds is None
            or not 3.0 <= dataset.effective_seconds <= 10.0
        ):
            raise VoiceDatasetStateError(
                "REFERENCE_DURATION_OUT_OF_RANGE",
                "参考音频有效人声时长需在 3–10 秒内",
            )
        return dataset

    def _reference_payload(
        self, dataset_id: str, owner_user_id: str | None
    ) -> tuple[str, str, str, str, str, float | None]:
        try:
            segment = self.datasets.primary_reference_segment(dataset_id, owner_user_id)
        except KeyError as exc:
            raise VoiceDatasetStateError(
                "REFERENCE_REVIEW_REQUIRED", "参考音频尚未完成人工确认"
            ) from exc
        try:
            asset = self.datasets.primary_asset(dataset_id, owner_user_id)
        except KeyError as exc:
            raise VoiceDatasetStateError(
                "REFERENCE_AUDIO_UNAVAILABLE", "参考音频资产缺失"
            ) from exc
        prompt_text = (segment.manual_transcript or segment.auto_transcript or "").strip()
        if not prompt_text:
            raise VoiceDatasetStateError(
                "REFERENCE_REVIEW_REQUIRED", "确认后的参考文本不能为空"
            )
        if segment.language not in {"zh", "en"}:
            raise VoiceDatasetStateError(
                "REFERENCE_REVIEW_REQUIRED", "参考语言必须是中文或英文"
            )
        emotion_label = segment.manual_emotion_label or segment.auto_emotion_label
        if emotion_label is None:
            raise VoiceDatasetStateError(
                "REFERENCE_REVIEW_REQUIRED", "参考情绪尚未确认"
            )
        try:
            EmotionLabel(emotion_label)
        except ValueError as exc:
            raise VoiceDatasetStateError(
                "REFERENCE_REVIEW_REQUIRED", "参考情绪标签无效"
            ) from exc
        return (
            asset.id,
            segment.segment_id,
            prompt_text,
            segment.language,
            emotion_label,
            segment.auto_emotion_confidence,
        )


class EmotionOverrideRequest(BaseModel):
    label: EmotionLabel
    reason: str = Field(min_length=1, max_length=500)


class EmotionOverrideStore:
    def __init__(self) -> None:
        self._segments: dict[tuple[str, str], dict[str, Any]] = {}

    def register_segment(
        self,
        dataset_id: str,
        segment_id: str,
        *,
        auto_label: str,
        auto_confidence: float,
        owner_user_id: str | None = None,
    ) -> None:
        self._segments[(dataset_id, segment_id)] = {
            "owner_user_id": owner_user_id,
            "auto_label": EmotionLabel(auto_label).value,
            "auto_confidence": float(auto_confidence),
            "effective_label": EmotionLabel(auto_label).value,
            "label_source": "auto",
            "reason": None,
            "updated_at": None,
        }

    def override(
        self,
        dataset_id: str,
        segment_id: str,
        request: EmotionOverrideRequest,
        *,
        owner_user_id: str | None = None,
    ) -> dict[str, Any]:
        segment = self._segments.get((dataset_id, segment_id))
        if segment is None or (
            owner_user_id is not None and segment.get("owner_user_id") != owner_user_id
        ):
            raise KeyError((dataset_id, segment_id))
        segment.update(
            effective_label=request.label.value,
            label_source="manual",
            reason=request.reason,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        return dict(segment)


def _store(request: Request) -> EmotionOverrideStore:
    store = getattr(request.app.state, "emotion_override_store", None)
    if store is None:
        store = EmotionOverrideStore()
        request.app.state.emotion_override_store = store
    return store


def _voice_service(request: Request) -> VoiceService:
    service = getattr(request.app.state, "voice_service", None)
    if service is None:
        service = VoiceService()
        request.app.state.voice_service = service
    return service


def _workspace(request: Request) -> WorkspaceService:
    return request.app.state.workspace_service


def _voice_error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def _profile_response(
    service: VoiceService, record: VoiceProfileRecord, owner_user_id: str
) -> VoiceProfileResponse:
    reference_snr_db, quality_warning_codes = service.reference_quality(
        record.id, owner_user_id
    )
    return VoiceProfileResponse(
        id=record.id,
        dataset_id=record.dataset_id,
        status=record.status,
        mode="zero_shot",
        base_model_id=record.base_model_id or "",
        display_name=public_voice_name(record.id, record.display_name),
        reference_emotions=service.reference_emotions(record.id, owner_user_id),
        reference_snr_db=reference_snr_db,
        quality_warning_codes=list(quality_warning_codes),
        references=service.references(record.id, owner_user_id),
    )


@router.post(
    "/voices",
    response_model=VoiceProfileResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_voice_profile(
    body: VoiceProfileCreate,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        profile = _voice_service(request).create(body, owner_user_id=current_user.id)
        return _profile_response(_voice_service(request), profile, current_user.id)
    except VoiceDatasetStateError as exc:
        status_code = 404 if exc.code == "DATASET_NOT_FOUND" else 409
        return _voice_error(status_code, exc.code, str(exc))


@router.get("/voices", response_model=VoiceListResponse)
def list_voice_profiles(
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
) -> VoiceListResponse:
    return _workspace(request).list_voices(current_user.id)


@router.post(
    "/voices/{profile_id}/references",
    response_model=VoiceProfileResponse,
    status_code=status.HTTP_201_CREATED,
)
def add_voice_reference(
    profile_id: str,
    body: VoiceReferenceCreate,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        service = _voice_service(request)
        profile = service.add_reference(
            profile_id, body, owner_user_id=current_user.id
        )
        return _profile_response(service, profile, current_user.id)
    except KeyError:
        return _voice_error(404, "VOICE_PROFILE_NOT_FOUND", "音色档案不存在")
    except ValueError as exc:
        return _voice_error(409, "REFERENCE_LIMIT_EXCEEDED", str(exc))
    except VoiceDatasetStateError as exc:
        status_code = 404 if exc.code == "DATASET_NOT_FOUND" else 409
        return _voice_error(status_code, exc.code, str(exc))


@router.delete(
    "/voices/{profile_id}/references/{reference_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_voice_reference(
    profile_id: str,
    reference_id: str,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        service = _voice_service(request)
        service.delete_reference(profile_id, reference_id, current_user.id)
        return None
    except KeyError:
        return _voice_error(404, "VOICE_REFERENCE_NOT_FOUND", "参考音频不存在")
    except ValueError as exc:
        return _voice_error(400, "PRIMARY_REFERENCE_CANNOT_BE_DELETED", str(exc))


@router.post("/voices/{profile_id}/train")
def train_voice_profile(
    profile_id: str,
    body: VoiceTrainRequest,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    # Client training is permanently disabled: zero-shot synthesis uses frozen
    # base weights, so no profile is read or mutated and no Job is created.
    del profile_id, body, request, current_user
    raise HTTPException(
        status_code=409,
        detail={
            "code": "CLIENT_TRAINING_DISABLED",
            "message": "客户端使用基础模型零样本合成，不创建本地训练任务",
        },
    )


@router.get("/voices/{profile_id}", response_model=VoiceProfileResponse)
def get_voice_profile(
    profile_id: str,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        service = _voice_service(request)
        profile = service.get(profile_id, owner_user_id=current_user.id)
        return _profile_response(service, profile, current_user.id)
    except KeyError:
        return _voice_error(404, "VOICE_PROFILE_NOT_FOUND", "音色档案不存在")


@router.put("/datasets/{dataset_id}/segments/{segment_id}/emotion")
def override_emotion(
    dataset_id: str,
    segment_id: str,
    body: EmotionOverrideRequest,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        return _store(request).override(
            dataset_id,
            segment_id,
            body,
            owner_user_id=current_user.id,
        )
    except KeyError:
        return JSONResponse(
            status_code=404,
            content={"error": {"code": "SEGMENT_NOT_FOUND", "message": "音频片段不存在"}},
        )
