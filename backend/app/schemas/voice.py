from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.schemas.common import EmotionLabel


class VoiceProfileCreate(BaseModel):
    dataset_id: str = Field(min_length=1, max_length=64)
    display_name: str = Field(min_length=1, max_length=64)
    reference_name: str | None = Field(default=None, max_length=64)

    @field_validator("display_name")
    @classmethod
    def normalize_display_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("音色名称不能为空")
        return normalized


class VoiceReferenceCreate(BaseModel):
    dataset_id: str = Field(min_length=1, max_length=64)
    reference_name: str | None = Field(default=None, max_length=64)


class VoiceTrainRequest(BaseModel):
    consent_confirmed: Literal[True]


class VoiceReferenceItemResponse(BaseModel):
    id: str
    asset_id: str
    segment_id: str
    prompt_text: str
    prompt_language: str
    emotion_label: str
    emotion_confidence: float | None = None
    is_primary: bool
    reference_name: str | None = None
    snr_db: float | None = None
    created_at: str


class VoiceProfileResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    status: str
    mode: Literal["zero_shot"]
    base_model_id: str
    display_name: str
    reference_emotions: list[EmotionLabel] = []
    reference_snr_db: float | None = None
    quality_warning_codes: list[str] = []
    references: list[VoiceReferenceItemResponse] = []


class VoiceTrainResponse(BaseModel):
    job_id: str
    status: Literal["queued"]
