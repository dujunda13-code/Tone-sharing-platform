from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.app.schemas.common import EmotionLabel


class AudioInspection(BaseModel):
    path: Path
    sample_rate: int
    channels: int
    duration_seconds: float
    snr_db: float
    clipping_ratio: float
    rejection_codes: tuple[str, ...] = ()
    warning_codes: tuple[str, ...] = ()


class DatasetManifestRow(BaseModel):
    segment_id: str
    path: str
    speaker: str
    language: Literal["zh", "en"] = "zh"
    text: str = ""
    duration_seconds: float = Field(gt=0)
    snr_db: float
    clipping_ratio: float
    warning_codes: tuple[str, ...] = ()
    # "reference" marks zero-shot reference rows; legacy training-era rows
    # keep their train/validation/test values for diagnostics only.
    split: Literal["train", "validation", "test", "reference"]


class DatasetManifest(BaseModel):
    dataset_id: str
    effective_seconds: float
    rows: list[DatasetManifestRow]
    manifest_path: Path


class DatasetPreprocessResponse(BaseModel):
    dataset_id: str
    effective_seconds: float
    status: str
    segment_count: int = 0
    snr_warning_db: float
    warning_codes: tuple[str, ...] = ()


class DatasetSegmentSummary(BaseModel):
    segment_id: str
    order_index: int
    relative_path: str
    duration_seconds: float
    snr_db: float | None
    snr_warning_db: float
    warning_codes: tuple[str, ...] = ()
    language: str
    split: str
    auto_transcript: str
    manual_transcript: str | None
    effective_transcript: str
    transcript_source: str
    auto_emotion_label: str | None
    auto_emotion_confidence: float | None
    manual_emotion_label: str | None
    effective_emotion_label: str | None
    emotion_source: str
    review_reason: str | None
    reviewed: bool


class SegmentListResponse(BaseModel):
    dataset_id: str
    status: str
    total_segments: int
    reviewed_segments: int
    effective_seconds: float | None
    reviewed_effective_seconds: float
    ready_for_profile: bool
    items: list[DatasetSegmentSummary]


class SegmentReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcript: str | None = Field(default=None, max_length=2000)
    emotion_label: EmotionLabel | None = None
    language: Literal["zh", "en"] | None = None
    reason: str = Field(min_length=1, max_length=500)


class SegmentConfirmRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1, max_length=500)
