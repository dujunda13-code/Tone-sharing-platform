from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from backend.app.schemas.common import JobKind, JobStatus
from backend.app.schemas.voice import VoiceReferenceItemResponse


class DatasetSummary(BaseModel):
    id: str
    authorization_confirmed: bool
    status: str
    effective_seconds: float | None
    asset_count: int
    created_at: datetime


class VoiceSummary(BaseModel):
    id: str
    dataset_id: str
    display_name: str
    status: str
    created_at: datetime
    can_synthesize: bool
    mode: Literal["zero_shot"] | None = None
    base_model_id: str | None = None
    reference_emotions: list[str] = []
    reference_snr_db: float | None = None
    quality_warning_codes: list[str] = []
    references: list[VoiceReferenceItemResponse] = []


class JobSummary(BaseModel):
    id: str
    kind: JobKind
    status: JobStatus
    error_code: str | None
    public_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class SynthesisSummary(BaseModel):
    job_id: str
    voice_profile_id: str | None
    text_lang: Literal["zh", "en"] | None
    status: JobStatus
    download_ready: bool
    watermark_probability: float | None
    fingerprint_anomaly: bool | None
    speaker_similarity: float | None = None
    quality_warning_codes: list[str] = []
    progress_message: str | None = None
    created_at: datetime


class ReadinessCheckSummary(BaseModel):
    ok: bool
    message: str


class ReadinessSummary(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, ReadinessCheckSummary]


class DashboardCounts(BaseModel):
    datasets: int
    voices: int
    jobs: int
    active_jobs: int
    syntheses: int


class DashboardResponse(BaseModel):
    counts: DashboardCounts
    recent_jobs: list[JobSummary]
    voices: list[VoiceSummary]
    readiness: ReadinessSummary


class DatasetListResponse(BaseModel):
    items: list[DatasetSummary]


class VoiceListResponse(BaseModel):
    items: list[VoiceSummary]


class JobListResponse(BaseModel):
    items: list[JobSummary]


class SynthesisListResponse(BaseModel):
    items: list[SynthesisSummary]
