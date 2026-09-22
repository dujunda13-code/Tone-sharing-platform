from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from backend.app.core.errors import PathOutsideStorage
from backend.app.db.models import (
    AudioAsset,
    Dataset,
    DatasetSegment,
    Job,
    VoiceProfile,
    VoiceReference,
)
from backend.app.db.session import session_factory
from backend.app.schemas.common import JobKind, JobStatus
from backend.app.schemas.voice import VoiceReferenceItemResponse
from backend.app.schemas.workspace import (
    DashboardCounts,
    DashboardResponse,
    DatasetListResponse,
    DatasetSummary,
    JobListResponse,
    JobSummary,
    ReadinessSummary,
    SynthesisListResponse,
    SynthesisSummary,
    VoiceListResponse,
    VoiceSummary,
)
from backend.app.services.audio_validation import (
    DEFAULT_SNR_WARNING_DB,
    snr_warning_codes,
)
from backend.app.services.storage import LocalStorage
from backend.app.services.voice_profiles import public_voice_name


PUBLIC_JOB_MESSAGES = {
    "SYNTHESIS_PIPELINE_UNAVAILABLE": "本机合成前置条件未满足",
    "GPU_OUT_OF_MEMORY": "GPU 显存不足，任务已停止",
    "QUALITY_GATE_FAILED": "质量评测未达到发布门槛",
    "CLIENT_TRAINING_DISABLED": "客户端使用基础模型零样本合成，不创建本地训练任务",
}


def public_job_message(error_code: str | None) -> str | None:
    if error_code is None:
        return None
    return PUBLIC_JOB_MESSAGES.get(error_code, "任务未完成，请查看错误代码")


class WorkspaceService:
    """Read owner-scoped, presentation-safe summaries from local SQLite state."""

    def __init__(
        self,
        *,
        engine: Engine,
        storage: LocalStorage,
        configured_base_model_ids: frozenset[str] | None = None,
        snr_warning_db: float = DEFAULT_SNR_WARNING_DB,
    ) -> None:
        self.storage = storage
        self.configured_base_model_ids = configured_base_model_ids or frozenset()
        self.snr_warning_db = float(snr_warning_db)
        self._session_factory = session_factory(engine)

    def list_datasets(self, user_id: str) -> DatasetListResponse:
        with self._session_factory() as session:
            datasets = session.scalars(
                select(Dataset)
                .where(Dataset.owner_user_id == user_id)
                .order_by(Dataset.created_at.desc(), Dataset.id.desc())
            ).all()
            counts = self._asset_counts(session, [dataset.id for dataset in datasets])
            return DatasetListResponse(
                items=[
                    DatasetSummary(
                        id=dataset.id,
                        authorization_confirmed=dataset.authorization_confirmed_at is not None,
                        status=dataset.status,
                        effective_seconds=dataset.effective_seconds,
                        asset_count=counts.get(dataset.id, 0),
                        created_at=dataset.created_at,
                    )
                    for dataset in datasets
                ]
            )

    def list_voices(self, user_id: str) -> VoiceListResponse:
        with self._session_factory() as session:
            profiles = session.scalars(
                select(VoiceProfile)
                .where(VoiceProfile.owner_user_id == user_id)
                .order_by(VoiceProfile.created_at.desc(), VoiceProfile.id.desc())
            ).all()
            references_by_profile = self._references_by_profile(
                session,
                [profile.id for profile in profiles if profile.mode == "zero_shot"],
            )
            quality_by_profile = self._reference_quality_by_profile(
                session, profiles, user_id
            )
            return VoiceListResponse(
                items=[
                    self._voice_summary(
                        profile,
                        references_by_profile.get(profile.id, []),
                        quality_by_profile.get(profile.id),
                    )
                    for profile in profiles
                ]
            )

    def _references_by_profile(
        self, session: Session, profile_ids: list[str]
    ) -> dict[str, list[VoiceReference]]:
        if not profile_ids:
            return {}
        rows = session.scalars(
            select(VoiceReference)
            .where(VoiceReference.profile_id.in_(profile_ids))
            .order_by(VoiceReference.is_primary.desc(), VoiceReference.created_at.asc())
        ).all()
        grouped: dict[str, list[VoiceReference]] = {}
        for reference in rows:
            grouped.setdefault(reference.profile_id, []).append(reference)
        return grouped

    def _reference_quality_by_profile(
        self,
        session: Session,
        profiles: list[VoiceProfile],
        user_id: str,
    ) -> dict[str, float | None]:
        profile_keys = {
            profile.id: (profile.dataset_id, profile.reference_segment_id)
            for profile in profiles
            if profile.mode == "zero_shot"
            and profile.dataset_id
            and profile.reference_segment_id
        }
        if not profile_keys:
            return {}
        dataset_ids = {key[0] for key in profile_keys.values()}
        segment_ids = {key[1] for key in profile_keys.values()}
        rows = session.execute(
            select(
                DatasetSegment.dataset_id,
                DatasetSegment.segment_id,
                DatasetSegment.snr_db,
            ).where(
                DatasetSegment.owner_user_id == user_id,
                DatasetSegment.dataset_id.in_(dataset_ids),
                DatasetSegment.segment_id.in_(segment_ids),
            )
        ).all()
        snr_by_key = {
            (dataset_id, segment_id): snr_db
            for dataset_id, segment_id, snr_db in rows
        }
        return {
            profile_id: snr_by_key[key]
            for profile_id, key in profile_keys.items()
            if key in snr_by_key
        }

    def list_jobs(
        self, user_id: str, status: JobStatus | None = None
    ) -> JobListResponse:
        with self._session_factory() as session:
            jobs = self._jobs_for_owner(session, user_id, status=status)
            return JobListResponse(items=[self._job_summary(job) for job in jobs])

    def get_job_summary(self, user_id: str, job_id: str) -> JobSummary:
        with self._session_factory() as session:
            job = self._job_for_owner(session, user_id, job_id)
            return self._job_summary(job)

    def list_syntheses(self, user_id: str) -> SynthesisListResponse:
        with self._session_factory() as session:
            jobs = self._jobs_for_owner(session, user_id, kind=JobKind.SYNTHESIZE)
            return SynthesisListResponse(
                items=[self._synthesis_summary(job) for job in jobs]
            )

    def get_synthesis_summary(self, user_id: str, job_id: str) -> SynthesisSummary:
        with self._session_factory() as session:
            job = self._job_for_owner(session, user_id, job_id)
            if job.kind != JobKind.SYNTHESIZE.value:
                raise KeyError(job_id)
            return self._synthesis_summary(job)

    def dashboard(self, user_id: str, readiness: ReadinessSummary) -> DashboardResponse:
        datasets = self.list_datasets(user_id)
        voices = self.list_voices(user_id)
        jobs = self.list_jobs(user_id)
        syntheses = self.list_syntheses(user_id)
        return DashboardResponse(
            counts=DashboardCounts(
                datasets=len(datasets.items),
                voices=len(voices.items),
                jobs=len(jobs.items),
                active_jobs=sum(
                    job.status in {JobStatus.QUEUED, JobStatus.RUNNING}
                    for job in jobs.items
                ),
                syntheses=len(syntheses.items),
            ),
            recent_jobs=jobs.items[:5],
            voices=voices.items,
            readiness=readiness,
        )

    @staticmethod
    def _asset_counts(session: Session, dataset_ids: list[str]) -> dict[str, int]:
        if not dataset_ids:
            return {}
        rows = session.execute(
            select(AudioAsset.dataset_id, func.count(AudioAsset.id))
            .where(AudioAsset.dataset_id.in_(dataset_ids))
            .group_by(AudioAsset.dataset_id)
        ).all()
        return {dataset_id: int(count) for dataset_id, count in rows}

    def _voice_summary(
        self,
        profile: VoiceProfile,
        references: list[VoiceReference],
        reference_snr_db: float | None = None,
    ) -> VoiceSummary:
        if profile.mode == "zero_shot":
            # Zero-shot profiles synthesize from persisted references plus the
            # configured frozen base weights, never from a personal weight dir.
            has_primary = any(reference.is_primary for reference in references)
            base_configured = profile.base_model_id in self.configured_base_model_ids
            references_items = [
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
                    created_at=ref.created_at.isoformat(),
                )
                for ref in references
            ]
            return VoiceSummary(
                id=profile.id,
                dataset_id=profile.dataset_id or "",
                display_name=public_voice_name(profile.id, profile.display_name),
                status=profile.status,
                created_at=profile.created_at,
                can_synthesize=(
                    profile.status == "ready" and has_primary and base_configured
                ),
                mode="zero_shot",
                base_model_id=profile.base_model_id,
                reference_emotions=[reference.emotion_label for reference in references],
                reference_snr_db=reference_snr_db,
                quality_warning_codes=list(
                    snr_warning_codes(reference_snr_db, self.snr_warning_db)
                ),
                references=references_items,
            )
        weights_available = bool(
            profile.public_weight_dir and Path(profile.public_weight_dir).is_dir()
        )
        return VoiceSummary(
            id=profile.id,
            dataset_id=profile.dataset_id or "",
            display_name=public_voice_name(profile.id, profile.display_name),
            status=profile.status,
            created_at=profile.created_at,
            can_synthesize=profile.status == "ready" and weights_available,
        )

    def _jobs_for_owner(
        self,
        session: Session,
        user_id: str,
        *,
        kind: JobKind | None = None,
        status: JobStatus | None = None,
    ) -> list[Job]:
        statement = select(Job).where(Job.owner_user_id == user_id)
        if kind is not None:
            statement = statement.where(Job.kind == kind.value)
        if status is not None:
            statement = statement.where(Job.status == status.value)
        return session.scalars(
            statement.order_by(Job.created_at.desc(), Job.id.desc())
        ).all()

    @staticmethod
    def _job_for_owner(session: Session, user_id: str, job_id: str) -> Job:
        job = session.scalar(
            select(Job).where(Job.id == job_id, Job.owner_user_id == user_id)
        )
        if job is None:
            raise KeyError(job_id)
        return job

    @staticmethod
    def _mapping(raw_json: str | None) -> dict[str, Any]:
        if not raw_json:
            return {}
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _job_summary(job: Job) -> JobSummary:
        return JobSummary(
            id=job.id,
            kind=JobKind(job.kind),
            status=JobStatus(job.status),
            error_code=job.error_code,
            public_message=public_job_message(job.error_code),
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )

    def _synthesis_summary(self, job: Job) -> SynthesisSummary:
        payload = self._mapping(job.payload_json)
        result = self._mapping(job.result_json)
        text_lang = payload.get("text_lang")
        if text_lang not in {"zh", "en"}:
            text_lang = None
        voice_profile_id = payload.get("voice_profile_id")
        if not isinstance(voice_profile_id, str):
            voice_profile_id = None
        watermark_probability = result.get("watermark_probability")
        if isinstance(watermark_probability, bool) or not isinstance(
            watermark_probability, (float, int)
        ):
            watermark_probability = None
        else:
            watermark_probability = float(watermark_probability)
        fingerprint = result.get("fingerprint")
        fingerprint_anomaly = (
            fingerprint.get("anomaly")
            if isinstance(fingerprint, dict)
            and isinstance(fingerprint.get("anomaly"), bool)
            else None
        )
        speaker_similarity = result.get("speaker_similarity")
        if isinstance(speaker_similarity, bool) or not isinstance(
            speaker_similarity, (float, int)
        ):
            speaker_similarity = None
        else:
            speaker_similarity = float(speaker_similarity)
        raw_warning_codes = result.get("quality_warning_codes")
        quality_warning_codes = (
            [str(code) for code in raw_warning_codes]
            if isinstance(raw_warning_codes, list)
            else []
        )
        return SynthesisSummary(
            job_id=job.id,
            voice_profile_id=voice_profile_id,
            text_lang=text_lang,
            status=JobStatus(job.status),
            download_ready=self._verified_output_exists(job, result),
            watermark_probability=watermark_probability,
            fingerprint_anomaly=fingerprint_anomaly,
            speaker_similarity=speaker_similarity,
            quality_warning_codes=quality_warning_codes,
            progress_message=job.progress_message
            if job.status == JobStatus.RUNNING.value
            else None,
            created_at=job.created_at,
        )

    def _verified_output_exists(self, job: Job, result: dict[str, Any]) -> bool:
        if job.status != JobStatus.SUCCEEDED.value:
            return False
        relative_path = result.get("public_audio_path")
        if not isinstance(relative_path, str) or not relative_path:
            return False
        try:
            return self.storage.resolve("outputs", relative_path).is_file()
        except (OSError, PathOutsideStorage):
            return False
