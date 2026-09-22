import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import Engine, select

from backend.app.api.dependencies import get_current_user
from backend.app.db.models import AudioAsset, Dataset, DatasetSegment, utc_now
from backend.app.db.session import create_database_engine, init_db, session_factory
from backend.app.schemas.auth import UserRecord
from backend.app.schemas.dataset import (
    DatasetPreprocessResponse,
    DatasetSegmentSummary,
    SegmentConfirmRequest,
    SegmentListResponse,
    SegmentReviewRequest,
)
from backend.app.schemas.workspace import DatasetListResponse
from backend.app.services.audio_validation import (
    DEFAULT_SNR_WARNING_DB,
    ReferenceDurationOutOfRange,
    prepare_reference_dataset,
    snr_warning_codes,
    validate_reference_duration,
)
from backend.app.services.storage import LocalStorage
from backend.app.services.transcription import (
    LocalFunASRTranscriber,
    TranscriptionModelUnavailable,
)
from backend.app.services.workspace import WorkspaceService


router = APIRouter(prefix="/api/datasets", tags=["datasets"])


class DatasetAuthorizationRequired(ValueError):
    """Raised when a local dataset operation lacks a recorded authorization."""


class EmotionModelUnavailable(ValueError):
    """Raised when the local emotion model is not prepared under models/."""


class SegmentTextRequired(ValueError):
    """Raised when a manual review would leave a segment without text."""


@dataclass(frozen=True)
class DatasetRecord:
    id: str
    owner_user_id: str | None
    authorization_confirmed: bool
    status: str
    effective_seconds: float | None
    asset_count: int
    asset_relative_paths: tuple[str, ...]
    created_at: datetime


class DatasetService:
    """Persist local upload state without exposing absolute paths or GPU shortcuts."""

    def __init__(
        self,
        storage: LocalStorage | None = None,
        engine: Engine | None = None,
        *,
        transcriber: LocalFunASRTranscriber | None = None,
        emotion_analyzer: Any | None = None,
        snr_warning_db: float = DEFAULT_SNR_WARNING_DB,
    ) -> None:
        self.storage = storage or LocalStorage(Path("data"))
        database_path = self.storage.root / "app.db"
        self.engine = engine or create_database_engine(
            f"sqlite+pysqlite:///{database_path.as_posix()}"
        )
        init_db(self.engine)
        self._session_factory = session_factory(self.engine)
        self.transcriber = transcriber or LocalFunASRTranscriber()
        self.emotion_analyzer = emotion_analyzer
        self.snr_warning_db = float(snr_warning_db)

    def register_dataset(
        self,
        dataset_id: str,
        *,
        effective_seconds: float | None = None,
        owner_user_id: str | None = None,
        authorization_confirmed: bool,
    ) -> DatasetRecord:
        """Test helper that seeds a fully reviewed reference-ready dataset."""
        with self._session_factory() as session:
            dataset = session.get(Dataset, dataset_id)
            if dataset is None:
                dataset = Dataset(
                    id=dataset_id,
                    owner_user_id=owner_user_id,
                    authorization_confirmed_at=(utc_now() if authorization_confirmed else None),
                    effective_seconds=effective_seconds,
                    status="ready_for_profile" if effective_seconds is not None else "created",
                )
                session.add(dataset)
            else:
                if dataset.owner_user_id != owner_user_id:
                    raise KeyError(dataset_id)
                dataset.authorization_confirmed_at = (
                    utc_now() if authorization_confirmed else None
                )
                dataset.effective_seconds = effective_seconds
                dataset.status = (
                    "ready_for_profile" if effective_seconds is not None else "created"
                )
            if effective_seconds is not None:
                # The profile gate re-derives readiness from persisted reviewed
                # segments, so a seeded reference-ready dataset must include one.
                existing = session.get(
                    DatasetSegment, {"dataset_id": dataset_id, "segment_id": "seg_0001"}
                )
                if existing is None:
                    session.add(
                        DatasetSegment(
                            dataset_id=dataset_id,
                            segment_id="seg_0001",
                            owner_user_id=owner_user_id,
                            order_index=0,
                            relative_path=f"datasets/{dataset_id}/segments/seg_0001.wav",
                            duration_seconds=effective_seconds,
                            snr_db=27.0,
                            clipping_ratio=0.0,
                            split="reference",
                            language="zh",
                            auto_transcript="种子审核分段文本。",
                            auto_emotion_label="neutral",
                            auto_emotion_confidence=0.9,
                            reviewed_at=utc_now(),
                        )
                    )
                # A real upload always owns one audio asset; mirror that here so
                # seeded datasets satisfy the primary-reference contract.
                existing_asset = session.scalar(
                    select(AudioAsset).where(AudioAsset.dataset_id == dataset_id)
                )
                if existing_asset is None:
                    session.add(
                        AudioAsset(
                            id=f"asset-{dataset_id}",
                            owner_user_id=owner_user_id,
                            dataset_id=dataset_id,
                            path=f"datasets/{dataset_id}/segments/seg_0001.wav",
                            duration_seconds=effective_seconds,
                        )
                    )
            session.commit()
            return self._record(dataset, [])

    def create_upload(
        self,
        filename: str,
        content: bytes,
        *,
        owner_user_id: str | None = None,
        consent_confirmed: bool = False,
    ) -> dict[str, str]:
        if not consent_confirmed:
            raise DatasetAuthorizationRequired("需要确认拥有参考音频授权")
        suffix = Path(filename).suffix.lower()
        if suffix not in {".wav", ".flac", ".mp3"}:
            raise ValueError("Only WAV, FLAC and MP3 uploads are supported")
        dataset_id = uuid4().hex
        asset_id = uuid4().hex
        content_hash = hashlib.sha256(content).hexdigest()
        relative = Path(dataset_id) / f"{content_hash}{suffix}"
        path = self.storage.resolve("uploads", relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        with self._session_factory() as session:
            session.add(
                Dataset(
                    id=dataset_id,
                    owner_user_id=owner_user_id,
                    authorization_confirmed_at=utc_now(),
                    status="uploaded",
                    effective_seconds=None,
                )
            )
            session.add(
                AudioAsset(
                    id=asset_id,
                    owner_user_id=owner_user_id,
                    dataset_id=dataset_id,
                    path=relative.as_posix(),
                    duration_seconds=None,
                )
            )
            session.commit()
        return {
            "dataset_id": dataset_id,
            "asset_id": asset_id,
            "content_hash": content_hash,
        }

    def list_for_owner(self, owner_user_id: str | None) -> list[DatasetRecord]:
        with self._session_factory() as session:
            datasets = session.scalars(
                select(Dataset)
                .where(Dataset.owner_user_id == owner_user_id)
                .order_by(Dataset.created_at.desc(), Dataset.id.desc())
            ).all()
            return [
                self._record(dataset, self._assets_for_dataset(session, dataset.id))
                for dataset in datasets
            ]

    def get_for_owner(
        self, dataset_id: str, owner_user_id: str | None
    ) -> DatasetRecord:
        with self._session_factory() as session:
            dataset = self._dataset_for_owner(session, dataset_id, owner_user_id)
            return self._record(dataset, self._assets_for_dataset(session, dataset.id))

    def preprocess(
        self, dataset_id: str, *, owner_user_id: str | None = None
    ) -> dict[str, str | float | int | list[str]]:
        with self._session_factory() as session:
            dataset = self._dataset_for_owner(session, dataset_id, owner_user_id)
            if dataset.authorization_confirmed_at is None:
                raise DatasetAuthorizationRequired("需要确认拥有参考音频授权")
            if dataset.status in {"review_required", "ready_for_profile"}:
                # Re-validate even on the fast path so a stale out-of-range
                # duration cannot slip through without a fresh rejection.
                validate_reference_duration(float(dataset.effective_seconds or 0.0))
                segments = self._segments_in_session(session, dataset.id)
                return DatasetPreprocessResponse(
                    dataset_id=dataset.id,
                    effective_seconds=float(dataset.effective_seconds or 0.0),
                    status=dataset.status,
                    segment_count=len(segments),
                    snr_warning_db=self.snr_warning_db,
                    warning_codes=self._warning_codes_for_snr_values(
                        segment.snr_db for segment in segments
                    ),
                ).model_dump(mode="json")
            # Legacy "preprocessed" datasets (before segmentation) fall through
            # to the full pipeline so the reference segment gets persisted text.
            asset_relative_paths = [
                asset.path for asset in self._assets_for_dataset(session, dataset.id)
            ]

        if not self.transcriber.available():
            raise TranscriptionModelUnavailable(
                "本地转写模型未就绪，请先在部署阶段下载到 models/"
            )
        if self.emotion_analyzer is None or not self.emotion_analyzer.available():
            raise EmotionModelUnavailable("本地情绪识别模型未就绪，请先在部署阶段下载到 models/")

        sources = [
            self.storage.resolve("uploads", relative)
            for relative in asset_relative_paths
        ]
        manifest = prepare_reference_dataset(
            dataset_id,
            sources,
            output_dir=self.storage.resolve("datasets", dataset_id),
        )
        effective_seconds = float(manifest.effective_seconds)
        validate_reference_duration(effective_seconds)

        with self._session_factory() as session:
            dataset = self._dataset_for_owner(session, dataset_id, owner_user_id)
            if dataset.authorization_confirmed_at is None:
                raise DatasetAuthorizationRequired("需要确认拥有训练音频授权")
            if manifest.rows:
                self._persist_segments(session, dataset, manifest)
            dataset.effective_seconds = effective_seconds
            dataset.status = "review_required"
            session.commit()
            return DatasetPreprocessResponse(
                dataset_id=dataset.id,
                effective_seconds=effective_seconds,
                status="review_required",
                segment_count=len(manifest.rows),
                snr_warning_db=self.snr_warning_db,
                warning_codes=self._warning_codes_for_snr_values(
                    row.snr_db for row in manifest.rows
                ),
            ).model_dump(mode="json")

    def _persist_segments(self, session, dataset: Dataset, manifest) -> None:
        """Persist one DatasetSegment row per manifest row (transcribe + label)."""
        for order_index, row in enumerate(manifest.rows):
            wav_path = self.storage.resolve("datasets", Path(row.path).relative_to("datasets"))
            transcript = self.transcriber.transcribe(wav_path, language=row.language)
            emotion_result = self.emotion_analyzer.analyze(wav_path)
            label = getattr(emotion_result.label, "value", emotion_result.label)
            confidence = float(emotion_result.confidence)
            if confidence < 0.55:
                label = "other"
            existing = session.get(
                DatasetSegment, {"dataset_id": dataset.id, "segment_id": row.segment_id}
            )
            if existing is not None:
                existing.order_index = order_index
                existing.duration_seconds = row.duration_seconds
                continue
            session.add(
                DatasetSegment(
                    dataset_id=dataset.id,
                    segment_id=row.segment_id,
                    owner_user_id=dataset.owner_user_id,
                    order_index=order_index,
                    relative_path=row.path,
                    duration_seconds=row.duration_seconds,
                    snr_db=row.snr_db,
                    clipping_ratio=row.clipping_ratio,
                    split=row.split,
                    language=row.language,
                    auto_transcript=transcript,
                    auto_emotion_label=label,
                    auto_emotion_confidence=confidence,
                )
            )

    def list_segments(
        self, dataset_id: str, owner_user_id: str | None
    ) -> list[DatasetSegment]:
        with self._session_factory() as session:
            self._dataset_for_owner(session, dataset_id, owner_user_id)
            return list(
                session.scalars(
                    select(DatasetSegment)
                    .where(DatasetSegment.dataset_id == dataset_id)
                    .order_by(DatasetSegment.order_index.asc(), DatasetSegment.segment_id.asc())
                ).all()
            )

    def primary_asset(self, dataset_id: str, owner_user_id: str | None) -> AudioAsset:
        """The dataset's oldest local audio asset (the primary reference)."""
        with self._session_factory() as session:
            self._dataset_for_owner(session, dataset_id, owner_user_id)
            assets = self._assets_for_dataset(session, dataset_id)
            if not assets:
                raise KeyError(dataset_id)
            return assets[0]

    def primary_reference_segment(
        self, dataset_id: str, owner_user_id: str | None
    ) -> DatasetSegment:
        """First reviewed segment with usable text, in dataset order."""
        for segment in self.list_segments(dataset_id, owner_user_id):
            reviewed_text = (
                segment.manual_transcript or segment.auto_transcript or ""
            ).strip()
            if segment.reviewed_at is not None and reviewed_text:
                return segment
        raise KeyError(dataset_id)

    def reference_segment(
        self,
        dataset_id: str,
        segment_id: str,
        owner_user_id: str | None,
    ) -> DatasetSegment:
        with self._session_factory() as session:
            self._dataset_for_owner(session, dataset_id, owner_user_id)
            segment = session.get(
                DatasetSegment,
                {"dataset_id": dataset_id, "segment_id": segment_id},
            )
            if segment is None or segment.owner_user_id != owner_user_id:
                raise KeyError((dataset_id, segment_id))
            session.expunge(segment)
            return segment

    def segment_summary(self, dataset_id: str, owner_user_id: str | None) -> dict:
        with self._session_factory() as session:
            dataset = self._dataset_for_owner(session, dataset_id, owner_user_id)
            segments = self._segments_in_session(session, dataset.id)
            items = [self._segment_record(segment) for segment in segments]
            reviewed = [record for record in items if record.reviewed]
            return {
                "dataset_id": dataset.id,
                "status": dataset.status,
                "total_segments": len(items),
                "reviewed_segments": len(reviewed),
                "effective_seconds": dataset.effective_seconds,
                "reviewed_effective_seconds": round(
                    sum(record.duration_seconds for record in reviewed), 6
                ),
                "ready_for_profile": dataset.status == "ready_for_profile",
                "items": [record.model_dump(mode="json") for record in items],
            }

    def review_segment(
        self,
        dataset_id: str,
        segment_id: str,
        *,
        transcript: str | None,
        emotion_label: str | None,
        reason: str,
        owner_user_id: str | None,
        language: str | None = None,
    ) -> DatasetSegmentSummary:
        with self._session_factory() as session:
            self._dataset_for_owner(session, dataset_id, owner_user_id)
            segment = session.get(
                DatasetSegment, {"dataset_id": dataset_id, "segment_id": segment_id}
            )
            if segment is None:
                raise KeyError((dataset_id, segment_id))
            if transcript is not None and not transcript.strip():
                raise SegmentTextRequired("转写文本不能为空")
            if transcript is not None:
                segment.manual_transcript = transcript.strip()
            if emotion_label is not None:
                segment.manual_emotion_label = emotion_label
            if language is not None:
                segment.language = language
            if transcript is None and emotion_label is None and language is None:
                raise SegmentTextRequired("至少提供转写、情绪或语言修改")
            if segment.manual_transcript is None and not (segment.auto_transcript or "").strip():
                raise SegmentTextRequired("该片段缺少可用文本，请先补充人工校对")
            segment.review_reason = reason
            segment.reviewed_at = utc_now()
            session.commit()
            return self._segment_record(segment)

    def confirm_segments(
        self, dataset_id: str, *, reason: str, owner_user_id: str | None
    ) -> dict:
        """Batch-confirm every segment that has usable text; refresh the gate."""
        with self._session_factory() as session:
            dataset = self._dataset_for_owner(session, dataset_id, owner_user_id)
            segments = self._segments_in_session(session, dataset.id)
            if not segments:
                raise KeyError(dataset_id)
            now = utc_now()
            for segment in segments:
                if (segment.manual_transcript or segment.auto_transcript or "").strip():
                    segment.review_reason = segment.review_reason or reason
                    segment.reviewed_at = now
            reviewed_seconds = sum(
                segment.duration_seconds
                for segment in segments
                if segment.reviewed_at is not None
                and (segment.manual_transcript or segment.auto_transcript or "").strip()
            )
            dataset.effective_seconds = round(reviewed_seconds, 6)
            validate_reference_duration(reviewed_seconds)
            dataset.status = "ready_for_profile"
            session.commit()
            return {
                "dataset_id": dataset.id,
                "status": dataset.status,
                "reviewed_effective_seconds": dataset.effective_seconds,
            }

    def build_training_manifest(self, dataset_id: str, owner_user_id: str | None):
        """Deterministic manifest from reviewed, non-empty-text segments only."""
        from backend.app.schemas.dataset import DatasetManifest, DatasetManifestRow

        with self._session_factory() as session:
            dataset = self._dataset_for_owner(session, dataset_id, owner_user_id)
            segments = self._segments_in_session(session, dataset.id)
            rows = [
                DatasetManifestRow(
                    segment_id=segment.segment_id,
                    path=segment.relative_path,
                    speaker=f"voice_{dataset.id}",
                    language=segment.language,
                    text=(segment.manual_transcript or segment.auto_transcript or "").strip(),
                    duration_seconds=segment.duration_seconds,
                    snr_db=segment.snr_db or 0.0,
                    clipping_ratio=segment.clipping_ratio or 0.0,
                    split=segment.split,
                )
                for segment in segments
                if segment.reviewed_at is not None
                and (segment.manual_transcript or segment.auto_transcript or "").strip()
            ]
            return DatasetManifest(
                dataset_id=dataset.id,
                effective_seconds=sum(row.duration_seconds for row in rows),
                rows=rows,
                manifest_path=self.storage.resolve("datasets", dataset.id) / "manifest.jsonl",
            )

    @staticmethod
    def _segments_in_session(session, dataset_id: str) -> list[DatasetSegment]:
        return list(
            session.scalars(
                select(DatasetSegment)
                .where(DatasetSegment.dataset_id == dataset_id)
                .order_by(DatasetSegment.order_index.asc(), DatasetSegment.segment_id.asc())
            ).all()
        )

    def _warning_codes_for_snr_values(
        self, snr_values: Iterable[float | None]
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    code
                    for snr_db in snr_values
                    for code in snr_warning_codes(snr_db, self.snr_warning_db)
                }
            )
        )

    def _segment_record(self, segment: DatasetSegment) -> DatasetSegmentSummary:
        manual_transcript = (segment.manual_transcript or "").strip() or None
        auto_transcript = (segment.auto_transcript or "").strip()
        effective_transcript = manual_transcript or auto_transcript
        effective_emotion = segment.manual_emotion_label or segment.auto_emotion_label
        return DatasetSegmentSummary(
            segment_id=segment.segment_id,
            order_index=segment.order_index,
            relative_path=segment.relative_path,
            duration_seconds=segment.duration_seconds,
            snr_db=segment.snr_db,
            snr_warning_db=self.snr_warning_db,
            warning_codes=snr_warning_codes(segment.snr_db, self.snr_warning_db),
            language=segment.language,
            split=segment.split,
            auto_transcript=auto_transcript,
            manual_transcript=manual_transcript,
            effective_transcript=effective_transcript or "",
            transcript_source="manual" if manual_transcript else ("auto" if auto_transcript else "missing"),
            auto_emotion_label=segment.auto_emotion_label,
            auto_emotion_confidence=segment.auto_emotion_confidence,
            manual_emotion_label=segment.manual_emotion_label,
            effective_emotion_label=effective_emotion,
            emotion_source="manual" if segment.manual_emotion_label else "auto",
            review_reason=segment.review_reason,
            reviewed=segment.reviewed_at is not None,
        )

    @staticmethod
    def _record(dataset: Dataset, assets: list[AudioAsset]) -> DatasetRecord:
        return DatasetRecord(
            id=dataset.id,
            owner_user_id=dataset.owner_user_id,
            authorization_confirmed=dataset.authorization_confirmed_at is not None,
            status=dataset.status,
            effective_seconds=dataset.effective_seconds,
            asset_count=len(assets),
            asset_relative_paths=tuple(asset.path for asset in assets),
            created_at=dataset.created_at,
        )

    @staticmethod
    def _assets_for_dataset(session, dataset_id: str) -> list[AudioAsset]:
        return session.scalars(
            select(AudioAsset)
            .where(AudioAsset.dataset_id == dataset_id)
            .order_by(AudioAsset.created_at.asc(), AudioAsset.id.asc())
        ).all()

    @staticmethod
    def _dataset_for_owner(session, dataset_id: str, owner_user_id: str | None) -> Dataset:
        dataset = session.scalar(
            select(Dataset).where(
                Dataset.id == dataset_id,
                Dataset.owner_user_id == owner_user_id,
            )
        )
        if dataset is None:
            raise KeyError(dataset_id)
        return dataset


def _error(code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": {},
                "request_id": uuid4().hex,
            }
        },
    )


def _not_found(code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": {},
                "request_id": uuid4().hex,
            }
        },
    )


def _service(request: Request) -> DatasetService:
    service = getattr(request.app.state, "dataset_service", None)
    if service is None:
        service = DatasetService()
        request.app.state.dataset_service = service
    return service


def _workspace(request: Request) -> WorkspaceService:
    return request.app.state.workspace_service


@router.post("")
async def create_dataset(
    request: Request,
    file: UploadFile = File(...),
    consent_confirmed: bool = Form(False),
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        return _service(request).create_upload(
            file.filename or "upload.wav",
            await file.read(),
            owner_user_id=current_user.id,
            consent_confirmed=consent_confirmed,
        )
    except DatasetAuthorizationRequired as exc:
        return _error("DATASET_AUTHORIZATION_REQUIRED", str(exc))
    except ValueError as exc:
        return _error("UNSUPPORTED_AUDIO_FORMAT", str(exc))


@router.get("", response_model=DatasetListResponse)
def list_datasets(
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
) -> DatasetListResponse:
    return _workspace(request).list_datasets(current_user.id)


@router.post("/{dataset_id}/preprocess", response_model=DatasetPreprocessResponse)
def preprocess_dataset(
    dataset_id: str,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        return _service(request).preprocess(dataset_id, owner_user_id=current_user.id)
    except DatasetAuthorizationRequired as exc:
        return _error("DATASET_AUTHORIZATION_REQUIRED", str(exc))
    except ReferenceDurationOutOfRange as exc:
        return _error("REFERENCE_DURATION_OUT_OF_RANGE", str(exc))
    except TranscriptionModelUnavailable as exc:
        return _error("TRANSCRIPTION_MODEL_UNAVAILABLE", str(exc))
    except EmotionModelUnavailable as exc:
        return _error("EMOTION_MODEL_UNAVAILABLE", str(exc))
    except KeyError:
        return _error("DATASET_NOT_FOUND", "数据集不存在")


@router.get("/{dataset_id}/segments", response_model=SegmentListResponse)
def list_dataset_segments(
    dataset_id: str,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        return SegmentListResponse(**_service(request).segment_summary(dataset_id, current_user.id))
    except KeyError:
        return _not_found("DATASET_NOT_FOUND", "数据集不存在")


@router.patch("/{dataset_id}/segments/{segment_id}", response_model=DatasetSegmentSummary)
def review_dataset_segment(
    dataset_id: str,
    segment_id: str,
    body: SegmentReviewRequest,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        return _service(request).review_segment(
            dataset_id,
            segment_id,
            transcript=body.transcript,
            emotion_label=body.emotion_label.value if body.emotion_label else None,
            reason=body.reason,
            owner_user_id=current_user.id,
            language=body.language,
        )
    except SegmentTextRequired as exc:
        return _error("SEGMENT_TEXT_REQUIRED", str(exc))
    except KeyError:
        return _not_found("SEGMENT_NOT_FOUND", "分段不存在")


@router.post("/{dataset_id}/segments/confirm")
def confirm_dataset_segments(
    dataset_id: str,
    body: SegmentConfirmRequest,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        return _service(request).confirm_segments(
            dataset_id, reason=body.reason, owner_user_id=current_user.id
        )
    except ReferenceDurationOutOfRange as exc:
        return _error("REFERENCE_DURATION_OUT_OF_RANGE", str(exc))
    except KeyError:
        return _error("DATASET_NOT_FOUND", "数据集不存在")
