from __future__ import annotations

import io
import json
import zipfile
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.db.models import (
    AudioAsset,
    Dataset,
    DatasetSegment,
    PlazaPost,
    VoiceProfile,
    VoiceReference,
    utc_now,
)
from backend.app.db.session import create_database_engine, init_db, session_factory
from backend.app.services.plaza import PlazaError
from backend.app.services.storage import LocalStorage
from backend.app.services.voice_profiles import VoiceProfileRecord, VoiceProfileStore, public_voice_name


class VoicePackageService:
    """Build downloadable voice packages and fork imports of plaza posts."""

    def __init__(
        self,
        *,
        engine: Engine | None = None,
        storage: LocalStorage | None = None,
        profiles: VoiceProfileStore | None = None,
    ) -> None:
        self.engine = engine or create_database_engine("sqlite:///data/app.db")
        init_db(self.engine)
        self._session_factory: sessionmaker[Session] = session_factory(self.engine)
        self.storage = storage or LocalStorage("data")
        self.profiles = profiles or VoiceProfileStore(engine=self.engine)

    def _post_profile(self, session: Session, post_id: str) -> tuple[PlazaPost, VoiceProfile]:
        post = session.get(PlazaPost, post_id)
        if post is None:
            raise KeyError(post_id)
        profile = session.get(VoiceProfile, post.voice_profile_id)
        if profile is None:
            raise KeyError(post_id)
        if profile.status != "ready":
            raise PlazaError("PLAZA_VOICE_PROFILE_NOT_READY", "音色档案尚不可用")
        return post, profile

    def _reference_files(
        self, session: Session, post: PlazaPost, profile: VoiceProfile
    ) -> list[tuple[VoiceReference, Path]]:
        references = session.scalars(
            select(VoiceReference)
            .where(VoiceReference.profile_id == profile.id)
            .order_by(VoiceReference.is_primary.desc(), VoiceReference.created_at.asc())
        ).all()
        pairs: list[tuple[VoiceReference, Path]] = []
        for reference in references:
            segment = session.scalar(
                select(DatasetSegment).where(
                    DatasetSegment.dataset_id == profile.dataset_id,
                    DatasetSegment.segment_id == reference.segment_id,
                )
            )
            if segment is None:
                raise PlazaError("PLAZA_REFERENCE_AUDIO_MISSING", "参考音频分段不存在")
            try:
                path = self.storage.resolve(
                    "datasets", Path(segment.relative_path).relative_to("datasets")
                )
            except ValueError as exc:
                raise PlazaError("PLAZA_REFERENCE_AUDIO_MISSING", "参考音频路径越界") from exc
            if not path.is_file():
                raise PlazaError("PLAZA_REFERENCE_AUDIO_MISSING", "参考音频文件不存在")
            pairs.append((reference, path))
        if not pairs or not pairs[0][0].is_primary:
            raise PlazaError("PLAZA_REFERENCE_AUDIO_MISSING", "主参考音频缺失")
        return pairs

    def build_package(self, post_id: str) -> bytes:
        with self._session_factory() as session:
            post, profile = self._post_profile(session, post_id)
            pairs = self._reference_files(session, post, profile)
            metadata = {
                "format_version": 1,
                "voice": {
                    "voice_profile_id": profile.id,
                    "display_name": public_voice_name(profile.id, profile.display_name),
                    "mode": profile.mode,
                    "base_model_id": profile.base_model_id,
                    "prompt_text": profile.prompt_text,
                    "prompt_language": profile.prompt_language,
                },
                "post": {
                    "post_id": post.id,
                    "description": post.description,
                    "created_at": post.created_at.isoformat(),
                },
                "references": [
                    {
                        "segment_id": reference.segment_id,
                        "prompt_text": reference.prompt_text,
                        "prompt_language": reference.prompt_language,
                        "emotion_label": reference.emotion_label,
                        "emotion_confidence": reference.emotion_confidence,
                        "is_primary": bool(reference.is_primary),
                        "sha256": sha256(path.read_bytes()).hexdigest(),
                        "file": (
                            f"references/primary-{reference.segment_id}.wav"
                            if reference.is_primary
                            else f"references/auxiliary-{reference.segment_id}.wav"
                        ),
                    }
                    for reference, path in pairs
                ],
            }
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(
                    "metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2)
                )
                for reference, path in pairs:
                    name = (
                        f"primary-{reference.segment_id}.wav"
                        if reference.is_primary
                        else f"auxiliary-{reference.segment_id}.wav"
                    )
                    archive.writestr(f"references/{name}", path.read_bytes())
            return buffer.getvalue()

    def reference_wav_path(self, post_id: str) -> Path:
        with self._session_factory() as session:
            post, profile = self._post_profile(session, post_id)
            pairs = self._reference_files(session, post, profile)
            return pairs[0][1]

    def import_fork(
        self,
        post_id: str,
        *,
        recipient_user_id: str,
        authorization_confirmed: bool,
        fallback_username: str,
    ) -> VoiceProfileRecord:
        if authorization_confirmed is not True:
            raise PlazaError("AUTHORIZATION_REQUIRED", "需要确认已获得该音色的使用授权")
        with self._session_factory() as session:
            post, profile = self._post_profile(session, post_id)
            pairs = self._reference_files(session, post, profile)
            source_dataset = session.get(Dataset, profile.dataset_id)
            effective_seconds = (
                source_dataset.effective_seconds if source_dataset is not None else None
            )
            payloads = []
            for index, (reference, path) in enumerate(pairs):
                content = path.read_bytes()
                segment = session.scalar(
                    select(DatasetSegment).where(
                        DatasetSegment.dataset_id == profile.dataset_id,
                        DatasetSegment.segment_id == reference.segment_id,
                    )
                )
                payloads.append((reference, segment, content, index))
            if effective_seconds is None:
                effective_seconds = float(
                    sum((segment.duration_seconds or 0.0) for _, segment, _, _ in payloads)
                )
            new_dataset_id = uuid4().hex
            asset_ids: dict[str, str] = {}
            for reference, segment, content, index in payloads:
                destination = self.storage.resolve(
                    "datasets",
                    Path(new_dataset_id) / "segments" / f"{reference.segment_id}.wav",
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
                if sha256(destination.read_bytes()).hexdigest() != sha256(content).hexdigest():
                    raise PlazaError("REFERENCE_COPY_INTEGRITY_FAILED", "参考音频复制校验失败")
                asset_ids[reference.segment_id] = uuid4().hex
                session.add(
                    AudioAsset(
                        id=asset_ids[reference.segment_id],
                        owner_user_id=recipient_user_id,
                        dataset_id=new_dataset_id,
                        path=f"datasets/{new_dataset_id}/segments/{reference.segment_id}.wav",
                        duration_seconds=segment.duration_seconds if segment else None,
                    )
                )
                session.add(
                    DatasetSegment(
                        dataset_id=new_dataset_id,
                        segment_id=reference.segment_id,
                        owner_user_id=recipient_user_id,
                        order_index=index,
                        relative_path=f"datasets/{new_dataset_id}/segments/{reference.segment_id}.wav",
                        duration_seconds=(segment.duration_seconds if segment else 0.0),
                        snr_db=(segment.snr_db if segment else None),
                        clipping_ratio=(segment.clipping_ratio if segment else None),
                        split=(segment.split if segment else "reference"),
                        language=(segment.language if segment else "zh"),
                        auto_transcript=(segment.auto_transcript if segment else ""),
                        manual_transcript=(segment.manual_transcript if segment else None),
                        auto_emotion_label=(segment.auto_emotion_label if segment else None),
                        auto_emotion_confidence=(segment.auto_emotion_confidence if segment else None),
                        manual_emotion_label=(segment.manual_emotion_label if segment else None),
                        review_reason=(segment.review_reason if segment else None),
                        reviewed_at=(segment.reviewed_at if segment else utc_now()),
                    )
                )
            session.add(
                Dataset(
                    id=new_dataset_id,
                    owner_user_id=recipient_user_id,
                    authorization_confirmed_at=utc_now(),
                    effective_seconds=effective_seconds,
                    status="ready_for_profile",
                )
            )
            session.commit()
        primary = next(reference for reference, _, _, _ in payloads if reference.is_primary)
        forked = self.profiles.create_zero_shot(
            dataset_id=new_dataset_id,
            owner_user_id=recipient_user_id,
            reference_asset_id=asset_ids[primary.segment_id],
            reference_segment_id=primary.segment_id,
            prompt_text=primary.prompt_text,
            prompt_language=primary.prompt_language,
            emotion_label=primary.emotion_label,
            emotion_confidence=primary.emotion_confidence,
            base_model_id=profile.base_model_id or "",
            display_name=public_voice_name(profile.id, profile.display_name),
            reference_name=primary.reference_name,
        )
        for reference, _, _, _ in payloads:
            if reference.is_primary:
                continue
            self.profiles.add_reference(
                forked.id,
                recipient_user_id,
                asset_id=asset_ids[reference.segment_id],
                segment_id=reference.segment_id,
                prompt_text=reference.prompt_text,
                prompt_language=reference.prompt_language,
                emotion_label=reference.emotion_label,
                emotion_confidence=reference.emotion_confidence,
                reference_name=reference.reference_name,
            )
        return forked
