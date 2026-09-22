from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.engine import Engine

from backend.app.db.models import VoiceProfile, VoiceReference
from backend.app.db.session import create_database_engine, init_db, session_factory
from backend.app.schemas.common import EmotionLabel


class VoiceProfileStateError(ValueError):
    """Raised when a persisted voice profile leaves its fixed lifecycle."""


VALID_PROMPT_LANGUAGES = {"zh", "en"}


@dataclass(frozen=True)
class VoiceProfileRecord:
    id: str
    owner_user_id: str | None
    dataset_id: str
    status: str
    public_weight_dir: str | None
    created_at: datetime
    mode: str | None = None
    reference_asset_id: str | None = None
    reference_segment_id: str | None = None
    prompt_text: str | None = None
    prompt_language: str | None = None
    base_model_id: str | None = None
    display_name: str | None = None


def public_voice_name(profile_id: str, display_name: str | None) -> str:
    """One stable public name for every profile, keyed by its immutable id."""
    normalized = (display_name or "").strip()
    return normalized or f"音色 {profile_id[:6]}"


@dataclass(frozen=True)
class VoiceReferenceRecord:
    id: str
    profile_id: str
    owner_user_id: str | None
    asset_id: str
    segment_id: str
    prompt_text: str
    prompt_language: str
    emotion_label: str
    emotion_confidence: float | None
    is_primary: bool
    created_at: datetime
    reference_name: str | None = None


def _validated_prompt_text(prompt_text: str) -> str:
    normalized = (prompt_text or "").strip()
    if not normalized:
        raise ValueError("参考文本必须非空")
    return normalized


def _validated_prompt_language(prompt_language: str) -> str:
    if prompt_language not in VALID_PROMPT_LANGUAGES:
        raise ValueError(f"参考语言只允许 {'、'.join(sorted(VALID_PROMPT_LANGUAGES))}")
    return prompt_language


def _validated_emotion_label(emotion_label: str) -> str:
    try:
        return EmotionLabel(emotion_label).value
    except ValueError as exc:
        raise ValueError(f"未知情绪标签: {emotion_label}") from exc


def _require_identifier(value: str, label: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        raise ValueError(f"{label}必须非空")
    return normalized


class VoiceProfileStore:
    """Persist local voice-profile lifecycle state in the project's SQLite database."""

    def __init__(self, engine: Engine | None = None) -> None:
        self.engine = engine or create_database_engine("sqlite:///data/app.db")
        init_db(self.engine)
        self._session_factory = session_factory(self.engine)

    @staticmethod
    def _record(profile: VoiceProfile) -> VoiceProfileRecord:
        if not profile.dataset_id:
            raise VoiceProfileStateError(f"voice profile {profile.id!r} has no dataset")
        return VoiceProfileRecord(
            id=profile.id,
            owner_user_id=profile.owner_user_id,
            dataset_id=profile.dataset_id,
            status=profile.status,
            public_weight_dir=profile.public_weight_dir,
            created_at=profile.created_at,
            mode=profile.mode,
            reference_asset_id=profile.reference_asset_id,
            reference_segment_id=profile.reference_segment_id,
            prompt_text=profile.prompt_text,
            prompt_language=profile.prompt_language,
            base_model_id=profile.base_model_id,
            display_name=profile.display_name,
        )

    @staticmethod
    def _reference_record(reference: VoiceReference) -> VoiceReferenceRecord:
        return VoiceReferenceRecord(
            id=reference.id,
            profile_id=reference.profile_id,
            owner_user_id=reference.owner_user_id,
            asset_id=reference.asset_id,
            segment_id=reference.segment_id,
            prompt_text=reference.prompt_text,
            prompt_language=reference.prompt_language,
            emotion_label=reference.emotion_label,
            emotion_confidence=reference.emotion_confidence,
            is_primary=bool(reference.is_primary),
            created_at=reference.created_at,
            reference_name=reference.reference_name,
        )

    def create_zero_shot(
        self,
        *,
        dataset_id: str,
        owner_user_id: str,
        reference_asset_id: str,
        reference_segment_id: str,
        prompt_text: str,
        prompt_language: str,
        emotion_label: str,
        emotion_confidence: float | None,
        base_model_id: str,
        display_name: str,
        reference_name: str | None = None,
    ) -> VoiceProfileRecord:
        """Create a ready zero-shot profile and its primary reference atomically."""
        normalized_dataset_id = _require_identifier(dataset_id, "dataset id")
        normalized_owner = _require_identifier(owner_user_id, "owner user id")
        normalized_asset_id = _require_identifier(reference_asset_id, "reference asset id")
        normalized_display_name = _require_identifier(display_name, "display name")
        if len(normalized_display_name) > 64:
            raise ValueError("音色名称长度需在 64 个字符以内")
        normalized_segment_id = _require_identifier(reference_segment_id, "reference segment id")
        normalized_base_model_id = _require_identifier(base_model_id, "base model id")
        validated_text = _validated_prompt_text(prompt_text)
        validated_language = _validated_prompt_language(prompt_language)
        validated_emotion = _validated_emotion_label(emotion_label)

        with self._session_factory() as session:
            profile = VoiceProfile(
                id=uuid4().hex,
                owner_user_id=normalized_owner,
                dataset_id=normalized_dataset_id,
                status="ready",
                public_weight_dir=None,
                mode="zero_shot",
                reference_asset_id=normalized_asset_id,
                reference_segment_id=normalized_segment_id,
                prompt_text=validated_text,
                prompt_language=validated_language,
                base_model_id=normalized_base_model_id,
                display_name=normalized_display_name,
            )
            session.add(profile)
            session.flush()
            session.add(
                VoiceReference(
                    id=uuid4().hex,
                    profile_id=profile.id,
                    owner_user_id=normalized_owner,
                    asset_id=normalized_asset_id,
                    segment_id=normalized_segment_id,
                    prompt_text=validated_text,
                    prompt_language=validated_language,
                    emotion_label=validated_emotion,
                    emotion_confidence=emotion_confidence,
                    is_primary=True,
                    reference_name=(reference_name or "").strip() or None,
                )
            )
            session.commit()
            return self._record(profile)

    def add_reference(
        self,
        profile_id: str,
        owner_user_id: str,
        *,
        asset_id: str,
        segment_id: str,
        prompt_text: str,
        prompt_language: str,
        emotion_label: str,
        emotion_confidence: float | None,
        reference_name: str | None = None,
    ) -> VoiceReferenceRecord:
        """Append one reviewed, non-primary emotion reference to an owned profile."""
        normalized_owner = _require_identifier(owner_user_id, "owner user id")
        validated_text = _validated_prompt_text(prompt_text)
        validated_language = _validated_prompt_language(prompt_language)
        validated_emotion = _validated_emotion_label(emotion_label)

        with self._session_factory() as session:
            profile = session.get(VoiceProfile, _require_identifier(profile_id, "profile id"))
            if profile is None or profile.owner_user_id != normalized_owner:
                raise KeyError(profile_id)
            existing_count = session.scalar(
                select(func.count(VoiceReference.id)).where(VoiceReference.profile_id == profile.id)
            )
            if existing_count >= 5:
                raise ValueError("音色参考音频最多保存 5 段")
            reference = VoiceReference(
                id=uuid4().hex,
                profile_id=profile.id,
                owner_user_id=normalized_owner,
                asset_id=_require_identifier(asset_id, "reference asset id"),
                segment_id=_require_identifier(segment_id, "reference segment id"),
                prompt_text=validated_text,
                prompt_language=validated_language,
                emotion_label=validated_emotion,
                emotion_confidence=emotion_confidence,
                is_primary=False,
                reference_name=(reference_name or "").strip() or None,
            )
            session.add(reference)
            session.commit()
            return self._reference_record(reference)

    def delete_reference(
        self,
        profile_id: str,
        reference_id: str,
        owner_user_id: str,
    ) -> None:
        """Delete an auxiliary reference from an owned profile. Primary reference cannot be deleted."""
        normalized_owner = _require_identifier(owner_user_id, "owner user id")
        with self._session_factory() as session:
            profile = session.get(VoiceProfile, _require_identifier(profile_id, "profile id"))
            if profile is None or profile.owner_user_id != normalized_owner:
                raise KeyError(profile_id)
            reference = session.get(VoiceReference, _require_identifier(reference_id, "reference id"))
            if reference is None or reference.profile_id != profile.id or reference.owner_user_id != normalized_owner:
                raise KeyError(reference_id)
            if reference.is_primary:
                raise ValueError("主参考音频不可删除")
            session.delete(reference)
            session.commit()

    def references(self, profile_id: str, owner_user_id: str) -> list[VoiceReferenceRecord]:
        normalized_owner = _require_identifier(owner_user_id, "owner user id")
        with self._session_factory() as session:
            profile = session.get(VoiceProfile, _require_identifier(profile_id, "profile id"))
            if profile is None or profile.owner_user_id != normalized_owner:
                raise KeyError(profile_id)
            rows = session.scalars(
                select(VoiceReference)
                .where(VoiceReference.profile_id == profile.id)
                .order_by(VoiceReference.is_primary.desc(), VoiceReference.created_at.asc())
            ).all()
            return [self._reference_record(reference) for reference in rows]

    def create(self, dataset_id: str, owner_user_id: str | None = None) -> VoiceProfileRecord:
        normalized_dataset_id = dataset_id.strip()
        if not normalized_dataset_id:
            raise ValueError("dataset id must not be empty")
        with self._session_factory() as session:
            profile = VoiceProfile(
                id=uuid4().hex,
                owner_user_id=owner_user_id,
                dataset_id=normalized_dataset_id,
                status="created",
                public_weight_dir=None,
            )
            session.add(profile)
            session.commit()
            return self._record(profile)

    def get(self, profile_id: str, owner_user_id: str | None = None) -> VoiceProfileRecord:
        with self._session_factory() as session:
            profile = session.get(VoiceProfile, profile_id)
            if profile is None or (owner_user_id is not None and profile.owner_user_id != owner_user_id):
                raise KeyError(profile_id)
            return self._record(profile)

    def mark_queued(self, profile_id: str) -> VoiceProfileRecord:
        return self._transition(profile_id, allowed={"created", "failed"}, target="queued")

    def mark_training(self, profile_id: str) -> VoiceProfileRecord:
        return self._transition(profile_id, allowed={"queued", "training"}, target="training")

    def publish(self, profile_id: str, public_weight_dir: str) -> VoiceProfileRecord:
        with self._session_factory() as session:
            profile = session.get(VoiceProfile, profile_id)
            if profile is None:
                raise KeyError(profile_id)
            if profile.status != "training":
                raise VoiceProfileStateError(f"{profile.status} -> ready")
            profile.status = "ready"
            profile.public_weight_dir = public_weight_dir
            session.commit()
            return self._record(profile)

    def fail(self, profile_id: str) -> VoiceProfileRecord:
        with self._session_factory() as session:
            profile = session.get(VoiceProfile, profile_id)
            if profile is None:
                raise KeyError(profile_id)
            if profile.status == "ready":
                raise VoiceProfileStateError("ready profiles cannot be failed by a stale worker")
            profile.status = "failed"
            profile.public_weight_dir = None
            session.commit()
            return self._record(profile)

    def _transition(self, profile_id: str, *, allowed: set[str], target: str) -> VoiceProfileRecord:
        with self._session_factory() as session:
            profile = session.get(VoiceProfile, profile_id)
            if profile is None:
                raise KeyError(profile_id)
            if profile.status not in allowed:
                raise VoiceProfileStateError(f"{profile.status} -> {target}")
            profile.status = target
            if target != "ready":
                profile.public_weight_dir = None
            session.commit()
            return self._record(profile)
