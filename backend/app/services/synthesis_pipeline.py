from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Protocol

from sqlalchemy import select

from backend.app.db.models import DatasetSegment
from backend.app.db.session import session_factory
from backend.app.schemas.synthesis import EmotionControl, SynthesisCreate
from backend.app.services.fingerprint import FingerprintService, SpectralFingerprint
from backend.app.services.gpt_sovits import GPTSoVITSAdapter, RawSynthesis, SynthesisSpec
from backend.app.services.sensitive_filter import SensitiveFilter
from backend.app.services.storage import LocalStorage
from backend.app.services.voice_base_registry import BaseModelUnavailable, ResolvedVoiceBase, VoiceBaseRegistry
from backend.app.services.voice_profiles import VoiceProfileRecord, VoiceProfileStore
from backend.app.services.watermark import WatermarkService
from backend.app.services.webui_bridge import WebUIInferenceBridge, WebUISynthesisSpec


WATERMARK_PROBABILITY_MINIMUM = 0.80
# Advisory value: outputs below this still publish with an explicit quality
# warning; only the mandatory safety gates can block publication.
OUTPUT_SIMILARITY_RECOMMENDED = 0.90
SIMILARITY_WARNING_CODE = "SPEAKER_SIMILARITY_BELOW_RECOMMENDED"
WEBUI_CUT_METHODS = {
    "none": "不切",
    "four_sentences": "凑四句一切",
    "fifty_chars": "凑50字一切",
    "zh_period": "按中文句号。切",
    "en_period": "按英文句号.切",
    "punctuation": "按标点符号切",
}


@dataclass(frozen=True)
class ReferenceCondition:
    reference_audio: Path
    prompt_text: str
    prompt_lang: str


@dataclass(frozen=True)
class ZeroShotSynthesisReference(ReferenceCondition):
    """Resolved zero-shot condition: persisted reference plus verified base weights."""

    emotion_label: str
    base: ResolvedVoiceBase
    primary_audio_path: Path
    auxiliary_audios: tuple[Path, ...] = ()


@dataclass(frozen=True)
class SynthesisRunResult:
    status: str
    error_code: str | None
    error_message: str | None
    public_audio_path: Path | None
    watermark_probability: float | None
    reference_language: str | None
    fingerprint: SpectralFingerprint | None
    speaker_similarity: float | None = None
    quality_warning_codes: tuple[str, ...] = ()
    candidate_count: int = 1
    selected_candidate_index: int = 0


class ReferenceResolver(Protocol):
    def resolve(self, profile: VoiceProfileRecord, emotion: EmotionControl) -> ReferenceCondition: ...


class OutputSimilarityGate(Protocol):
    """CAM++-style speaker similarity between one output and its reference."""

    def similarity(self, output_wav: Path, reference_wav: Path) -> float: ...


class LocalSynthesizer(Protocol):
    def synthesize(
        self,
        request: SynthesisCreate,
        profile: VoiceProfileRecord,
        reference: ReferenceCondition,
        output_wav: Path,
    ) -> Path: ...


class GPTSoVITSLocalSynthesizer:
    """Adapt frozen base weights or published local weights to the guarded pipeline."""

    def __init__(
        self,
        adapter: GPTSoVITSAdapter | None = None,
        *,
        webui_bridge: WebUIInferenceBridge | None = None,
    ) -> None:
        self.adapter = adapter
        self.webui_bridge = webui_bridge

    def synthesize(
        self,
        request: SynthesisCreate,
        profile: VoiceProfileRecord,
        reference: ReferenceCondition,
        output_wav: Path,
        *,
        seed: int = 20260916,
    ) -> Path:
        if isinstance(reference, ZeroShotSynthesisReference):
            gpt_weight = reference.base.gpt_weight
            sovits_weight = reference.base.sovits_weight
            if self.webui_bridge is not None:
                spec = WebUISynthesisSpec(
                    text=request.text,
                    text_lang=request.text_lang,
                    prompt_text=reference.prompt_text,
                    prompt_lang=reference.prompt_lang,
                    reference_audio=reference.reference_audio,
                    gpt_weight=gpt_weight,
                    sovits_weight=sovits_weight,
                    output_wav=output_wav,
                    auxiliary_audios=reference.auxiliary_audios,
                    top_k=request.local_options.top_k,
                    top_p=request.local_options.top_p,
                    temperature=request.local_options.temperature,
                    how_to_cut=WEBUI_CUT_METHODS[request.local_options.cut_method],
                    speed=request.local_options.speed,
                    pause_second=request.local_options.pause_seconds,
                    seed=seed,
                )
                return self.webui_bridge.synthesize(spec)
        else:
            if profile.public_weight_dir is None:
                raise SynthesisPipelineError("voice profile has no published local weights")
            public_dir = Path(profile.public_weight_dir).resolve()
            weights_dir = (public_dir / "weights").resolve()
            try:
                weights_dir.relative_to(public_dir)
            except ValueError as exc:
                raise SynthesisPipelineError("published weight directory escapes profile storage") from exc
            gpt_weight = self._latest_weight(weights_dir, "*.ckpt", "GPT")
            sovits_weight = self._latest_weight(weights_dir, "*_e*_s*.pth", "SoVITS")

        if self.adapter is None:
            raise SynthesisPipelineError("no local adapter configured for synthesis")

        result = self.adapter.synthesize(
            SynthesisSpec(
                text=request.text,
                text_lang=request.text_lang,
                prompt_text=reference.prompt_text,
                prompt_lang=reference.prompt_lang,
                reference_audio=reference.reference_audio,
                gpt_weight=gpt_weight,
                sovits_weight=sovits_weight,
                output_wav=output_wav,
            )
        )
        if not isinstance(result, RawSynthesis) or result.output_wav.resolve() != output_wav.resolve():
            raise SynthesisPipelineError("GPT-SoVITS adapter returned an invalid local WAV contract")
        return result.output_wav

    @staticmethod
    def _latest_weight(weights_dir: Path, pattern: str, label: str) -> Path:
        if not weights_dir.is_dir():
            raise SynthesisPipelineError("published profile has no local weights directory")
        candidates = sorted(
            (path.resolve() for path in weights_dir.glob(pattern) if path.is_file()),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )
        if not candidates:
            raise SynthesisPipelineError(f"published profile has no {label} weight")
        return candidates[-1]


class HybridSynthesizer:
    """Route normal requests locally and emotion requests to the configured provider."""

    def __init__(self, local: LocalSynthesizer, cloud: LocalSynthesizer | None) -> None:
        self.local = local
        self.cloud = cloud

    def synthesize(
        self,
        request: SynthesisCreate,
        profile: VoiceProfileRecord,
        reference: ReferenceCondition,
        output_wav: Path,
        *,
        seed: int = 20260916,
    ) -> Path:
        if request.synthesis_mode == "local":
            return self.local.synthesize(request, profile, reference, output_wav, seed=seed)
        if self.cloud is None:
            raise CloudTTSUnavailable("cloud emotion synthesis is not configured")
        return self.cloud.synthesize(request, profile, reference, output_wav, seed=seed)


class ZeroShotReferenceResolver:
    """Resolve the synthesis condition only from persisted, owner-scoped data.

    Reference audio is derived from the profile's own VoiceReference rows and
    the reviewed DatasetSegment they point to; request payloads carry no path.
    Base S1/S2 weights come from the checksum-verified registry on every call.
    """

    def __init__(
        self,
        *,
        profiles: VoiceProfileStore,
        storage: LocalStorage,
        registry: VoiceBaseRegistry,
    ) -> None:
        self.profiles = profiles
        self.storage = storage
        self.registry = registry

    def resolve(
        self, profile: VoiceProfileRecord, emotion: EmotionControl
    ) -> ZeroShotSynthesisReference:
        if profile.mode != "zero_shot":
            raise VoiceProfileUnavailable("voice profile is not a zero-shot profile")
        try:
            base = self.registry.resolve(profile.base_model_id)
        except BaseModelUnavailable as exc:
            raise BaseModelUnavailableError(str(exc)) from exc

        owner_user_id = profile.owner_user_id
        if not owner_user_id:
            raise VoiceProfileUnavailable("zero-shot profile has no owner")

        references = self.profiles.references(profile.id, owner_user_id)
        if not references:
            raise VoiceProfileUnavailable("voice profile has no references")

        # Query quality metrics for all segments in this dataset to enable ranking
        segment_qualities = self._query_segment_qualities(profile.dataset_id, owner_user_id)

        requested_label = emotion.label if emotion.mode == "manual" else None
        if requested_label is not None:
            matching = [row for row in references if row.emotion_label == requested_label.value]
            if not matching:
                raise EmotionReferenceUnavailable(
                    "所选情绪没有可用参考片段，请先上传并审核对应情绪的参考音频"
                )
            # Sort matching references by quality: higher SNR, lower clipping, earlier created_at
            sorted_matching = sorted(
                matching,
                key=lambda r: self._quality_sort_key(r, segment_qualities),
                reverse=True,
            )
            selected = sorted_matching[0]
            auxiliary_rows = sorted_matching[1:4]
        else:
            primary_row = next((row for row in references if row.is_primary), None)
            if primary_row is None:
                raise VoiceProfileUnavailable("voice profile has no primary reference")
            selected = primary_row
            # Auxiliary references: sort remaining references by quality and take up to 3
            remaining = [row for row in references if row.id != primary_row.id]
            sorted_remaining = sorted(
                remaining,
                key=lambda r: self._quality_sort_key(r, segment_qualities),
                reverse=True,
            )
            auxiliary_rows = sorted_remaining[:3]

        audio_path = self._segment_audio_path(
            profile.dataset_id,
            selected.segment_id,
            owner_user_id,
        )
        primary = next((row for row in references if row.is_primary), selected)
        primary_path = (
            audio_path
            if primary.id == selected.id
            else self._segment_audio_path(
                profile.dataset_id,
                primary.segment_id,
                owner_user_id,
            )
        )
        auxiliary_paths = tuple(
            self._segment_audio_path(
                profile.dataset_id,
                row.segment_id,
                owner_user_id,
            )
            for row in auxiliary_rows
        )
        return ZeroShotSynthesisReference(
            reference_audio=audio_path,
            prompt_text=selected.prompt_text,
            prompt_lang=selected.prompt_language,
            emotion_label=selected.emotion_label,
            base=base,
            primary_audio_path=primary_path,
            auxiliary_audios=auxiliary_paths,
        )

    def _query_segment_qualities(
        self, dataset_id: str, owner_user_id: str
    ) -> dict[str, tuple[float, float, float]]:
        with session_factory(self.profiles.engine)() as session:
            rows = session.execute(
                select(
                    DatasetSegment.segment_id,
                    DatasetSegment.snr_db,
                    DatasetSegment.clipping_ratio,
                    DatasetSegment.created_at,
                ).where(
                    DatasetSegment.dataset_id == dataset_id,
                    DatasetSegment.owner_user_id == owner_user_id,
                )
            ).all()
            result = {}
            for seg_id, snr, clip, created in rows:
                snr_val = float(snr) if snr is not None else 0.0
                clip_val = float(clip) if clip is not None else 0.0
                ts = created.timestamp() if created else 0.0
                result[seg_id] = (snr_val, clip_val, ts)
            return result

    @staticmethod
    def _quality_sort_key(
        ref, segment_qualities: dict[str, tuple[float, float, float]]
    ) -> tuple[float, float, float]:
        snr, clip, ts = segment_qualities.get(ref.segment_id, (0.0, 0.0, 0.0))
        # Higher score is better: snr ascending, clipping descending (so -clip), ts ascending for tie-break (earlier first so -ts)
        return (snr, -clip, -ts)

    def _segment_audio_path(
        self,
        dataset_id: str,
        segment_id: str,
        owner_user_id: str,
    ) -> Path:
        with session_factory(self.profiles.engine)() as session:
            segment = session.scalar(
                select(DatasetSegment).where(
                    DatasetSegment.dataset_id == dataset_id,
                    DatasetSegment.segment_id == segment_id,
                    DatasetSegment.owner_user_id == owner_user_id,
                )
            )
        if segment is None:
            raise VoiceProfileUnavailable("参考音频分段不存在")
        try:
            return self.storage.resolve(
                "datasets", Path(segment.relative_path).relative_to("datasets")
            )
        except ValueError as exc:
            raise VoiceProfileUnavailable("参考音频路径越出本地存储边界") from exc


class SynthesisPipelineError(RuntimeError):
    code = "SYNTHESIS_PIPELINE_FAILED"


class SensitiveTextBlocked(SynthesisPipelineError):
    code = "SENSITIVE_TEXT_BLOCKED"


class VoiceProfileUnavailable(SynthesisPipelineError):
    code = "VOICE_PROFILE_NOT_READY"


class WatermarkVerificationFailed(SynthesisPipelineError):
    code = "WATERMARK_VERIFICATION_FAILED"


class BaseModelUnavailableError(SynthesisPipelineError):
    code = "BASE_MODEL_UNAVAILABLE"


class EmotionReferenceUnavailable(SynthesisPipelineError):
    code = "EMOTION_REFERENCE_UNAVAILABLE"


class CloudTTSUnavailable(SynthesisPipelineError):
    code = "CLOUD_TTS_NOT_CONFIGURED"


class SynthesisPipeline:
    """Run the safety-critical, local-only synthesis sequence before publication."""

    def __init__(
        self,
        *,
        profiles: VoiceProfileStore,
        sensitive_filter: SensitiveFilter,
        storage: LocalStorage,
        synthesizer: LocalSynthesizer,
        reference_resolver: ReferenceResolver,
        watermark: WatermarkService,
        fingerprint: FingerprintService,
        similarity_gate: OutputSimilarityGate,
        max_candidates: int = 3,
    ) -> None:
        self.profiles = profiles
        self.sensitive_filter = sensitive_filter
        self.storage = storage
        self.synthesizer = synthesizer
        self.reference_resolver = reference_resolver
        self.watermark = watermark
        self.fingerprint = fingerprint
        self.similarity_gate = similarity_gate
        self.max_candidates = max_candidates

    def run(
        self,
        job_id: str,
        request: SynthesisCreate,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> SynthesisRunResult:
        emit = progress or (lambda message: None)
        temporary_paths: list[Path] = []
        try:
            self._require_safe_request(request)
            profile = self._ready_profile(request.voice_profile_id)
            resolver_emotion = request.emotion
            if request.synthesis_mode == "emotion_api":
                resolver_emotion = EmotionControl(mode="auto", strength=request.emotion.strength)
            reference = self.reference_resolver.resolve(profile, resolver_emotion)
            self._require_local_reference(reference.reference_audio)

            work_dir = self.storage.resolve("temp", Path("synthesis") / profile.id / self._safe_identifier(job_id))
            work_dir.mkdir(parents=True, exist_ok=True)
            temporary_paths.append(work_dir)

            payload = self._payload(job_id)
            primary_audio = getattr(reference, "primary_audio_path", None) or reference.reference_audio

            # Generate and evaluate candidates sequentially
            candidates: list[dict[str, Any]] = []
            base_seed = int.from_bytes(sha256(job_id.encode("utf-8")).digest()[:4], "big")

            candidate_count = self.max_candidates if request.synthesis_mode == "local" else 1
            for c_idx in range(candidate_count):
                emit(f"正在生成候选 {c_idx + 1}/{candidate_count}...")
                raw_wav = work_dir / f"candidate_{c_idx}_raw.wav"
                watermarked_wav = work_dir / f"candidate_{c_idx}_wm.wav"
                temporary_paths.extend([raw_wav, watermarked_wav])

                cand_seed = (base_seed + c_idx) % 1_000_000
                try:
                    produced = Path(
                        self.synthesizer.synthesize(request, profile, reference, raw_wav, seed=cand_seed)
                    ).resolve()
                except TypeError:
                    produced = Path(
                        self.synthesizer.synthesize(request, profile, reference, raw_wav)
                    ).resolve()

                if produced != raw_wav or not raw_wav.is_file():
                    raise SynthesisPipelineError("synthesizer did not produce the guarded local temporary WAV")

                emit(f"正在验证候选 {c_idx + 1}/{candidate_count}...")
                similarity = self.similarity_gate.similarity(raw_wav, Path(primary_audio))

                # Embed watermark and detect
                embedded = self.watermark.embed(raw_wav, watermarked_wav, payload)
                if embedded.output_wav != watermarked_wav:
                    raise SynthesisPipelineError("watermark service returned an unexpected output path")

                detection = self.watermark.detect(watermarked_wav)
                if detection.probability >= WATERMARK_PROBABILITY_MINIMUM and detection.payload == payload:
                    candidates.append({
                        "index": c_idx,
                        "raw_wav": raw_wav,
                        "watermarked_wav": watermarked_wav,
                        "similarity": similarity,
                        "watermark_probability": detection.probability,
                    })

            if not candidates:
                raise WatermarkVerificationFailed("watermark verification did not meet the local gate")

            # Winner selection: highest similarity first; break tie by earliest candidate index
            emit("正在择优选择最佳结果...")
            winner = max(candidates, key=lambda c: (c["similarity"], -c["index"]))
            winning_similarity = winner["similarity"]
            winning_watermarked_wav = winner["watermarked_wav"]
            winning_prob = winner["watermark_probability"]

            quality_warning_codes: tuple[str, ...] = ()
            if winning_similarity < OUTPUT_SIMILARITY_RECOMMENDED:
                quality_warning_codes = (SIMILARITY_WARNING_CODE,)

            fingerprint = self.fingerprint.analyze(winning_watermarked_wav)
            public_audio = self.storage.atomic_publish(
                winning_watermarked_wav,
                "outputs",
                Path(profile.id) / f"{self._safe_identifier(job_id)}.wav",
            )
            return SynthesisRunResult(
                status="succeeded",
                error_code=None,
                error_message=None,
                public_audio_path=public_audio,
                watermark_probability=winning_prob,
                reference_language=reference.prompt_lang,
                fingerprint=fingerprint,
                speaker_similarity=winning_similarity,
                quality_warning_codes=quality_warning_codes,
                candidate_count=candidate_count,
                selected_candidate_index=winner["index"],
            )
        except SynthesisPipelineError as exc:
            return self._failed(exc.code, str(exc))
        except Exception:
            return self._failed("SYNTHESIS_PIPELINE_FAILED", "local synthesis pipeline failed")
        finally:
            self._remove_temporary_paths(temporary_paths)

    @staticmethod
    def _failed(code: str, message: str) -> SynthesisRunResult:
        return SynthesisRunResult(
            status="failed",
            error_code=code,
            error_message=message,
            public_audio_path=None,
            watermark_probability=None,
            reference_language=None,
            fingerprint=None,
            speaker_similarity=None,
        )

    def _require_safe_request(self, request: SynthesisCreate) -> None:
        if request.consent_confirmed is not True:
            raise SynthesisPipelineError("literal synthesis consent is required")
        if self.sensitive_filter.check(request.text).blocked:
            raise SensitiveTextBlocked("synthesis text is blocked")

    def _ready_profile(self, profile_id: str) -> VoiceProfileRecord:
        try:
            profile = self.profiles.get(profile_id)
        except KeyError as exc:
            raise VoiceProfileUnavailable("voice profile does not exist") from exc
        if profile.status != "ready":
            raise VoiceProfileUnavailable("voice profile is not ready")
        if profile.mode == "zero_shot":
            # Zero-shot profiles carry no personal weights; the frozen base
            # weights are re-verified by the registry in the reference resolver.
            if not profile.base_model_id:
                raise VoiceProfileUnavailable("zero-shot profile has no base model")
            return profile
        if profile.public_weight_dir is None:
            raise VoiceProfileUnavailable("voice profile is not ready")
        weights_dir = Path(profile.public_weight_dir).resolve()
        self._require_under_storage(weights_dir)
        if not weights_dir.is_dir():
            raise VoiceProfileUnavailable("voice profile has no local published weights")
        return profile

    def _require_local_reference(self, reference_audio: Path) -> None:
        reference_path = Path(reference_audio).resolve()
        self._require_under_storage(reference_path)
        if not reference_path.is_file():
            raise SynthesisPipelineError("selected local reference audio is unavailable")

    def _temporary_paths(self, profile_id: str, job_id: str) -> tuple[Path, Path, Path]:
        safe_profile_id = self._safe_identifier(profile_id)
        safe_job_id = self._safe_identifier(job_id)
        work_dir = self.storage.resolve("temp", Path("syntheses") / safe_profile_id / safe_job_id)
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir, work_dir / "raw.wav", work_dir / "watermarked.wav"

    def _require_under_storage(self, path: Path) -> None:
        try:
            path.resolve().relative_to(self.storage.root)
        except ValueError as exc:
            raise SynthesisPipelineError("local artifact path escapes the configured data root") from exc

    @staticmethod
    def _safe_identifier(value: str) -> str:
        if not value or Path(value).name != value or value in {".", ".."}:
            raise SynthesisPipelineError("invalid local synthesis identifier")
        return value

    @staticmethod
    def _payload(job_id: str) -> int:
        return int.from_bytes(sha256(job_id.encode("utf-8")).digest()[:2], "big")

    @staticmethod
    def _remove_temporary_paths(paths: list[Path] | tuple[Path, ...]) -> None:
        work_dirs = []
        for path in paths:
            if path.is_file():
                with suppress(FileNotFoundError):
                    path.unlink()
            elif path.is_dir():
                work_dirs.append(path)
        for d in reversed(work_dirs):
            with suppress(OSError):
                d.rmdir()
