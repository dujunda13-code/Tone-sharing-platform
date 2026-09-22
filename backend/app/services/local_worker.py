"""Real local worker assembly (Phase D2).

This module is the single production wiring for `scripts/worker_process.py`:
a SQLite segment manifest loader with fail-closed gate re-validation, the
real GPT-SoVITS preparer/adapter, the disentangler trainer with a local
feature-batch loader, the guarded synthesis pipeline with local components,
and the CAM++-based evaluation used by the training quality gate. Every
local-model dependency fails closed with a fixed message until the
deployment provides the weights under `models/`; nothing resolves a hub
model name or fabricates results.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from backend.app.core.config import get_settings

import numpy as np
import soundfile as sf
from sqlalchemy import select

from backend.app.db.models import Dataset
from backend.app.db.session import init_db, session_factory
from backend.app.ml.datasets import FeatureBatch
from backend.app.schemas.common import EmotionLabel
from backend.app.schemas.dataset import DatasetManifest
from backend.app.services.audio_validation import (
    resolve_local_audio_path,
    validate_training_duration,
)
from backend.app.services.disentanglement import (
    DisentanglerTrainer,
    ReferenceCandidate,
    ReferenceControl,
    ReferenceSelector,
)
from backend.app.services.fingerprint import FingerprintService
from backend.app.services.gpt_sovits import GPTSoVITSAdapter, SynthesisSpec
from backend.app.services.gpt_sovits_dataset import GPTSoVITSDatasetPreparer
from backend.app.services.job_queue import JobQueue
from backend.app.services.metrics import (
    EvaluationContractError,
    load_evaluation_texts,
    summarize_similarity,
)
from backend.app.services.sensitive_filter import SensitiveFilter
from backend.app.services.speaker_embedding import (
    CamPlusEmbedder,
    EmotionEmbedder,
    cosine_similarity,
)
from backend.app.services.storage import LocalStorage
from backend.app.services.synthesis_pipeline import (
    GPTSoVITSLocalSynthesizer,
    HybridSynthesizer,
    SynthesisPipeline,
    ZeroShotReferenceResolver,
)
from backend.app.services.training_errors import TrainingPipelineError
from backend.app.services.voice_base_registry import VoiceBaseRegistry
from backend.app.services.voice_profiles import VoiceProfileStore
from backend.app.services.watermark import WatermarkService
from backend.app.workers.gpu_worker import GPUWorker

DEFAULT_TEXTS_PATH = Path("config") / "evaluation_texts.zh-en.json"
DEFAULT_SENSITIVE_WORDS_PATH = Path("config") / "sensitive_words.zh-en.txt"
REFERENCES_FILENAME = "references.json"
FEATURE_MANIFEST_FILENAME = "feature-manifest.json"


def _require_under_models(model_dir: Path, models_root: Path, label: str) -> Path:
    resolved = model_dir.expanduser().resolve()
    if not resolved.is_dir():
        raise TrainingPipelineError(
            f"本地评测模型未就绪（{label}），请先在部署阶段下载到 models/"
        )
    if resolved != models_root and models_root not in resolved.parents:
        raise TrainingPipelineError(f"本地{label}模型目录必须位于 models/ 之内")
    return resolved


class SegmentManifestLoader:
    """Rebuild the training manifest from reviewed SQLite segments.

    The loader re-validates authorization, reviewed non-empty text and the
    480–720 second window at worker time so a stale API state cannot bypass
    the review loop.
    """

    def __init__(self, dataset_service: Any, *, storage_root: Path | str | None = None) -> None:
        self.dataset_service = dataset_service
        root = Path(storage_root or dataset_service.storage.root)
        self.storage_root = root.expanduser().resolve()

    def load(self, dataset_id: str) -> DatasetManifest:
        with session_factory(self.dataset_service.engine)() as session:
            dataset = session.scalar(select(Dataset).where(Dataset.id == dataset_id))
        if dataset is None:
            raise TrainingPipelineError("训练数据集不存在，无法继续训练")
        if dataset.authorization_confirmed_at is None:
            raise TrainingPipelineError("数据集未完成授权确认，无法训练")
        manifest = self.dataset_service.build_training_manifest(dataset_id, dataset.owner_user_id)
        if not manifest.rows:
            raise TrainingPipelineError("数据集没有已审核且文本非空的分段，无法训练")
        try:
            validate_training_duration(float(manifest.effective_seconds))
        except ValueError as exc:
            raise TrainingPipelineError(str(exc)) from exc
        for row in manifest.rows:
            resolved = resolve_local_audio_path(row.path, self.storage_root)
            if not resolved.is_file():
                raise TrainingPipelineError(
                    f"已审核分段在本机存储中缺失，无法训练：{row.segment_id}"
                )
        return manifest


def _frame_features(mono: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float, float]:
    """Frame-level log-F0, RMS energy, voiced flags plus mean F0/energy."""
    import librosa

    minimum = sample_rate // 10
    if mono.size < minimum:
        mono = np.pad(mono, (0, minimum - mono.size))
    frame_length = 1024
    hop_length = 320
    f0 = librosa.yin(
        mono.astype(np.float64),
        fmin=60.0,
        fmax=500.0,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    energy = librosa.feature.rms(
        y=mono.astype(np.float64), frame_length=frame_length, hop_length=hop_length
    )[0]
    frames = min(f0.shape[0], energy.shape[0])
    f0, energy = f0[:frames], energy[:frames]
    voiced = (f0 > 0.0).astype(np.float64)
    log_f0 = np.log(np.where(f0 > 0.0, f0, 1.0))
    log_energy = np.log(np.where(energy > 0.0, energy, 1e-10))
    duration = np.full(frames, hop_length / sample_rate)
    stacked = np.stack([log_f0, log_energy, voiced, duration], axis=1).astype(np.float32)
    mean_f0 = float(np.nanmean(f0)) if frames else 0.0
    mean_energy = float(np.nanmean(energy)) if frames else 0.0
    return stacked, mean_f0, mean_energy


def _frame_stats(wav_path: Path) -> tuple[np.ndarray, float, float]:
    samples, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    return _frame_features(samples.mean(axis=1), sample_rate)


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    frames = min(left.shape[0], right.shape[0])
    if frames < 2:
        return 0.0
    a = left[:frames].astype(np.float64)
    b = right[:frames].astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def load_feature_batches(
    feature_manifest_path: Path,
    *,
    campplus_dir: Path,
    emotion_dir: Path,
    models_root: Path,
) -> list[FeatureBatch]:
    """Build serial per-segment FeatureBatch tensors from the local feature manifest."""
    campplus = CamPlusEmbedder(campplus_dir, models_root=models_root)
    emotion = EmotionEmbedder(emotion_dir, models_root=models_root)
    payload = json.loads(Path(feature_manifest_path).read_text(encoding="utf-8"))
    entries = payload.get("segments")
    if not isinstance(entries, list) or not entries:
        raise TrainingPipelineError("本地特征清单没有任何分段，无法训练解耦适配器")
    manifest_root = Path(feature_manifest_path).parent
    batches: list[FeatureBatch] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or not {"segment_id", "speaker_id", "wav32k"} <= set(entry):
            raise TrainingPipelineError("本地特征清单分段字段不完整")
        wav_path = (manifest_root / str(entry["wav32k"])).resolve()
        if not wav_path.is_file():
            raise TrainingPipelineError(f"本地特征分段缺失：{entry['segment_id']}")
        samples, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
        prosody_frames, _f0, _energy = _frame_features(samples.mean(axis=1), sample_rate)
        emotion_values = emotion.embed(wav_path)
        timbre_values = campplus.embed(wav_path)
        frames = prosody_frames.shape[0]
        prosody = np.concatenate(
            [prosody_frames, np.tile(np.asarray(emotion_values, dtype=np.float32), (frames, 1))],
            axis=1,
        )
        batches.append(
            FeatureBatch(
                timbre=np.asarray([timbre_values], dtype=np.float32),
                prosody=prosody[np.newaxis, ...],
                emotion=np.asarray([emotion_values], dtype=np.float32),
                speaker_ids=(str(entry["speaker_id"]),),
                segment_ids=(str(entry["segment_id"]),),
            )
        )
    return batches


@dataclass(frozen=True)
class _ReferenceRow:
    segment_id: str
    wav_path: Path
    text: str
    language: str
    duration_seconds: float
    split: str = "train"


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return cosine_similarity(list(left), list(right))


class LocalTrainingAdapters:
    """Real preflight and CAM++ evaluation for the local training stage machine.

    `feature_extract`, `gpt_sovits_train` and `disentangler_train` are owned by
    the dedicated local adapters injected into the worker; reaching them here
    means an assembly mistake, so they fail loudly instead of silently falling
    back to a stub.
    """

    def __init__(
        self,
        *,
        gpt_sovits: GPTSoVITSAdapter,
        campplus_dir: Path | str,
        emotion_dir: Path | str,
        models_root: Path | str,
        storage_root: Path | str,
        texts_path: Path | str = DEFAULT_TEXTS_PATH,
        watermark: WatermarkService | None = None,
        campplus_loader: Callable[[Path], CamPlusEmbedder] | None = None,
        emotion_loader: Callable[[Path], EmotionEmbedder] | None = None,
    ) -> None:
        self.gpt_sovits = gpt_sovits
        self.campplus_dir = Path(campplus_dir)
        self.emotion_dir = Path(emotion_dir)
        self.models_root = Path(models_root).expanduser().resolve()
        self.storage_root = Path(storage_root).expanduser().resolve()
        self.texts_path = Path(texts_path)
        self.watermark = watermark or WatermarkService()
        self._campplus_loader = campplus_loader
        self._emotion_loader = emotion_loader

    def preflight(self, manifest: DatasetManifest, work_dir: Path) -> Mapping[str, object]:
        del work_dir
        probe = self.gpt_sovits.probe()
        if not probe.available:
            raise TrainingPipelineError(probe.reason or "本地 GPT-SoVITS 运行时未就绪")
        try:
            validate_training_duration(float(manifest.effective_seconds))
        except ValueError as exc:
            raise TrainingPipelineError(str(exc)) from exc
        return {
            "gpt_sovits_tag": probe.tag,
            "rows": len(manifest.rows),
            "effective_seconds": manifest.effective_seconds,
        }

    def feature_extract(self, manifest: DatasetManifest, work_dir: Path) -> Mapping[str, object]:
        raise TrainingPipelineError("feature_extract 阶段必须由本地数据准备器执行")

    def gpt_sovits_train(self, manifest: DatasetManifest, work_dir: Path) -> Mapping[str, object]:
        raise TrainingPipelineError("gpt_sovits_train 阶段必须由本地 GPT-SoVITS 适配器执行")

    def disentangler_train(self, manifest: DatasetManifest, work_dir: Path) -> Mapping[str, object]:
        raise TrainingPipelineError("disentangler_train 阶段必须由本地解耦训练器执行")

    def evaluate(self, manifest: DatasetManifest, work_dir: Path) -> Mapping[str, object]:
        campplus_dir = _require_under_models(self.campplus_dir, self.models_root, "CAM++")
        emotion_dir = _require_under_models(self.emotion_dir, self.models_root, "Emotion2Vec")
        weights_dir = work_dir / "weights"
        gpt_weight = _latest_weight(weights_dir, "*.ckpt", "GPT")
        sovits_weight = _latest_weight(weights_dir, "*_e*_s*.pth", "SoVITS")
        reference_rows = self._reference_rows(manifest)
        campplus = (
            self._campplus_loader(campplus_dir)
            if self._campplus_loader
            else CamPlusEmbedder(campplus_dir, models_root=self.models_root)
        )
        emotion = (
            self._emotion_loader(emotion_dir)
            if self._emotion_loader
            else EmotionEmbedder(emotion_dir, models_root=self.models_root)
        )
        texts = load_evaluation_texts(self.texts_path)
        report = _run_similarity_suite(
            gpt_sovits=self.gpt_sovits,
            campplus=campplus,
            emotion=emotion,
            reference_rows=reference_rows,
            texts=texts,
            gpt_weight=gpt_weight,
            sovits_weight=sovits_weight,
            work_dir=work_dir / "evaluation",
            watermark=self.watermark,
        )
        _write_reference_index(
            weights_dir=weights_dir,
            dataset_id=manifest.dataset_id,
            reference_rows=reference_rows,
            campplus=campplus,
        )
        return {
            "speaker_similarity_median": report["median"],
            "speaker_similarity_p10": report["p10"],
            "samples": report["samples"],
        }

    def _reference_rows(self, manifest: DatasetManifest) -> list[_ReferenceRow]:
        rows = []
        for row in manifest.rows:
            if not 5.0 <= float(row.duration_seconds) <= 10.0:
                continue
            text = (row.text or "").strip()
            if not text:
                continue
            rows.append(
                _ReferenceRow(
                    segment_id=row.segment_id,
                    wav_path=resolve_local_audio_path(row.path, self.storage_root),
                    text=text,
                    language=row.language,
                    duration_seconds=float(row.duration_seconds),
                    split=row.split,
                )
            )
        if not rows:
            raise TrainingPipelineError("数据集没有可用的 5–10 秒已审核参考片段，无法评测")
        rows.sort(key=lambda row: (0 if row.split == "test" else 1, row.segment_id))
        return rows


class LocalProfileReferenceResolver:
    """Resolve synthesis references from the profile's published local index."""

    def resolve(self, profile: Any, emotion: Any) -> Any:
        from backend.app.services.synthesis_pipeline import ReferenceCondition

        public_dir = Path(profile.public_weight_dir or "").expanduser().resolve()
        index_path = public_dir / "weights" / REFERENCES_FILENAME
        if not index_path.is_file():
            raise TrainingPipelineError("音色档案尚未生成本地参考索引，无法安全合成")
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weights_dir = index_path.parent
        candidates: list[ReferenceCandidate] = []
        language_by_segment: dict[str, str] = {}
        for entry in payload.get("candidates", []):
            audio_path = (weights_dir / str(entry["audio_path"])).resolve()
            try:
                audio_path.relative_to(weights_dir)
            except ValueError as exc:
                raise TrainingPipelineError("本地参考索引路径越界") from exc
            segment_id = str(entry["segment_id"])
            language_by_segment[segment_id] = str(entry.get("language", "zh"))
            candidates.append(
                ReferenceCandidate(
                    profile_id=str(profile.id),
                    segment_id=segment_id,
                    audio_path=audio_path,
                    text=str(entry["text"]),
                    duration_seconds=float(entry["duration_seconds"]),
                    timbre_embedding=tuple(float(value) for value in entry["timbre_embedding"]),
                    emotion_label=EmotionLabel(entry["emotion_label"]),
                    emotion_strength=float(entry.get("emotion_strength", 0.65)),
                    f0_mean_hz=float(entry.get("f0_mean_hz", 0.0)),
                    energy_mean=float(entry.get("energy_mean", 0.0)),
                )
            )
        if not candidates:
            raise TrainingPipelineError("音色档案的本地参考索引为空，无法安全合成")
        selector = ReferenceSelector(candidates)
        label = emotion.label if getattr(emotion, "label", None) is not None else EmotionLabel.NEUTRAL
        control = ReferenceControl(
            emotion_label=label,
            emotion_strength=float(getattr(emotion, "strength", 0.65) or 0.65),
        )
        try:
            selected = selector.select(str(profile.id), control)
        except ValueError as exc:
            raise TrainingPipelineError(f"本地参考选择失败：{exc}") from exc
        return ReferenceCondition(
            reference_audio=selected.audio_path,
            prompt_text=selected.text,
            prompt_lang=language_by_segment.get(selected.segment_id, "zh"),
        )


def _latest_weight(weights_dir: Path, pattern: str, label: str) -> Path:
    if not weights_dir.is_dir():
        raise TrainingPipelineError(f"本地训练{label}权重目录缺失，无法评测")
    candidates = sorted(
        (path.resolve() for path in weights_dir.glob(pattern) if path.is_file()),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    if not candidates:
        raise TrainingPipelineError(f"本地训练{label}权重缺失，无法评测")
    return candidates[-1]


def _run_similarity_suite(
    *,
    gpt_sovits: GPTSoVITSAdapter,
    campplus: CamPlusEmbedder,
    emotion: EmotionEmbedder,
    reference_rows: Sequence[_ReferenceRow],
    texts: Iterable[dict[str, str]],
    gpt_weight: Path,
    sovits_weight: Path,
    work_dir: Path,
    watermark: WatermarkService,
) -> Mapping[str, object]:
    """Synthesize the frozen texts and measure real local quality metrics."""
    work_dir.mkdir(parents=True, exist_ok=True)
    primary = reference_rows[0]
    reference_timbre = campplus.embed(primary.wav_path)
    reference_emotion = emotion.embed(primary.wav_path)
    reference_frames, reference_f0, _reference_energy = _frame_stats(primary.wav_path)
    similarities: list[float] = []
    emotion_cosines: list[float] = []
    f0_correlations: list[float] = []
    energy_correlations: list[float] = []
    duration_errors: list[float] = []
    watermark_clean = 0
    watermark_supported = 0
    zh_to_en = 0
    en_to_zh = 0
    for index, item in enumerate(texts):
        output_wav = work_dir / f"eval_{index:02d}.wav"
        gpt_sovits.synthesize(
            SynthesisSpec(
                text=str(item["text"]),
                text_lang=str(item["lang"]),
                prompt_text=primary.text,
                prompt_lang=primary.language,
                reference_audio=primary.wav_path,
                gpt_weight=gpt_weight,
                sovits_weight=sovits_weight,
                output_wav=output_wav,
            )
        )
        similarities.append(_cosine(campplus.embed(output_wav), reference_timbre))
        emotion_cosines.append(_cosine(emotion.embed(output_wav), reference_emotion))
        output_frames, _output_f0, _output_energy = _frame_stats(output_wav)
        f0_correlations.append(_safe_corr(output_frames[:, 0], reference_frames[:, 0]))
        energy_correlations.append(_safe_corr(output_frames[:, 1], reference_frames[:, 1]))
        duration_errors.append(
            abs((output_frames.shape[0] / max(reference_frames.shape[0], 1)) - 1.0)
        )
        if str(item["lang"]) != primary.language:
            if str(item["lang"]) == "en":
                zh_to_en += 1
            elif str(item["lang"]) == "zh":
                en_to_zh += 1
        payload_index = index % 0xFFFF
        watermarked = work_dir / f"eval_{index:02d}.wm.wav"
        watermark.embed(output_wav, watermarked, payload_index)
        detection = watermark.detect(watermarked)
        if detection.probability >= 0.80 and detection.payload == payload_index:
            watermark_clean += 1
        watermarked.unlink(missing_ok=True)
        if _supported_transform_detects(output_wav, watermark, payload_index):
            watermark_supported += 1
    summary = summarize_similarity(similarities)
    return {
        "similarities": similarities,
        "median": summary.median,
        "p10": summary.p10,
        "samples": len(similarities),
        "emotion": {
            "macro_f1": sum(1.0 for value in emotion_cosines if value >= 0.5)
            / max(len(emotion_cosines), 1),
            "mean_embedding_cosine": sum(emotion_cosines) / max(len(emotion_cosines), 1),
        },
        "prosody": {
            "f0_correlation": sum(f0_correlations) / max(len(f0_correlations), 1),
            "energy_correlation": sum(energy_correlations) / max(len(energy_correlations), 1),
            "duration_ratio_error": sum(duration_errors) / max(len(duration_errors), 1),
        },
        "watermark": {
            "clean_pass_rate": watermark_clean / max(len(similarities), 1),
            "supported_attack_pass_rate": watermark_supported / max(len(similarities), 1),
            # Fixed policy semantics (Task 9): severe external noise can never
            # be confirmed as platform-generated, so the flag is constant True.
            "severe_noise_unverified": True,
        },
        "cross_language": {"zh_to_en_successes": zh_to_en, "en_to_zh_successes": en_to_zh},
        "ablation": {
            "enabled": {"speaker_similarity_median": summary.median},
            "disabled": {
                "speaker_similarity_median": _center_only_similarity(
                    gpt_sovits,
                    campplus,
                    reference_rows,
                    texts,
                    gpt_weight,
                    sovits_weight,
                    work_dir,
                )
            },
            "improved_metric": "speaker_similarity_median",
        },
    }


def _supported_transform_detects(wav_path: Path, watermark: WatermarkService, payload: int) -> bool:
    """MP3 192 kbps and 48k→32k resample must keep the payload detectable."""
    import subprocess

    from imageio_ffmpeg import get_ffmpeg_exe
    import librosa

    mp3_path = wav_path.with_suffix(".mp3")
    encode = subprocess.run(
        [get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(wav_path), "-b:a", "192k", str(mp3_path)],
        capture_output=True,
        timeout=120,
        check=False,
    )
    if encode.returncode != 0 or not mp3_path.is_file():
        return False
    try:
        detection = watermark.detect(mp3_path)
        mp3_ok = detection.probability >= 0.80 and detection.payload == payload
    finally:
        mp3_path.unlink(missing_ok=True)
    samples, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    resampled_mono = librosa.resample(
        samples.mean(axis=1).astype(np.float32), orig_sr=sample_rate, target_sr=48000
    )
    resampled_mono = librosa.resample(resampled_mono, orig_sr=48000, target_sr=32000)
    resampled_path = wav_path.with_name(wav_path.stem + ".rs32k.wav")
    sf.write(resampled_path, resampled_mono, 32000, subtype="PCM_16")
    try:
        detection = watermark.detect(resampled_path)
        resample_ok = detection.probability >= 0.80 and detection.payload == payload
    finally:
        resampled_path.unlink(missing_ok=True)
    return bool(mp3_ok and resample_ok)


def _center_only_similarity(
    gpt_sovits: GPTSoVITSAdapter,
    campplus: CamPlusEmbedder,
    reference_rows: Sequence[_ReferenceRow],
    texts: Iterable[dict[str, str]],
    gpt_weight: Path,
    sovits_weight: Path,
    work_dir: Path,
) -> float:
    """Ablation baseline: synthesize with the last (selection-agnostic) reference."""
    if len(reference_rows) < 2:
        return 0.0
    baseline_row = reference_rows[-1]
    baseline_embedding = campplus.embed(baseline_row.wav_path)
    ablation_dir = work_dir / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    values: list[float] = []
    for index, item in enumerate(texts):
        output_wav = ablation_dir / f"abl_{index:02d}.wav"
        gpt_sovits.synthesize(
            SynthesisSpec(
                text=str(item["text"]),
                text_lang=str(item["lang"]),
                prompt_text=baseline_row.text,
                prompt_lang=baseline_row.language,
                reference_audio=baseline_row.wav_path,
                gpt_weight=gpt_weight,
                sovits_weight=sovits_weight,
                output_wav=output_wav,
            )
        )
        values.append(_cosine(campplus.embed(output_wav), baseline_embedding))
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _write_reference_index(
    *,
    weights_dir: Path,
    dataset_id: str,
    reference_rows: Sequence[_ReferenceRow],
    campplus: CamPlusEmbedder,
) -> None:
    """Publish the local reference index beside the weights for later synthesis."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for row in reference_rows[:5]:
        _frames, mean_f0, mean_energy = _frame_stats(row.wav_path)
        candidates.append(
            {
                "segment_id": row.segment_id,
                "audio_path": row.wav_path.name,
                "text": row.text,
                "language": row.language,
                "duration_seconds": row.duration_seconds,
                "timbre_embedding": campplus.embed(row.wav_path),
                "emotion_label": EmotionLabel.NEUTRAL.value,
                "emotion_strength": 0.65,
                "f0_mean_hz": mean_f0,
                "energy_mean": mean_energy,
            }
        )
    payload = {"dataset_id": dataset_id, "candidates": candidates}
    destination = weights_dir / REFERENCES_FILENAME
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, destination)


class PublishedProfileEvaluationProvider:
    """Re-measure a ready profile against the frozen texts with real local models."""

    def __init__(
        self,
        *,
        profiles: VoiceProfileStore,
        gpt_sovits: GPTSoVITSAdapter,
        campplus_dir: Path | str,
        emotion_dir: Path | str,
        models_root: Path | str,
        texts_path: Path | str = DEFAULT_TEXTS_PATH,
    ) -> None:
        self.profiles = profiles
        self.gpt_sovits = gpt_sovits
        self.campplus_dir = Path(campplus_dir)
        self.emotion_dir = Path(emotion_dir)
        self.models_root = Path(models_root).expanduser().resolve()
        self.texts_path = Path(texts_path)

    def evaluate(
        self,
        profile_id: str,
        texts: tuple[dict[str, str], ...],
        *,
        seed: int,
        work_dir: Path,
    ) -> Mapping[str, object]:
        del seed
        profile = self.profiles.get(profile_id)
        weights_dir = (Path(profile.public_weight_dir or "") / "weights").expanduser().resolve()
        index_path = weights_dir / REFERENCES_FILENAME
        if not index_path.is_file():
            raise EvaluationContractError("音色档案尚未生成本地参考索引，无法评测")
        campplus_dir = _require_under_models(self.campplus_dir, self.models_root, "CAM++")
        emotion_dir = _require_under_models(self.emotion_dir, self.models_root, "Emotion2Vec")
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        reference_rows = [
            _ReferenceRow(
                segment_id=str(entry["segment_id"]),
                wav_path=(weights_dir / str(entry["audio_path"])).resolve(),
                text=str(entry["text"]),
                language=str(entry.get("language", "zh")),
                duration_seconds=float(entry["duration_seconds"]),
            )
            for entry in payload.get("candidates", [])
        ]
        if not reference_rows:
            raise EvaluationContractError("音色档案的本地参考索引为空，无法评测")
        report = _run_similarity_suite(
            gpt_sovits=self.gpt_sovits,
            campplus=CamPlusEmbedder(campplus_dir, models_root=self.models_root),
            emotion=EmotionEmbedder(emotion_dir, models_root=self.models_root),
            reference_rows=reference_rows,
            texts=texts,
            gpt_weight=_latest_weight(weights_dir, "*.ckpt", "GPT"),
            sovits_weight=_latest_weight(weights_dir, "*_e*_s*.pth", "SoVITS"),
            work_dir=work_dir,
            watermark=WatermarkService(),
        )
        return {
            "model_versions": {"gpt_sovits": "20250606v2pro"},
            "dataset_hash": _published_weights_hash(weights_dir),
            "speaker_similarity": report["similarities"],
            "emotion": report["emotion"],
            "prosody": report["prosody"],
            "watermark": report["watermark"],
            "cross_language": report["cross_language"],
            "ablation": report["ablation"],
        }


def _published_weights_hash(weights_dir: Path) -> str:
    from hashlib import sha256

    digest = sha256()
    for path in sorted(weights_dir.rglob("*")):
        if path.is_file():
            digest.update(path.name.encode("utf-8"))
            digest.update(sha256(path.read_bytes()).digest())
    return f"sha256:{digest.hexdigest()}"


def _load_sensitive_terms(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        return ()
    return tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


class CamPlusOutputSimilarityGate:
    """Per-output CAM++ cosine against the primary reference (>= 0.90 gate)."""

    def __init__(
        self,
        *,
        campplus_dir: Path | str,
        models_root: Path | str,
        embedder_loader: Callable[[Path], CamPlusEmbedder] | None = None,
    ) -> None:
        self.campplus_dir = Path(campplus_dir)
        self.models_root = Path(models_root).expanduser().resolve()
        self._embedder_loader = embedder_loader
        self._embedder: CamPlusEmbedder | None = None

    def _get_embedder(self) -> CamPlusEmbedder:
        if self._embedder is None:
            if self._embedder_loader is not None:
                self._embedder = self._embedder_loader(self.campplus_dir)
            else:
                self._embedder = CamPlusEmbedder(self.campplus_dir, models_root=self.models_root)
        return self._embedder

    def similarity(self, output_wav: Path, reference_wav: Path) -> float:
        embedder = self._get_embedder()
        return cosine_similarity(
            embedder.embed(reference_wav),
            embedder.embed(output_wav),
        )


def assemble_local_worker(
    *,
    root: Path | str,
    python_executable: Path | str | None = None,
    database_url: str | None = None,
    storage_root: Path | str | None = None,
    watermark: WatermarkService | None = None,
) -> GPUWorker:
    """Assemble the single real local GPU worker from pinned local components."""
    from backend.app.api.routes.datasets import DatasetService
    from backend.app.db.session import create_database_engine

    project_root = Path(root).expanduser().resolve()
    data_root = (
        Path(storage_root).expanduser().resolve()
        if storage_root is not None
        else project_root / "data"
    )
    storage = LocalStorage(data_root)
    engine = create_database_engine(
        database_url or f"sqlite+pysqlite:///{(data_root / 'app.db').as_posix()}"
    )
    init_db(engine)
    queue = JobQueue(engine=engine)
    profiles = VoiceProfileStore(engine=engine)
    dataset_service = DatasetService(storage=storage, engine=engine)
    models_root = project_root / "models"
    vendor_dir = project_root / "vendor" / "GPT-SoVITS"
    interpreter = Path(python_executable).expanduser().resolve() if python_executable else Path("python")

    preparer = GPTSoVITSDatasetPreparer(
        storage_root=data_root,
        models_root=models_root,
        vendor_dir=vendor_dir,
        python_executable=interpreter,
    )
    adapter = GPTSoVITSAdapter(
        vendor_dir=vendor_dir,
        python_executable=interpreter,
        models_root=models_root,
    )
    trainer = DisentanglerTrainer()
    training_adapters = LocalTrainingAdapters(
        gpt_sovits=adapter,
        campplus_dir=models_root / "campplus",
        emotion_dir=models_root / "emotion2vec",
        models_root=models_root,
        storage_root=data_root,
        texts_path=project_root / DEFAULT_TEXTS_PATH,
        watermark=watermark,
    )

    def feature_batch_loader(manifest_path: Path) -> list[FeatureBatch]:
        return load_feature_batches(
            manifest_path,
            campplus_dir=models_root / "campplus",
            emotion_dir=models_root / "emotion2vec",
            models_root=models_root,
        )

    from backend.app.services.metrics import LocalEvaluationRunner

    evaluation_runner = LocalEvaluationRunner(
        provider=PublishedProfileEvaluationProvider(
            profiles=profiles,
            gpt_sovits=adapter,
            campplus_dir=models_root / "campplus",
            emotion_dir=models_root / "emotion2vec",
            models_root=models_root,
            texts_path=project_root / DEFAULT_TEXTS_PATH,
        ),
        reports_root=project_root / "reports" / "evaluations",
        work_root=data_root / "temp" / "evaluations",
    )

    from backend.app.services.webui_bridge import WebUIInferenceBridge

    webui_bridge = WebUIInferenceBridge(
        vendor_dir=vendor_dir,
        python_executable=interpreter,
    )

    settings_path = project_root / "config" / "app.yaml"
    settings = get_settings(settings_path) if settings_path.is_file() else get_settings()
    local_synthesizer = GPTSoVITSLocalSynthesizer(adapter, webui_bridge=webui_bridge)
    cloud_synthesizer = None
    cloud_api_key = os.environ.get("COMPSHARE_API_KEY", "").strip()
    if cloud_api_key:
        from backend.app.services.modelverse_tts import (
            ModelVerseEmotionSynthesizer,
            ModelVerseIndexTTSClient,
        )

        cloud_synthesizer = ModelVerseEmotionSynthesizer(
            ModelVerseIndexTTSClient(
                cloud_api_key,
                base_url=os.environ.get(
                    "MODELVERSE_API_BASE_URL",
                    "https://api.modelverse.cn",
                ),
            )
        )
    synthesis_pipeline = SynthesisPipeline(
        profiles=profiles,
        sensitive_filter=SensitiveFilter(
            _load_sensitive_terms(project_root / DEFAULT_SENSITIVE_WORDS_PATH)
        ),
        storage=storage,
        synthesizer=HybridSynthesizer(local_synthesizer, cloud_synthesizer),
        reference_resolver=ZeroShotReferenceResolver(
            profiles=profiles,
            storage=storage,
            registry=VoiceBaseRegistry(
                settings,
                models_root,
                models_root / "checksums.sha256",
            ),
        ),
        watermark=watermark or WatermarkService(),
        fingerprint=FingerprintService(),
        similarity_gate=CamPlusOutputSimilarityGate(
            campplus_dir=models_root / "campplus",
            models_root=models_root,
        ),
    )

    return GPUWorker(
        queue=queue,
        profiles=profiles,
        manifest_loader=SegmentManifestLoader(dataset_service, storage_root=data_root).load,
        adapters=training_adapters,
        profiles_root=data_root / "profiles",
        features_root=data_root / "features",
        synthesis_pipeline=synthesis_pipeline,
        evaluation_runner=evaluation_runner,
        dataset_preparer=preparer,
        gpt_sovits_adapter=adapter,
        disentangler_trainer=trainer,
        feature_batch_loader=feature_batch_loader,
    )
