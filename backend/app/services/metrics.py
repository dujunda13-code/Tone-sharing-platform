from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np


EXPECTED_EVALUATION_SAMPLES = 20
EVALUATION_SEED = 20260902
SPEAKER_MEDIAN_MINIMUM = 0.90
SPEAKER_P10_MINIMUM = 0.85
SUPPORTED_ATTACK_PASS_RATE_MINIMUM = 1.0
ABLATION_IMPROVEMENT_MINIMUM = 0.01


class EvaluationContractError(ValueError):
    """Raised when a local evaluation input or result violates the fixed contract."""


@dataclass(frozen=True)
class SimilaritySummary:
    median: float
    p10: float
    samples: int
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "median": self.median,
            "p10": self.p10,
            "samples": self.samples,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class EvaluationReport:
    profile_id: str
    model_versions: dict[str, object]
    dataset_hash: str
    speaker_similarity: SimilaritySummary
    emotion: dict[str, float]
    prosody: dict[str, float]
    watermark: dict[str, object]
    cross_language: dict[str, int]
    ablation: dict[str, object]
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "model_versions": dict(self.model_versions),
            "dataset_hash": self.dataset_hash,
            "speaker_similarity": self.speaker_similarity.to_dict(),
            "emotion": dict(self.emotion),
            "prosody": dict(self.prosody),
            "watermark": dict(self.watermark),
            "cross_language": dict(self.cross_language),
            "ablation": dict(self.ablation),
            "passed": self.passed,
        }


class EvaluationRunner(Protocol):
    def run(self, job_id: str, payload: dict[str, object]) -> Mapping[str, object]: ...


class EvaluationProvider(Protocol):
    def evaluate(
        self,
        profile_id: str,
        texts: tuple[dict[str, str], ...],
        *,
        seed: int,
        work_dir: Path,
    ) -> Mapping[str, object]: ...


def summarize_similarity(values: Sequence[float]) -> SimilaritySummary:
    if len(values) != EXPECTED_EVALUATION_SAMPLES:
        raise EvaluationContractError(
            f"speaker similarity requires exactly {EXPECTED_EVALUATION_SAMPLES} samples"
        )
    normalized: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise EvaluationContractError("speaker similarity contains a non-numeric value") from exc
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise EvaluationContractError("speaker similarity values must be finite in [0, 1]")
        normalized.append(numeric)
    array = np.asarray(normalized, dtype=np.float64)
    median = float(np.median(array))
    p10 = float(np.percentile(array, 10))
    return SimilaritySummary(
        median=median,
        p10=p10,
        samples=len(normalized),
        passed=median >= SPEAKER_MEDIAN_MINIMUM and p10 >= SPEAKER_P10_MINIMUM,
    )


def load_evaluation_texts(path: Path | str) -> tuple[dict[str, str], ...]:
    text_path = Path(path).resolve()
    if not text_path.is_file():
        raise EvaluationContractError("evaluation text file is unavailable")
    try:
        payload = json.loads(text_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationContractError("evaluation text file is invalid") from exc
    if not isinstance(payload, list) or len(payload) != EXPECTED_EVALUATION_SAMPLES:
        raise EvaluationContractError(
            f"evaluation text set requires exactly {EXPECTED_EVALUATION_SAMPLES} items"
        )
    texts: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for item in payload:
        if not isinstance(item, dict) or set(item) != {"id", "lang", "text"}:
            raise EvaluationContractError("evaluation text entries must contain id, lang and text")
        item_id = item["id"]
        lang = item["lang"]
        text = item["text"]
        if (
            not isinstance(item_id, str)
            or not item_id
            or item_id in seen_ids
            or lang not in {"zh", "en"}
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise EvaluationContractError("evaluation text entries are invalid")
        seen_ids.add(item_id)
        texts.append({"id": item_id, "lang": lang, "text": text})
    if sum(item["lang"] == "zh" for item in texts) != 10 or sum(
        item["lang"] == "en" for item in texts
    ) != 10:
        raise EvaluationContractError("evaluation text set requires ten zh and ten en items")
    return tuple(texts)


def _bounded_float(mapping: Mapping[str, Any], key: str, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        value = float(mapping[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationContractError(f"evaluation metric {key!r} is invalid") from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise EvaluationContractError(f"evaluation metric {key!r} is outside its allowed range")
    return value


def _non_negative_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationContractError(f"evaluation count {key!r} is invalid")
    return value


def _ablation_report(raw: Mapping[str, Any]) -> tuple[dict[str, object], bool]:
    enabled = raw.get("enabled")
    disabled = raw.get("disabled")
    if not isinstance(enabled, Mapping) or not isinstance(disabled, Mapping):
        raise EvaluationContractError("ablation metrics are incomplete")
    enabled_values = {
        "speaker_similarity_median": _bounded_float(enabled, "speaker_similarity_median"),
        "emotion_mean_embedding_cosine": _bounded_float(enabled, "emotion_mean_embedding_cosine"),
    }
    disabled_values = {
        "speaker_similarity_median": _bounded_float(disabled, "speaker_similarity_median"),
        "emotion_mean_embedding_cosine": _bounded_float(disabled, "emotion_mean_embedding_cosine"),
    }
    improved_metric = ""
    if (
        enabled_values["speaker_similarity_median"]
        - disabled_values["speaker_similarity_median"]
        >= ABLATION_IMPROVEMENT_MINIMUM
    ):
        improved_metric = "speaker_similarity.median"
    elif (
        enabled_values["emotion_mean_embedding_cosine"]
        - disabled_values["emotion_mean_embedding_cosine"]
        >= ABLATION_IMPROVEMENT_MINIMUM
    ):
        improved_metric = "emotion.mean_embedding_cosine"
    return {
        "enabled": enabled_values,
        "disabled": disabled_values,
        "improved_metric": improved_metric,
    }, bool(improved_metric)


def build_evaluation_report(profile_id: str, raw: Mapping[str, object]) -> EvaluationReport:
    if not profile_id or not isinstance(raw, Mapping):
        raise EvaluationContractError("evaluation report input is invalid")
    model_versions = raw.get("model_versions", {})
    if not isinstance(model_versions, Mapping):
        raise EvaluationContractError("model_versions must be an object")
    dataset_hash = raw.get("dataset_hash")
    if (
        not isinstance(dataset_hash, str)
        or not dataset_hash.startswith("sha256:")
        or len(dataset_hash) != len("sha256:") + 64
        or not all(character in "0123456789abcdefABCDEF" for character in dataset_hash[7:])
    ):
        raise EvaluationContractError("dataset_hash must be a sha256 digest")
    speaker_values = raw.get("speaker_similarity")
    if not isinstance(speaker_values, Sequence) or isinstance(speaker_values, (str, bytes)):
        raise EvaluationContractError("speaker_similarity must be a sample sequence")
    speaker_similarity = summarize_similarity(speaker_values)

    emotion_raw = raw.get("emotion")
    if not isinstance(emotion_raw, Mapping):
        raise EvaluationContractError("emotion metrics are incomplete")
    emotion = {
        "macro_f1": _bounded_float(emotion_raw, "macro_f1"),
        "mean_embedding_cosine": _bounded_float(emotion_raw, "mean_embedding_cosine"),
    }
    prosody_raw = raw.get("prosody")
    if not isinstance(prosody_raw, Mapping):
        raise EvaluationContractError("prosody metrics are incomplete")
    prosody = {
        "f0_correlation": _bounded_float(prosody_raw, "f0_correlation", minimum=-1.0),
        "energy_correlation": _bounded_float(prosody_raw, "energy_correlation", minimum=-1.0),
        "duration_ratio_error": _bounded_float(prosody_raw, "duration_ratio_error", maximum=float("inf")),
    }

    watermark_raw = raw.get("watermark")
    if not isinstance(watermark_raw, Mapping):
        raise EvaluationContractError("watermark metrics are incomplete")
    watermark = {
        "clean_pass_rate": _bounded_float(watermark_raw, "clean_pass_rate"),
        "supported_attack_pass_rate": _bounded_float(
            watermark_raw, "supported_attack_pass_rate"
        ),
        "severe_noise_unverified": watermark_raw.get("severe_noise_unverified") is True,
    }
    cross_language_raw = raw.get("cross_language")
    if not isinstance(cross_language_raw, Mapping):
        raise EvaluationContractError("cross-language metrics are incomplete")
    cross_language = {
        "zh_to_en_successes": _non_negative_int(cross_language_raw, "zh_to_en_successes"),
        "en_to_zh_successes": _non_negative_int(cross_language_raw, "en_to_zh_successes"),
    }
    ablation, ablation_passed = _ablation_report(raw.get("ablation", {}))
    passed = (
        speaker_similarity.passed
        and watermark["clean_pass_rate"] == 1.0
        and watermark["supported_attack_pass_rate"]
        >= SUPPORTED_ATTACK_PASS_RATE_MINIMUM
        and watermark["severe_noise_unverified"] is True
        and cross_language["zh_to_en_successes"] >= 5
        and cross_language["en_to_zh_successes"] >= 5
        and ablation_passed
    )
    return EvaluationReport(
        profile_id=profile_id,
        model_versions={str(key): value for key, value in model_versions.items()},
        dataset_hash=dataset_hash,
        speaker_similarity=speaker_similarity,
        emotion=emotion,
        prosody=prosody,
        watermark=watermark,
        cross_language=cross_language,
        ablation=ablation,
        passed=passed,
    )


class LocalEvaluationRunner:
    """Run an injected local evaluator and atomically persist its machine-readable report."""

    def __init__(
        self,
        *,
        provider: EvaluationProvider,
        texts_path: Path | str = Path("config/evaluation_texts.zh-en.json"),
        reports_root: Path | str = Path("reports/evaluations"),
        work_root: Path | str = Path("data/temp/evaluations"),
    ) -> None:
        self.provider = provider
        self.texts_path = Path(texts_path).resolve()
        self.reports_root = Path(reports_root).resolve()
        self.work_root = Path(work_root).resolve()

    def run(self, job_id: str, payload: dict[str, object]) -> Mapping[str, object]:
        profile_id = payload.get("profile_id")
        if not isinstance(profile_id, str) or Path(profile_id).name != profile_id:
            raise EvaluationContractError("evaluation job profile_id is invalid")
        if not job_id or Path(job_id).name != job_id:
            raise EvaluationContractError("evaluation job id is invalid")
        texts = load_evaluation_texts(self.texts_path)
        work_dir = self.work_root / profile_id / job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        raw = self.provider.evaluate(
            profile_id,
            texts,
            seed=EVALUATION_SEED,
            work_dir=work_dir,
        )
        report = build_evaluation_report(profile_id, raw)
        report_path = self.reports_root / profile_id / f"{job_id}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_name(f".{report_path.name}.tmp")
        temporary.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, report_path)
        return {
            "report_path": report_path.relative_to(self.reports_root).as_posix(),
            "report": report.to_dict(),
        }
