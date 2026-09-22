from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.services.metrics import (
    EvaluationContractError,
    LocalEvaluationRunner,
    build_evaluation_report,
    load_evaluation_texts,
    summarize_similarity,
)


ROOT = Path(__file__).resolve().parents[3]
TEXTS_PATH = ROOT / "config" / "evaluation_texts.zh-en.json"


def _raw_metrics() -> dict[str, object]:
    return {
        "model_versions": {"gpt_sovits": "20250606v2pro"},
        "dataset_hash": "sha256:" + "a" * 64,
        "speaker_similarity": [0.91] * 20,
        "emotion": {"macro_f1": 0.82, "mean_embedding_cosine": 0.91},
        "prosody": {
            "f0_correlation": 0.81,
            "energy_correlation": 0.84,
            "duration_ratio_error": 0.06,
        },
        "watermark": {
            "clean_pass_rate": 1.0,
            "supported_attack_pass_rate": 1.0,
            "severe_noise_unverified": True,
        },
        "cross_language": {"zh_to_en_successes": 5, "en_to_zh_successes": 5},
        "ablation": {
            "enabled": {
                "speaker_similarity_median": 0.92,
                "emotion_mean_embedding_cosine": 0.91,
            },
            "disabled": {
                "speaker_similarity_median": 0.90,
                "emotion_mean_embedding_cosine": 0.90,
            },
        },
    }


def test_similarity_gate_uses_median_and_p10():
    values = [0.91] * 18 + [0.84, 0.86]

    report = summarize_similarity(values)

    assert report.median >= 0.90
    assert report.p10 >= 0.85
    assert report.samples == 20
    assert report.passed is True


def test_gate_fails_without_exactly_twenty_samples():
    with pytest.raises(EvaluationContractError, match="exactly 20"):
        summarize_similarity([0.95] * 19)


def test_evaluation_report_requires_severe_noise_to_remain_unverified():
    raw = _raw_metrics()
    raw["watermark"] = {
        "clean_pass_rate": 1.0,
        "supported_attack_pass_rate": 1.0,
        "severe_noise_unverified": False,
    }

    report = build_evaluation_report("profile-1", raw)

    assert report.passed is False
    assert report.watermark["severe_noise_unverified"] is False


def test_evaluation_report_rejects_a_non_sha256_dataset_hash():
    raw = _raw_metrics()
    raw["dataset_hash"] = "sha256:" + "z" * 64

    with pytest.raises(EvaluationContractError, match="sha256"):
        build_evaluation_report("profile-1", raw)


def test_evaluation_report_passes_only_with_ablation_and_cross_language_gates():
    report = build_evaluation_report("profile-1", _raw_metrics())

    assert report.passed is True
    assert report.speaker_similarity.median == pytest.approx(0.91)
    assert report.speaker_similarity.p10 == pytest.approx(0.91)
    assert report.ablation["improved_metric"] in {
        "speaker_similarity.median",
        "emotion.mean_embedding_cosine",
    }


def test_frozen_evaluation_texts_have_ten_examples_per_language():
    texts = load_evaluation_texts(TEXTS_PATH)

    assert len(texts) == 20
    assert sum(item["lang"] == "zh" for item in texts) == 10
    assert sum(item["lang"] == "en" for item in texts) == 10
    assert len({item["id"] for item in texts}) == 20


def test_evaluation_text_file_is_machine_readable():
    payload = json.loads(TEXTS_PATH.read_text(encoding="utf-8"))

    assert isinstance(payload, list)
    assert all(set(item) == {"id", "lang", "text"} for item in payload)


def test_local_evaluation_runner_persists_a_reproducible_report(tmp_path: Path):
    class Provider:
        def evaluate(self, profile_id, texts, *, seed, work_dir):
            assert profile_id == "profile-1"
            assert len(texts) == 20
            assert seed == 20260902
            assert work_dir.is_dir()
            return _raw_metrics()

    runner = LocalEvaluationRunner(
        provider=Provider(),
        texts_path=TEXTS_PATH,
        reports_root=tmp_path / "reports",
        work_root=tmp_path / "data" / "temp",
    )

    result = runner.run("job-1", {"profile_id": "profile-1"})

    report_path = tmp_path / "reports" / "profile-1" / "job-1.json"
    assert result["report_path"] == "profile-1/job-1.json"
    assert report_path.is_file()
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is True
