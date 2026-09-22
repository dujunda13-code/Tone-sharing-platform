from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.schemas.synthesis import SynthesisCreate


def _base_payload() -> dict[str, object]:
    return {
        "voice_profile_id": "voice-1",
        "text": "你好，世界。",
        "text_lang": "zh",
        "emotion": {"mode": "auto", "strength": 0.65},
        "consent_confirmed": True,
    }


def test_local_mode_accepts_official_webui_controls() -> None:
    request = SynthesisCreate.model_validate(
        {
            **_base_payload(),
            "synthesis_mode": "local",
            "local_options": {
                "cut_method": "punctuation",
                "speed": 1.2,
                "pause_seconds": 0.25,
                "top_k": 30,
                "top_p": 0.8,
                "temperature": 0.7,
            },
        }
    )

    assert request.synthesis_mode == "local"
    assert request.local_options.cut_method == "punctuation"
    assert request.local_options.top_k == 30


def test_local_mode_rejects_cloud_only_language() -> None:
    with pytest.raises(ValidationError, match="本地合成只支持中文和英文"):
        SynthesisCreate.model_validate({**_base_payload(), "text_lang": "ja"})


def test_emotion_api_requires_explicit_cloud_consent() -> None:
    payload = {
        **_base_payload(),
        "synthesis_mode": "emotion_api",
        "text_lang": "ja",
        "cloud_options": {"control_mode": "label", "label": "happy"},
    }

    with pytest.raises(ValidationError, match="云端处理授权"):
        SynthesisCreate.model_validate(payload)


def test_emotion_api_accepts_documented_index_tts_controls() -> None:
    request = SynthesisCreate.model_validate(
        {
            **_base_payload(),
            "synthesis_mode": "emotion_api",
            "text_lang": "ja",
            "cloud_processing_confirmed": True,
            "cloud_options": {
                "control_mode": "label",
                "label": "surprised",
                "emotion_strength": 0.8,
                "sample_rate": 48000,
                "speed": 3.5,
                "gain": 1.2,
                "use_random": True,
                "interval_silence": 350,
            },
        }
    )

    assert request.text_lang == "ja"
    assert request.cloud_options is not None
    assert request.cloud_options.label.value == "surprised"
    assert request.cloud_options.sample_rate == 48000
    assert request.cloud_options.speed == 3.5
    assert request.cloud_options.gain == 1.2
    assert request.cloud_options.use_random is True
    assert request.cloud_options.interval_silence == 350


def test_emotion_api_vector_obeys_provider_contract() -> None:
    payload = {
        **_base_payload(),
        "synthesis_mode": "emotion_api",
        "cloud_processing_confirmed": True,
        "cloud_options": {
            "control_mode": "vector",
            "emotion_vector": [1.0, 0, 0, 0, 0, 0, 0, 0.2],
            "emotion_vector_mode": "mixed",
        },
    }
    request = SynthesisCreate.model_validate(payload)
    assert request.cloud_options is not None
    assert sum(request.cloud_options.emotion_vector or []) == pytest.approx(1.2)

    payload["cloud_options"] = {
        "control_mode": "vector",
        "emotion_vector": [1.0, 0.6, 0, 0, 0, 0, 0, 0],
        "emotion_vector_mode": "mixed",
    }
    with pytest.raises(ValidationError, match="元素之和"):
        SynthesisCreate.model_validate(payload)


def test_emotion_api_rejects_text_longer_than_provider_limit() -> None:
    with pytest.raises(ValidationError, match="600"):
        SynthesisCreate.model_validate(
            {
                **_base_payload(),
                "text": "情" * 601,
                "synthesis_mode": "emotion_api",
                "cloud_processing_confirmed": True,
                "cloud_options": {"control_mode": "label", "label": "happy"},
            }
        )
