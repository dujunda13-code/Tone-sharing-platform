from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from urllib.request import Request

import numpy as np
import pytest
import soundfile as sf

from backend.app.schemas.synthesis import CloudEmotionOptions, EmotionControl, SynthesisCreate
from backend.app.services.modelverse_tts import (
    ModelVerseAPIError,
    ModelVerseEmotionSynthesizer,
    ModelVerseIndexTTSClient,
)


class FakeResponse:
    def __init__(self, body: bytes, *, content_type: str = "audio/wav") -> None:
        self.body = body
        self.headers = {"Content-Type": content_type}

    def read(self, amount: int = -1) -> bytes:
        return self.body if amount < 0 else self.body[:amount]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _write_wav(path: Path, *, seconds: float, sample_rate: int = 16_000) -> None:
    sf.write(
        path,
        np.zeros(round(sample_rate * seconds), dtype=np.float32),
        sample_rate,
        subtype="PCM_16",
    )


def _wav_bytes(*, seconds: float = 1.0, sample_rate: int = 22_050) -> bytes:
    output = BytesIO()
    sf.write(
        output,
        np.zeros(round(sample_rate * seconds), dtype=np.float32),
        sample_rate,
        format="WAV",
        subtype="PCM_16",
    )
    return output.getvalue()


def test_provider_client_posts_one_documented_multipart_request(tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    _write_wav(reference, seconds=5)
    calls: list[Request] = []

    def open_request(request: Request, *, timeout: float):
        del timeout
        calls.append(request)
        return FakeResponse(_wav_bytes())

    client = ModelVerseIndexTTSClient("test-key", opener=open_request)
    output = tmp_path / "output.wav"
    client.synthesize_from_reference(
        reference,
        {
            "input": "你好",
            "sample_rate": 44_100,
            "speed": 1.1,
            "gain": 1.0,
            "emo_control_method": 2,
            "emo_alpha": 0.7,
            "emo_vec": [0.7, 0, 0, 0, 0, 0, 0, 0],
            "use_random": False,
            "interval_silence": 200,
        },
        output,
    )

    assert output.is_file()
    assert len(calls) == 1
    request = calls[0]
    assert request.full_url == "https://api.modelverse.cn/v1/audio/infer"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer test-key"
    assert (request.get_header("Content-type") or "").startswith("multipart/form-data; boundary=")
    body = request.data or b""
    assert b'name="model"' in body
    assert b"IndexTeam/IndexTTS-2" in body
    assert b'name="spk_audio_file"; filename="reference.wav"' in body
    assert b'name="payload"' in body
    assert json.dumps(44_100).encode() in body
    assert '"emo_control_method":2'.encode() in body


def test_provider_client_rejects_reference_shorter_than_five_seconds(tmp_path: Path) -> None:
    reference = tmp_path / "too-short.wav"
    _write_wav(reference, seconds=4.9)
    client = ModelVerseIndexTTSClient(
        "test-key",
        opener=lambda *_args, **_kwargs: pytest.fail("provider must not be called"),
    )

    with pytest.raises(ModelVerseAPIError, match="duration"):
        client.synthesize_from_reference(reference, {"input": "你好"}, tmp_path / "output.wav")


def _request(options: CloudEmotionOptions) -> SynthesisCreate:
    return SynthesisCreate(
        voice_profile_id="voice-1",
        text="今天真是太棒了",
        text_lang="zh",
        emotion=EmotionControl(mode="auto"),
        consent_confirmed=True,
        synthesis_mode="emotion_api",
        cloud_processing_confirmed=True,
        cloud_options=options,
    )


def test_emotion_label_builds_documented_eight_dimension_vector() -> None:
    payload = ModelVerseEmotionSynthesizer.build_infer_payload(
        _request(
            CloudEmotionOptions(
                control_mode="label",
                label="happy",
                emotion_strength=0.7,
                sample_rate=48_000,
                speed=1.25,
                gain=1.1,
                use_random=True,
                interval_silence=320,
            )
        )
    )

    assert payload == {
        "input": "今天真是太棒了",
        "sample_rate": 48_000,
        "speed": 1.25,
        "gain": 1.1,
        "emo_control_method": 2,
        "emo_alpha": 0.7,
        "emo_vec": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "use_random": True,
        "interval_silence": 320,
    }


def test_emotion_description_uses_documented_text_control() -> None:
    payload = ModelVerseEmotionSynthesizer.build_infer_payload(
        _request(
            CloudEmotionOptions(
                control_mode="description",
                description="克制但明显的悲伤",
            )
        )
    )

    assert payload["emo_control_method"] == 3
    assert payload["emo_text"] == "克制但明显的悲伤"
    assert "emo_vec" not in payload
