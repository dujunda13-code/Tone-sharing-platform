from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

import soundfile as sf

from backend.app.schemas.synthesis import SynthesisCreate
from backend.app.services.synthesis_pipeline import ReferenceCondition, SynthesisPipelineError
from backend.app.services.voice_profiles import VoiceProfileRecord


MAX_REFERENCE_BYTES = 20 * 1024 * 1024
MAX_OUTPUT_BYTES = 100 * 1024 * 1024
MODEL_ID = "IndexTeam/IndexTTS-2"
CONTENT_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg"}
LABEL_VECTOR_INDEX = {
    "happy": 0,
    "angry": 1,
    "sad": 2,
    "afraid": 3,
    "disgusted": 4,
    "melancholic": 5,
    "surprised": 6,
    "calm": 7,
}


class ModelVerseAPIError(RuntimeError):
    """Provider failure safe for logs and public error wrapping."""


class CloudTTSFailed(SynthesisPipelineError):
    code = "CLOUD_TTS_FAILED"


class ModelVerseIndexTTSClient:
    """Small client for ModelVerse IndexTTS-2's synchronous multipart endpoint."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.modelverse.cn",
        timeout_seconds: float = 600,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        if not api_key.strip():
            raise ValueError("ModelVerse API key is required")
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self._require_https(self.base_url)
        self.timeout_seconds = timeout_seconds
        self.opener = opener

    def synthesize_from_reference(
        self,
        reference_audio: Path,
        infer_payload: dict[str, Any],
        output_wav: Path,
    ) -> Path:
        reference, content_type = self._validate_reference(reference_audio)
        boundary = f"jianxi-{uuid4().hex}"
        body = self._build_multipart(reference, content_type, infer_payload, boundary)
        request = Request(
            f"{self.base_url}/v1/audio/infer",
            data=body,
            method="POST",
            headers={
                "Accept": "audio/wav",
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )
        audio = self._read_response(request)
        output = Path(output_wav).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(audio)
        try:
            info = sf.info(str(output))
        except (RuntimeError, OSError) as exc:
            output.unlink(missing_ok=True)
            raise ModelVerseAPIError("provider output is not a readable WAV file") from exc
        if info.frames <= 0 or info.channels <= 0:
            output.unlink(missing_ok=True)
            raise ModelVerseAPIError("provider output WAV is empty")
        return output

    @staticmethod
    def _validate_reference(reference_audio: Path) -> tuple[Path, str]:
        reference = Path(reference_audio).resolve()
        if not reference.is_file():
            raise ModelVerseAPIError("reference audio is unavailable")
        content_type = CONTENT_TYPES.get(reference.suffix.lower())
        if content_type is None:
            raise ModelVerseAPIError("reference audio format is not supported by the provider")
        size = reference.stat().st_size
        if size <= 0 or size > MAX_REFERENCE_BYTES:
            raise ModelVerseAPIError("reference audio size is outside the provider limit")
        try:
            info = sf.info(str(reference))
        except (RuntimeError, OSError) as exc:
            raise ModelVerseAPIError("reference audio is unreadable") from exc
        if not 5.0 <= info.duration <= 30.0:
            raise ModelVerseAPIError("reference audio duration is outside the provider limit")
        if info.samplerate < 16_000:
            raise ModelVerseAPIError("reference audio sample rate is below the provider limit")
        return reference, content_type

    @staticmethod
    def _build_multipart(
        reference: Path,
        content_type: str,
        infer_payload: dict[str, Any],
        boundary: str,
    ) -> bytes:
        chunks: list[bytes] = []

        def append_field(name: str, value: bytes, *, field_type: str) -> None:
            chunks.extend(
                [
                    f"--{boundary}\r\n".encode("ascii"),
                    f'Content-Disposition: form-data; name="{name}"\r\n'.encode("ascii"),
                    f"Content-Type: {field_type}\r\n\r\n".encode("ascii"),
                    value,
                    b"\r\n",
                ]
            )

        append_field("model", MODEL_ID.encode("utf-8"), field_type="text/plain; charset=utf-8")
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("ascii"),
                (
                    'Content-Disposition: form-data; name="spk_audio_file"; '
                    f'filename="reference{reference.suffix.lower()}"\r\n'
                ).encode("ascii"),
                f"Content-Type: {content_type}\r\n\r\n".encode("ascii"),
                reference.read_bytes(),
                b"\r\n",
            ]
        )
        payload = json.dumps(
            infer_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        append_field("payload", payload, field_type="application/json; charset=utf-8")
        chunks.append(f"--{boundary}--\r\n".encode("ascii"))
        return b"".join(chunks)

    def _read_response(self, request: Request) -> bytes:
        try:
            with self.opener(request, timeout=self.timeout_seconds) as response:
                body = response.read(MAX_OUTPUT_BYTES + 1)
        except HTTPError as exc:
            raise ModelVerseAPIError(
                f"provider request was rejected with HTTP {exc.code}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ModelVerseAPIError("provider request failed") from exc
        if len(body) > MAX_OUTPUT_BYTES:
            raise ModelVerseAPIError("provider response exceeded the size limit")
        return body

    @staticmethod
    def _require_https(url: str) -> None:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ModelVerseAPIError("provider URL must use HTTPS")


class ModelVerseEmotionSynthesizer:
    def __init__(self, client: ModelVerseIndexTTSClient) -> None:
        self.client = client

    def synthesize(
        self,
        request: SynthesisCreate,
        profile: VoiceProfileRecord,
        reference: ReferenceCondition,
        output_wav: Path,
        *,
        seed: int = 0,
    ) -> Path:
        del profile, seed
        if request.synthesis_mode != "emotion_api" or request.cloud_options is None:
            raise CloudTTSFailed("cloud emotion synthesis configuration is missing")
        try:
            return self.client.synthesize_from_reference(
                reference.reference_audio,
                self.build_infer_payload(request),
                output_wav,
            )
        except ModelVerseAPIError as exc:
            raise CloudTTSFailed("cloud emotion synthesis failed") from exc

    @staticmethod
    def build_infer_payload(request: SynthesisCreate) -> dict[str, Any]:
        options = request.cloud_options
        if options is None:
            raise ValueError("cloud options are required")
        payload: dict[str, Any] = {
            "input": request.text,
            "sample_rate": options.sample_rate,
            "speed": options.speed,
            "gain": options.gain,
            "emo_control_method": 0,
            "emo_alpha": options.emotion_strength,
            "use_random": options.use_random,
            "interval_silence": options.interval_silence,
        }
        if options.control_mode == "auto":
            payload["emo_control_method"] = 3
            payload["emo_text"] = request.text
        elif options.control_mode == "description":
            payload["emo_control_method"] = 3
            payload["emo_text"] = options.description
        elif options.control_mode == "vector":
            payload["emo_control_method"] = 2
            payload["emo_vec"] = options.emotion_vector
        elif options.control_mode == "label" and options.label is not None:
            index = LABEL_VECTOR_INDEX.get(options.label.value)
            if index is not None:
                vector = [0.0] * 8
                vector[index] = 1.0
                payload["emo_control_method"] = 2
                payload["emo_vec"] = vector
        return payload
