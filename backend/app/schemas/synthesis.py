from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.schemas.common import EmotionLabel, JobStatus


Language = Literal["zh", "en"]
SynthesisLanguage = Literal["zh", "en", "ja", "es", "ar"]
SynthesisMode = Literal["local", "emotion_api"]


class CloudEmotionLabel(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    ANGRY = "angry"
    SAD = "sad"
    AFRAID = "afraid"
    DISGUSTED = "disgusted"
    MELANCHOLIC = "melancholic"
    SURPRISED = "surprised"
    CALM = "calm"


class LocalSynthesisOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cut_method: Literal[
        "none",
        "four_sentences",
        "fifty_chars",
        "zh_period",
        "en_period",
        "punctuation",
    ] = "none"
    speed: float = Field(default=1.0, ge=0.6, le=1.65)
    pause_seconds: float = Field(default=0.3, ge=0.1, le=0.5)
    top_k: int = Field(default=15, ge=1, le=100)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    temperature: float = Field(default=1.0, ge=0.0, le=1.0)


EmotionVector = Annotated[
    list[Annotated[float, Field(ge=0.0, le=1.5)]],
    Field(min_length=8, max_length=8),
]


class CloudEmotionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    control_mode: Literal["auto", "label", "description", "vector"] = "label"
    label: CloudEmotionLabel | None = None
    description: str | None = Field(default=None, min_length=1, max_length=200)
    emotion_strength: float = Field(default=0.8, ge=0.0, le=1.0)
    sample_rate: Literal[22050, 44100, 48000] = 44100
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    gain: float = Field(default=1.0, gt=0.0, le=4.0)
    use_random: bool = False
    interval_silence: int = Field(default=200, ge=0, le=5000)
    emotion_vector: EmotionVector | None = None
    emotion_vector_mode: Literal["single", "mixed"] | None = None

    @model_validator(mode="after")
    def require_mode_specific_input(self) -> "CloudEmotionOptions":
        if self.control_mode == "label" and self.label is None:
            raise ValueError("标签情绪控制需要目标情绪")
        if self.control_mode == "description" and not self.description:
            raise ValueError("描述情绪控制需要情绪描述")
        if self.control_mode == "vector":
            if self.emotion_vector is None or self.emotion_vector_mode is None:
                raise ValueError("向量情绪控制需要八维向量和向量模式")
            non_zero = sum(value > 0 for value in self.emotion_vector)
            total = sum(self.emotion_vector)
            if total <= 0 or total > 1.5 + 1e-9:
                raise ValueError("情绪向量元素之和必须大于 0 且不超过 1.5")
            if self.emotion_vector_mode == "single" and non_zero != 1:
                raise ValueError("single 向量模式必须且只能有一个非零元素")
            if self.emotion_vector_mode == "mixed" and non_zero < 2:
                raise ValueError("mixed 向量模式至少需要两个非零元素")
        return self


class EmotionControl(BaseModel):
    mode: Literal["auto", "manual", "reference"]
    label: EmotionLabel | None = None
    strength: float = Field(default=0.65, ge=0.0, le=1.0)
    reference_audio_id: str | None = Field(default=None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_mode_specific_input(self) -> "EmotionControl":
        if self.mode == "manual" and self.label is None:
            raise ValueError("manual emotion control requires a label")
        if self.mode == "reference" and self.reference_audio_id is None:
            raise ValueError("reference emotion control requires a reference_audio_id")
        return self


class SynthesisCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    voice_profile_id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=1000)
    text_lang: SynthesisLanguage
    emotion: EmotionControl
    consent_confirmed: Literal[True]
    synthesis_mode: SynthesisMode = "local"
    local_options: LocalSynthesisOptions = Field(default_factory=LocalSynthesisOptions)
    cloud_options: CloudEmotionOptions | None = None
    cloud_processing_confirmed: bool = False

    @model_validator(mode="after")
    def validate_synthesis_mode(self) -> "SynthesisCreate":
        if self.synthesis_mode == "local" and self.text_lang not in {"zh", "en"}:
            raise ValueError("本地合成只支持中文和英文")
        if self.synthesis_mode == "emotion_api":
            if self.cloud_processing_confirmed is not True:
                raise ValueError("情绪合成需要明确确认云端处理授权")
            if self.cloud_options is None:
                raise ValueError("情绪合成需要云端情绪配置")
            if len(self.text) > 600:
                raise ValueError("云端情绪合成文本不能超过 600 个字符")
        return self


class SynthesisQueuedResponse(BaseModel):
    job_id: str
    status: Literal["queued"]


class SynthesisJobResponse(BaseModel):
    id: str
    status: JobStatus
    error_code: str | None = None
    error_message: str | None = None
    public_audio_path: str | None = None


class WatermarkVerificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1, max_length=64)


class WatermarkVerificationResponse(BaseModel):
    job_id: str
    probability: float
    payload: int | None
    payload_matches_job: bool
