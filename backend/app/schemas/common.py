from enum import Enum


class JobKind(str, Enum):
    PREPROCESS = "preprocess"
    TRAIN = "train"
    SYNTHESIZE = "synthesize"
    EVALUATE = "evaluate"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class EmotionLabel(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    SURPRISED = "surprised"
    OTHER = "other"
