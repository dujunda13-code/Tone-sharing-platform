import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from backend.app.schemas.common import EmotionLabel


LOW_CONFIDENCE_THRESHOLD = 0.55


@dataclass(frozen=True)
class EmotionResult:
    label: EmotionLabel
    confidence: float
    scores: dict[EmotionLabel, float]
    embedding_path: Path


DEFAULT_EMOTION_MODEL_DIR = Path("models") / "emotion2vec"
REQUIRED_EMOTION_ARTIFACTS = ("config.yaml", "model.pt", "tokens.txt")


def _local_model_dir() -> Path:
    return Path(DEFAULT_EMOTION_MODEL_DIR).expanduser().resolve()


def _local_model_is_available() -> bool:
    model_dir = _local_model_dir()
    return model_dir.is_dir() and all(
        (model_dir / artifact).is_file() for artifact in REQUIRED_EMOTION_ARTIFACTS
    )


def _default_model_loader() -> Any:
    from funasr import AutoModel

    model_dir = _local_model_dir()
    if not _local_model_is_available():
        raise RuntimeError(
            "本地 Emotion2Vec 模型未就绪，请先在部署阶段下载到 models/emotion2vec"
        )
    return AutoModel(model=str(model_dir), device="cpu", disable_update=True)


def _map_label(raw_label: str) -> EmotionLabel:
    value = raw_label.casefold().strip()
    if "/" in value:
        value = value.rsplit("/", 1)[-1]
    aliases = {
        "neutral": EmotionLabel.NEUTRAL,
        "中性": EmotionLabel.NEUTRAL,
        "happy": EmotionLabel.HAPPY,
        "高兴": EmotionLabel.HAPPY,
        "sad": EmotionLabel.SAD,
        "悲伤": EmotionLabel.SAD,
        "angry": EmotionLabel.ANGRY,
        "生气": EmotionLabel.ANGRY,
        "fear": EmotionLabel.FEARFUL,
        "fearful": EmotionLabel.FEARFUL,
        "恐惧": EmotionLabel.FEARFUL,
        "disgust": EmotionLabel.DISGUSTED,
        "disgusted": EmotionLabel.DISGUSTED,
        "厌恶": EmotionLabel.DISGUSTED,
        "surprise": EmotionLabel.SURPRISED,
        "surprised": EmotionLabel.SURPRISED,
        "惊讶": EmotionLabel.SURPRISED,
    }
    return aliases.get(value, EmotionLabel.OTHER)


class EmotionAnalyzer:
    def __init__(
        self,
        model_loader: Callable[[], Any] | None = None,
        *,
        embedding_root: Path | str = Path("data/features/emotion"),
        confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._model_loader = model_loader or _default_model_loader
        self._model: Any | None = None
        self.embedding_root = Path(embedding_root)
        self.confidence_threshold = confidence_threshold

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = self._model_loader()
        return self._model

    def available(self) -> bool:
        """Check fixed local Emotion2Vec artifacts without loading or downloading a model."""
        return _local_model_is_available()

    @staticmethod
    def _prediction(model: Any, audio_path: Path) -> tuple[list[str], list[float], Any]:
        if hasattr(model, "generate"):
            generated = model.generate(
                input=str(audio_path),
                granularity="utterance",
                extract_embedding=True,
            )
            prediction = generated[0] if isinstance(generated, list) and generated else generated
        else:
            prediction = model.predict(audio_path)
        if isinstance(prediction, dict):
            return (
                list(prediction.get("labels", [])),
                [float(value) for value in prediction.get("scores", [])],
                prediction.get("embedding", []),
            )
        labels, scores, embedding = prediction
        return list(labels), [float(value) for value in scores], embedding

    def analyze(self, audio_path: Path | str) -> EmotionResult:
        path = Path(audio_path)
        labels, scores, embedding = self._prediction(self._get_model(), path)
        if not labels or not scores or len(labels) != len(scores):
            raise ValueError("Emotion model returned an invalid prediction")
        best_index = max(range(len(scores)), key=scores.__getitem__)
        confidence = float(scores[best_index])
        raw_label = _map_label(labels[best_index])
        label = raw_label if confidence >= self.confidence_threshold else EmotionLabel.OTHER
        mapped_scores: dict[EmotionLabel, float] = {}
        for raw, score in zip(labels, scores):
            mapped_scores[_map_label(raw)] = float(score)

        self.embedding_root.mkdir(parents=True, exist_ok=True)
        identity = hashlib.sha256(str(path).encode()).hexdigest()[:24]
        embedding_path = self.embedding_root / f"{identity}.json"
        embedding_path.write_text(json.dumps(embedding), encoding="utf-8")
        return EmotionResult(label, confidence, mapped_scores, embedding_path)
