"""Local speaker and emotion embedding adapters (Phase D2).

Both embedders are strictly local: they only accept a prepared model
directory under the local ``models/`` root, load it lazily through FunASR,
and fail closed when the directory is missing. They never resolve a hub
model name at runtime, so a missing deployment stays offline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

DEFAULT_CAMPPLUS_DIR = Path("models") / "campplus"
DEFAULT_EMOTION_DIR = Path("models") / "emotion2vec"


class EmbeddingModelUnavailable(ValueError):
    """Raised when a local embedding model directory is missing or outside models/."""


def _require_local_model_dir(model_dir: Path, models_root: Path) -> Path:
    resolved = model_dir.expanduser().resolve()
    if not resolved.is_dir():
        raise EmbeddingModelUnavailable(
            "本地嵌入模型目录不存在，请先在部署阶段下载到 models/"
        )
    if resolved != models_root and models_root not in resolved.parents:
        raise EmbeddingModelUnavailable("本地嵌入模型目录必须位于 models/ 之内")
    return resolved


class _LocalFunASREmbedder:
    """Shared lazy FunASR loader for one local model directory."""

    def __init__(
        self,
        *,
        model_dir: Path | str,
        models_root: Path | str,
        loader: Callable[[Path], Any] | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.models_root = Path(models_root).expanduser().resolve()
        self._loader = loader
        self._model: Any | None = None

    def _require_dir(self) -> Path:
        return _require_local_model_dir(self.model_dir, self.models_root)

    def available(self) -> bool:
        try:
            self._require_dir()
        except EmbeddingModelUnavailable:
            return False
        return True

    def _get_model(self) -> Any:
        if self._model is None:
            model_dir = self._require_dir()
            if self._loader is not None:
                self._model = self._loader(model_dir)
            else:
                from funasr import AutoModel

                self._model = AutoModel(
                    model=str(model_dir), device="cpu", disable_update=True
                )
        return self._model

    def embed(self, wav_path: Path | str) -> list[float]:
        model = self._get_model()
        if hasattr(model, "generate"):
            prediction = model.generate(
                input=str(Path(wav_path)),
                granularity="utterance",
                extract_embedding=True,
            )
        else:
            prediction = model.predict(Path(wav_path))
        items = prediction if isinstance(prediction, list) else [prediction]
        for item in items:
            if not isinstance(item, dict):
                continue
            embedding = item.get("embedding")
            if embedding is None:
                embedding = item.get("spk_embedding")
            if embedding is None:
                continue
            if hasattr(embedding, "detach"):
                embedding = embedding.detach().cpu().reshape(-1).tolist()
            elif (
                isinstance(embedding, (list, tuple))
                and len(embedding) == 1
                and isinstance(embedding[0], (list, tuple))
            ):
                embedding = embedding[0]
            values = [float(value) for value in embedding]
            if values:
                return values
        raise EmbeddingModelUnavailable("本地嵌入模型没有返回可用的 embedding")


class CamPlusEmbedder(_LocalFunASREmbedder):
    """192-dimensional frozen CAM++ speaker embedding from a local model dir."""

    DIMENSION = 192

    def __init__(
        self,
        model_dir: Path | str = DEFAULT_CAMPPLUS_DIR,
        *,
        models_root: Path | str = Path("models"),
        loader: Callable[[Path], Any] | None = None,
    ) -> None:
        super().__init__(model_dir=model_dir, models_root=models_root, loader=loader)

    def embed(self, wav_path: Path | str) -> list[float]:
        values = super().embed(wav_path)
        if len(values) != self.DIMENSION:
            raise EmbeddingModelUnavailable(
                f"CAM++ embedding must contain {self.DIMENSION} values, got {len(values)}"
            )
        return values


class EmotionEmbedder(_LocalFunASREmbedder):
    """1024-dimensional frozen Emotion2Vec embedding from a local model dir."""

    DIMENSION = 1024

    def __init__(
        self,
        model_dir: Path | str = DEFAULT_EMOTION_DIR,
        *,
        models_root: Path | str = Path("models"),
        loader: Callable[[Path], Any] | None = None,
    ) -> None:
        super().__init__(model_dir=model_dir, models_root=models_root, loader=loader)

    def embed(self, wav_path: Path | str) -> list[float]:
        values = super().embed(wav_path)
        if len(values) != self.DIMENSION:
            raise EmbeddingModelUnavailable(
                f"Emotion2Vec embedding must contain {self.DIMENSION} values, got {len(values)}"
            )
        return values


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise EmbeddingModelUnavailable("embedding dimensions do not match")
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        raise EmbeddingModelUnavailable("cannot compare zero embeddings")
    return float(numerator / (left_norm * right_norm))
