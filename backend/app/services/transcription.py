"""Local FunASR transcription adapter (Phase D1).

The adapter is strictly local: it only accepts a prepared model directory
under the local ``models/`` root and never resolves a hub model name at
runtime, so a missing model fails closed instead of triggering a network
download. Empty transcripts are rejected so untrainable segments cannot be
persisted silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

DEFAULT_MODEL_DIR = Path("models") / "funasr" / "paraformer-zh"


class TranscriptionModelUnavailable(ValueError):
    """Raised when the local transcription model directory is missing or outside models/."""


class EmptyTranscriptionError(ValueError):
    """Raised when the local model returns a blank transcript."""


def _default_runner(model_dir: Path, wav_path: Path, language: str) -> str:
    from funasr import AutoModel

    model = AutoModel(model=str(model_dir), device="cpu", disable_update=True)
    result = model.generate(input=str(wav_path), language=language)
    texts: list[str] = []
    for item in result or []:
        text = item.get("text") if isinstance(item, dict) else None
        if text:
            texts.append(str(text))
    return "".join(texts)


class LocalFunASRTranscriber:
    """Transcribe one local WAV segment with the pinned local FunASR model."""

    def __init__(
        self,
        model_dir: Path | str | None = None,
        *,
        models_root: Path | str = Path("models"),
        runner: Callable[[Path, Path, str], str] | None = None,
    ) -> None:
        self.model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
        self.models_root = Path(models_root).expanduser().resolve()
        self._runner = runner or _default_runner

    def available(self) -> bool:
        try:
            self._resolve_model_dir()
        except TranscriptionModelUnavailable:
            return False
        return True

    def transcribe(self, wav_path: Path | str, language: str = "zh") -> str:
        model_dir = self._resolve_model_dir()
        text = self._runner(model_dir, Path(wav_path), language).strip()
        if not text:
            raise EmptyTranscriptionError("本地转写结果为空")
        return text

    def _resolve_model_dir(self) -> Path:
        resolved = self.model_dir.expanduser().resolve()
        if not resolved.is_dir():
            raise TranscriptionModelUnavailable(
                "本地转写模型目录不存在，请先在部署阶段下载到 models/"
            )
        if resolved != self.models_root and self.models_root not in resolved.parents:
            raise TranscriptionModelUnavailable(
                "本地转写模型目录必须位于 models/ 之内"
            )
        return resolved
