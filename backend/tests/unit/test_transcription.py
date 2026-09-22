"""Local FunASR transcription adapter contract (Phase D1).

The adapter must only accept prepared local model directories under models/;
it must never resolve a model by hub name at runtime, and empty transcripts
must fail closed instead of producing untrainable segments.
"""

from pathlib import Path

import pytest

from backend.app.services.transcription import (
    EmptyTranscriptionError,
    LocalFunASRTranscriber,
    TranscriptionModelUnavailable,
)


def _local_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "models" / "funasr" / "paraformer-zh"
    model_dir.mkdir(parents=True)
    return model_dir


def test_missing_local_model_directory_fails_closed(tmp_path):
    transcriber = LocalFunASRTranscriber(
        model_dir=tmp_path / "models" / "funasr" / "paraformer-zh",
        models_root=tmp_path / "models",
    )

    assert transcriber.available() is False
    with pytest.raises(TranscriptionModelUnavailable):
        transcriber.transcribe(tmp_path / "seg.wav")


def test_model_directory_must_stay_under_local_models_root(tmp_path):
    outside = tmp_path / "elsewhere" / "paraformer-zh"
    outside.mkdir(parents=True)
    transcriber = LocalFunASRTranscriber(
        model_dir=outside, models_root=tmp_path / "models"
    )

    with pytest.raises(TranscriptionModelUnavailable):
        transcriber.transcribe(tmp_path / "seg.wav")


def test_transcribe_uses_local_model_directory_and_injected_runner(tmp_path):
    model_dir = _local_model_dir(tmp_path)
    seen: dict[str, object] = {}

    def fake_runner(model_dir_arg, wav_path, language):
        seen["model_dir"] = Path(model_dir_arg)
        seen["wav"] = Path(wav_path)
        seen["language"] = language
        return "  今天天气很好。  "

    transcriber = LocalFunASRTranscriber(
        model_dir=model_dir, models_root=tmp_path / "models", runner=fake_runner
    )
    wav = tmp_path / "seg_0001.wav"
    wav.write_bytes(b"RIFF")

    assert transcriber.available() is True
    assert transcriber.transcribe(wav, language="zh") == "今天天气很好。"
    assert seen["model_dir"] == model_dir
    assert seen["wav"] == wav
    assert seen["language"] == "zh"


def test_blank_transcription_is_rejected(tmp_path):
    transcriber = LocalFunASRTranscriber(
        model_dir=_local_model_dir(tmp_path),
        models_root=tmp_path / "models",
        runner=lambda *_args, **_kwargs: "   ",
    )

    with pytest.raises(EmptyTranscriptionError):
        transcriber.transcribe(tmp_path / "seg.wav")
