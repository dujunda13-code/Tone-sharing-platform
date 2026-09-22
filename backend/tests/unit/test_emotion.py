from pathlib import Path

import pytest

from backend.app.schemas.common import EmotionLabel
from backend.app.services import emotion as emotion_module
from backend.app.services.emotion import EmotionAnalyzer


class EmotionModelStub:
    def __init__(self):
        self.labels = ["neutral"]
        self.scores = [1.0]
        self.embedding = [0.1, 0.2, 0.3]

    def result(self, labels, scores):
        self.labels = labels
        self.scores = scores

    def predict(self, path: Path):
        return self.labels, self.scores, self.embedding


def test_low_confidence_maps_to_other(tmp_path: Path):
    model_stub = EmotionModelStub()
    model_stub.result(labels=["生气/angry"], scores=[0.54])
    analyzer = EmotionAnalyzer(
        model_loader=lambda: model_stub,
        embedding_root=tmp_path,
    )

    result = analyzer.analyze(Path("sample.wav"))

    assert result.label is EmotionLabel.OTHER
    assert result.confidence == 0.54
    assert result.scores[EmotionLabel.ANGRY] == 0.54
    assert result.embedding_path.is_file()


def test_analyzer_loads_model_once(tmp_path: Path):
    model_stub = EmotionModelStub()
    load_count = 0

    def load_model():
        nonlocal load_count
        load_count += 1
        return model_stub

    analyzer = EmotionAnalyzer(model_loader=load_model, embedding_root=tmp_path)
    analyzer.analyze(Path("first.wav"))
    analyzer.analyze(Path("second.wav"))

    assert load_count == 1


def test_emotion_analyzer_available_requires_all_local_runtime_artifacts(
    tmp_path: Path, monkeypatch
):
    model_dir = tmp_path / "models" / "emotion2vec"
    monkeypatch.setattr(emotion_module, "DEFAULT_EMOTION_MODEL_DIR", model_dir)
    analyzer = EmotionAnalyzer()

    assert analyzer.available() is False

    model_dir.mkdir(parents=True)
    (model_dir / "config.yaml").write_text("config", encoding="utf-8")
    (model_dir / "model.pt").write_bytes(b"weights")
    assert analyzer.available() is False

    (model_dir / "tokens.txt").write_text("tokens", encoding="utf-8")
    assert analyzer.available() is True


def test_analyzer_consumes_local_funasr_generate_result(tmp_path: Path):
    """A deployed FunASR model exposes generate(), not the old hub-only predict() shim."""

    captured: dict[str, object] = {}

    class LocalFunASRModel:
        def generate(self, *, input: str, **kwargs):
            captured["input"] = input
            captured.update(kwargs)
            return [
                {
                    "labels": ["happy"],
                    "scores": [0.91],
                    "embedding": [[0.1, 0.2, 0.3]],
                }
            ]

    audio_path = tmp_path / "segment.wav"
    analyzer = EmotionAnalyzer(model_loader=LocalFunASRModel, embedding_root=tmp_path)

    result = analyzer.analyze(audio_path)

    assert result.label is EmotionLabel.HAPPY
    assert result.confidence == 0.91
    assert captured["input"] == str(audio_path)
    assert captured["granularity"] == "utterance"
    assert captured["extract_embedding"] is True


def test_default_model_loader_targets_local_model_directory(monkeypatch, tmp_path):
    """The preprocessing analyzer must load the deployed local model, never a hub name.

    Runtime code is forbidden from resolving model names online (Phase D3), so
    the default FunASR loader has to point at models/emotion2vec on this host
    and must fail closed when that directory has not been deployed.
    """
    import sys
    import types

    captured: dict[str, object] = {}

    class StubAutoModel:
        def __new__(cls, **kwargs):
            captured.update(kwargs)
            return object()

    model_dir = tmp_path / "models" / "emotion2vec"
    monkeypatch.setattr(emotion_module, "DEFAULT_EMOTION_MODEL_DIR", model_dir)
    monkeypatch.setitem(sys.modules, "funasr", types.SimpleNamespace(AutoModel=StubAutoModel))

    with pytest.raises(RuntimeError, match="models/emotion2vec"):
        emotion_module._default_model_loader()

    model_dir.mkdir(parents=True)
    (model_dir / "config.yaml").write_text("config", encoding="utf-8")
    (model_dir / "model.pt").write_bytes(b"weights")
    (model_dir / "tokens.txt").write_text("tokens", encoding="utf-8")
    emotion_module._default_model_loader()

    resolved = str(captured.get("model", "")).replace("\\", "/")
    assert resolved.endswith("models/emotion2vec")
    assert "iic/" not in resolved
    assert captured.get("device") == "cpu"
