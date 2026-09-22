"""Local embedding adapter contract (Phase D2/D3).

The embedders only accept prepared directories under models/, fail closed
when the deployment is missing, validate the frozen dimensions, and expose a
local cosine helper. FunASR itself is injected so no weights are needed here.
"""

from pathlib import Path

import pytest

from backend.app.services.speaker_embedding import (
    CamPlusEmbedder,
    EmbeddingModelUnavailable,
    EmotionEmbedder,
    cosine_similarity,
)


def _campplus_loader(model_dir: Path):
    def loader(_dir: Path):
        class Stub:
            def predict(self, wav):
                return [{"embedding": [0.5] * 192}]

        return Stub()

    return loader


def test_embedders_fail_closed_without_local_model_dir(tmp_path):
    campplus = CamPlusEmbedder(
        tmp_path / "models" / "campplus", models_root=tmp_path / "models"
    )
    emotion = EmotionEmbedder(
        tmp_path / "models" / "emotion2vec", models_root=tmp_path / "models"
    )

    assert campplus.available() is False
    assert emotion.available() is False
    with pytest.raises(EmbeddingModelUnavailable):
        campplus.embed(tmp_path / "seg.wav")
    with pytest.raises(EmbeddingModelUnavailable):
        emotion.embed(tmp_path / "seg.wav")


def test_embedder_model_dir_must_stay_under_models_root(tmp_path):
    outside = tmp_path / "elsewhere" / "campplus"
    outside.mkdir(parents=True)
    embedder = CamPlusEmbedder(outside, models_root=tmp_path / "models")

    assert embedder.available() is False
    with pytest.raises(EmbeddingModelUnavailable):
        embedder.embed(tmp_path / "seg.wav")


def test_campplus_embedding_requires_frozen_dimension(tmp_path):
    model_dir = tmp_path / "models" / "campplus"
    model_dir.mkdir(parents=True)

    def wrong_dimension_loader(_dir: Path):
        class Stub:
            def predict(self, wav):
                return [{"embedding": [0.5] * 191}]

        return Stub()

    embedder = CamPlusEmbedder(
        model_dir, models_root=tmp_path / "models", loader=wrong_dimension_loader
    )

    with pytest.raises(EmbeddingModelUnavailable, match="192"):
        embedder.embed(tmp_path / "seg.wav")


def test_campplus_embedding_consumes_local_funasr_generate_result(tmp_path):
    """The local CAM++ adapter must use FunASR's deployed generate() API."""
    import torch

    model_dir = tmp_path / "models" / "campplus"
    model_dir.mkdir(parents=True)
    captured: dict[str, object] = {}

    class GenerateOnlyModel:
        def generate(self, *, input: str, **kwargs):
            captured["input"] = input
            captured.update(kwargs)
            return [{"spk_embedding": torch.tensor([[0.5] * 192])}]

    embedder = CamPlusEmbedder(
        model_dir,
        models_root=tmp_path / "models",
        loader=lambda _dir: GenerateOnlyModel(),
    )
    wav_path = tmp_path / "segment.wav"

    assert len(embedder.embed(wav_path)) == 192
    assert captured["input"] == str(wav_path)
    assert captured["granularity"] == "utterance"
    assert captured["extract_embedding"] is True


def test_emotion_embedding_accepts_nested_embedding_lists(tmp_path):
    model_dir = tmp_path / "models" / "emotion2vec"
    model_dir.mkdir(parents=True)

    def nested_loader(_dir: Path):
        class Stub:
            def predict(self, wav):
                return [{"embedding": [[0.25] * 1024]}]

        return Stub()

    embedder = EmotionEmbedder(
        model_dir, models_root=tmp_path / "models", loader=nested_loader
    )

    values = embedder.embed(tmp_path / "seg.wav")

    assert len(values) == 1024
    assert all(value == 0.25 for value in values)


def test_cosine_similarity_computes_local_alignment():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 2.0]) == pytest.approx(0.0)
    with pytest.raises(EmbeddingModelUnavailable):
        cosine_similarity([], [])
