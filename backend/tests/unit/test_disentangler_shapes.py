import pytest
import torch
from torch.nn import functional as F

from backend.app.ml.datasets import FeatureBatch
from backend.app.ml.disentangler import DualBranchDisentangler, TimbreEncoder, gradient_reverse


def test_forward_contract_preserves_timbre_and_projects_prosody():
    model = DualBranchDisentangler()
    timbre_features = torch.randn(2, 192)
    prosody_features = torch.randn(2, 120, 1028)

    output = model(timbre_features, prosody_features)

    assert output.timbre.shape == (2, 192)
    assert output.prosody.shape == (2, 120, 192)
    assert torch.allclose(output.timbre.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_forward_accepts_feature_batch_and_decodes_emotion_from_prosody():
    model = DualBranchDisentangler()
    batch = FeatureBatch(
        timbre=torch.randn(2, 192),
        prosody=torch.randn(2, 12, 1028),
        emotion=torch.randn(2, 1024),
        speaker_ids=("speaker-a", "speaker-a"),
        segment_ids=("segment-a", "segment-b"),
    )

    output = model(batch)

    assert output.timbre.shape == (2, 192)
    assert output.prosody.shape == (2, 12, 192)
    assert model.reconstruct_frame_prosody(output.prosody).shape == (2, 12, 4)
    assert model.predict_emotion(output.prosody).shape == (2, 1024)


def test_gradient_reversal_changes_sign():
    value = torch.tensor([1.0], requires_grad=True)

    gradient_reverse(value, 0.2).sum().backward()

    assert value.grad is not None
    assert value.grad.item() == pytest.approx(-0.2)


def test_timbre_consistency_has_a_trainable_adapter_path():
    encoder = TimbreEncoder()
    output = encoder(torch.randn(2, 192))
    target = torch.randn(2, 192)

    F.mse_loss(output, target).backward()

    parameters = tuple(encoder.parameters())
    assert parameters
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters)
