import torch

from backend.app.ml.losses import compute_disentanglement_losses


def test_total_loss_has_all_required_finite_terms():
    losses = compute_disentanglement_losses(
        timbre=torch.randn(2, 192),
        timbre_target=torch.randn(2, 192),
        prosody=torch.randn(2, 120, 192),
        prosody_target=torch.randn(2, 120, 192),
        emotion=torch.randn(2, 1024),
        emotion_target=torch.randn(2, 1024),
        discriminator_logits=torch.randn(2),
        same_pair_targets=torch.tensor([1.0, 0.0]),
    )

    assert set(losses) == {
        "adversarial",
        "timbre_consistency",
        "prosody",
        "emotion",
        "xcov",
    }
    assert all(term.ndim == 0 and torch.isfinite(term) for term in losses.values())


def test_cross_covariance_uses_the_latent_prosody_condition_when_provided():
    losses = compute_disentanglement_losses(
        timbre=torch.randn(1, 192),
        timbre_target=torch.randn(1, 192),
        prosody=torch.randn(1, 12, 4),
        prosody_target=torch.randn(1, 12, 4),
        emotion=torch.randn(1, 1024),
        emotion_target=torch.randn(1, 1024),
        discriminator_logits=torch.randn(2),
        same_pair_targets=torch.tensor([1.0, 0.0]),
        xcov_timbre=torch.randn(2, 192),
        xcov_prosody=torch.randn(2, 192),
    )

    assert losses["xcov"].ndim == 0
    assert torch.isfinite(losses["xcov"])
