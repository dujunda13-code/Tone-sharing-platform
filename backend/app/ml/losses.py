from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor
from torch.nn import functional as F


def _cross_covariance_penalty(timbre: Tensor, prosody: Tensor) -> Tensor:
    if timbre.ndim != 2:
        raise ValueError("cross-covariance timbre must have shape [batch, dimensions]")
    if prosody.ndim == 3:
        pooled_prosody = prosody.mean(dim=1)
    elif prosody.ndim == 2:
        pooled_prosody = prosody
    else:
        raise ValueError("cross-covariance prosody must have shape [batch, frames, dimensions]")
    if pooled_prosody.shape[0] != timbre.shape[0]:
        raise ValueError("cross-covariance features must share a batch")
    timbre_centered = timbre - timbre.mean(dim=0, keepdim=True)
    prosody_centered = pooled_prosody - pooled_prosody.mean(dim=0, keepdim=True)
    covariance = timbre_centered.transpose(0, 1).matmul(prosody_centered)
    return covariance.pow(2).mean() / max(timbre.shape[0] - 1, 1)


def compute_disentanglement_losses(
    *,
    timbre: Tensor,
    timbre_target: Tensor,
    prosody: Tensor,
    prosody_target: Tensor,
    emotion: Tensor,
    emotion_target: Tensor,
    discriminator_logits: Tensor,
    same_pair_targets: Tensor,
    xcov_timbre: Tensor | None = None,
    xcov_prosody: Tensor | None = None,
) -> Mapping[str, Tensor]:
    """Return the complete, named objective terms for dual-branch training."""
    if timbre.shape != timbre_target.shape:
        raise ValueError("timbre target shape must match timbre")
    if prosody.shape != prosody_target.shape:
        raise ValueError("prosody target shape must match prosody")
    if emotion.shape != emotion_target.shape:
        raise ValueError("emotion target shape must match emotion")
    if discriminator_logits.shape != same_pair_targets.shape:
        raise ValueError("pair target shape must match discriminator logits")
    latent_timbre = xcov_timbre if xcov_timbre is not None else timbre
    latent_prosody = xcov_prosody if xcov_prosody is not None else prosody

    return {
        "adversarial": F.binary_cross_entropy_with_logits(discriminator_logits, same_pair_targets),
        "timbre_consistency": F.mse_loss(timbre, timbre_target),
        "prosody": F.mse_loss(prosody, prosody_target),
        "emotion": F.mse_loss(emotion, emotion_target),
        "xcov": _cross_covariance_penalty(latent_timbre, latent_prosody),
    }
