from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.autograd import Function
from torch.nn import functional as F

from backend.app.ml.datasets import FeatureBatch


class _GradientReverse(Function):
    @staticmethod
    def forward(ctx: object, value: Tensor, scale: float) -> Tensor:
        ctx.scale = scale
        return value.view_as(value)

    @staticmethod
    def backward(ctx: object, gradient: Tensor) -> tuple[Tensor, None]:
        return gradient.neg().mul(ctx.scale), None


def gradient_reverse(value: Tensor, scale: float) -> Tensor:
    """Pass values through while multiplying their backward gradient by -scale."""
    if scale < 0:
        raise ValueError("gradient reversal scale must be non-negative")
    return _GradientReverse.apply(value, scale)


@dataclass(frozen=True)
class DisentangledCondition:
    timbre: Tensor
    prosody: Tensor


class TimbreEncoder(nn.Module):
    """Adapt frozen CAM++ inputs while retaining a normalized 192-D condition."""

    def __init__(self) -> None:
        super().__init__()
        self.adapter = nn.Linear(192, 192, bias=False)
        nn.init.eye_(self.adapter.weight)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[-1] != 192:
            raise ValueError("timbre features must have shape [batch, 192]")
        return F.normalize(self.adapter(features), dim=-1)


class ProsodyEncoder(nn.Module):
    """Project frame-level 1028-dimensional prosody inputs to 192 dimensions."""

    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=1028,
            hidden_size=96,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.projection = nn.Linear(192, 192)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[-1] != 1028:
            raise ValueError("prosody features must have shape [batch, frames, 1028]")
        encoded, _ = self.gru(features)
        return self.projection(encoded)


class IndependenceDiscriminator(nn.Module):
    """Predict matched timbre/prosody pairs while reversing the prosody gradient."""

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 1),
        )

    def forward(self, timbre: Tensor, prosody: Tensor, reversal_scale: float = 1.0) -> Tensor:
        if timbre.ndim != 2 or timbre.shape[-1] != 192:
            raise ValueError("timbre condition must have shape [batch, 192]")
        if prosody.ndim != 3 or prosody.shape[:1] != timbre.shape[:1] or prosody.shape[-1] != 192:
            raise ValueError("prosody condition must have shape [batch, frames, 192]")
        pooled_prosody = gradient_reverse(prosody.mean(dim=1), reversal_scale)
        return self.classifier(torch.cat((timbre, pooled_prosody), dim=-1)).squeeze(-1)


class DualBranchDisentangler(nn.Module):
    """Produce independent timbre and prosody conditions for later GPU-worker stages."""

    def __init__(self) -> None:
        super().__init__()
        self.timbre_encoder = TimbreEncoder()
        self.prosody_encoder = ProsodyEncoder()
        self.frame_prosody_decoder = nn.Linear(192, 4)
        self.emotion_decoder = nn.Linear(192, 1024)
        self.independence_discriminator = IndependenceDiscriminator()

    def forward(
        self,
        batch_or_timbre: FeatureBatch | Tensor,
        prosody_features: Tensor | None = None,
    ) -> DisentangledCondition:
        """Accept the auditable feature batch or explicit tensors for unit-level use."""
        if isinstance(batch_or_timbre, FeatureBatch):
            if prosody_features is not None:
                raise ValueError("FeatureBatch forward calls must not include separate prosody features")
            timbre_features = batch_or_timbre.timbre
            prosody_features = batch_or_timbre.prosody
        else:
            timbre_features = batch_or_timbre
            if prosody_features is None:
                raise ValueError("explicit timbre features require prosody features")

        return DisentangledCondition(
            timbre=self.timbre_encoder(timbre_features),
            prosody=self.prosody_encoder(prosody_features),
        )

    def predict_emotion(self, prosody: Tensor) -> Tensor:
        """Reconstruct Emotion2Vec targets from the temporal prosody condition."""
        if prosody.ndim != 3 or prosody.shape[-1] != 192:
            raise ValueError("prosody condition must have shape [batch, frames, 192]")
        return self.emotion_decoder(prosody.mean(dim=1))

    def reconstruct_frame_prosody(self, prosody: Tensor) -> Tensor:
        """Reconstruct log-F0, energy, voiced state, and duration per frame."""
        if prosody.ndim != 3 or prosody.shape[-1] != 192:
            raise ValueError("prosody condition must have shape [batch, frames, 192]")
        return self.frame_prosody_decoder(prosody)
