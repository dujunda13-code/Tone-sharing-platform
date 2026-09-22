from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FeatureBatch:
    """Frozen local feature tensors used by the dual-branch adapter.

    ``timbre`` is the CAM++ 192-dimensional embedding, ``prosody`` contains
    frame-level log-F0/energy/voiced/duration plus the 1024-dimensional
    Emotion2Vec feature, and ``emotion`` retains that source embedding as the
    reconstruction target.  Segment IDs are unique inside a batch so an
    exchange pair cannot accidentally use the same source segment twice.
    """

    timbre: Tensor
    prosody: Tensor
    emotion: Tensor
    speaker_ids: tuple[str, ...]
    segment_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        batch_size = self.timbre.shape[0] if self.timbre.ndim >= 1 else 0
        if self.timbre.ndim != 2 or self.timbre.shape[-1] != 192:
            raise ValueError("timbre features must have shape [batch, 192]")
        if self.prosody.ndim != 3 or self.prosody.shape[-1] != 1028:
            raise ValueError("prosody features must have shape [batch, frames, 1028]")
        if self.prosody.shape[0] != batch_size or self.prosody.shape[1] < 1:
            raise ValueError("prosody features must share a non-empty batch with timbre")
        if self.emotion.ndim != 2 or self.emotion.shape != (batch_size, 1024):
            raise ValueError("emotion features must have shape [batch, 1024]")
        if len(self.speaker_ids) != batch_size or len(self.segment_ids) != batch_size:
            raise ValueError("speaker and segment identifiers must match the feature batch")
        if any(not identifier for identifier in (*self.speaker_ids, *self.segment_ids)):
            raise ValueError("speaker and segment identifiers must be non-empty")
        if len(set(self.segment_ids)) != batch_size:
            raise ValueError("segment identifiers must be unique within a feature batch")
        if self.timbre.device != self.prosody.device or self.timbre.device != self.emotion.device:
            raise ValueError("all feature tensors must be on the same device")

    def to(self, device: torch.device | str) -> FeatureBatch:
        """Move tensors without altering their local-only audit identifiers."""
        return FeatureBatch(
            timbre=self.timbre.to(device),
            prosody=self.prosody.to(device),
            emotion=self.emotion.to(device),
            speaker_ids=self.speaker_ids,
            segment_ids=self.segment_ids,
        )


@dataclass(frozen=True)
class SerialExchangePair:
    """Two different segments of one speaker, each kept at micro-batch size one."""

    source: FeatureBatch
    partner: FeatureBatch

    def __post_init__(self) -> None:
        if self.source.timbre.shape[0] != 1 or self.partner.timbre.shape[0] != 1:
            raise ValueError("serial exchange pairs require batch_size=1 feature batches")
        if self.source.speaker_ids[0] != self.partner.speaker_ids[0]:
            raise ValueError("serial exchange pairs must use the same speaker")
        if self.source.segment_ids[0] == self.partner.segment_ids[0]:
            raise ValueError("serial exchange pairs must use different segments")


def exchange_partner_indices(batch: FeatureBatch) -> Tensor:
    """Return deterministic same-speaker, different-segment exchange partners."""
    members_by_speaker: dict[str, list[int]] = {}
    for index, speaker_id in enumerate(batch.speaker_ids):
        members_by_speaker.setdefault(speaker_id, []).append(index)

    partners = [0] * len(batch.speaker_ids)
    for speaker_id, members in members_by_speaker.items():
        if len(members) < 2:
            raise ValueError(
                f"speaker {speaker_id!r} must provide at least two segments for exchange pairing"
            )
        for offset, index in enumerate(members):
            partners[index] = members[(offset + 1) % len(members)]

    return torch.tensor(partners, dtype=torch.long, device=batch.timbre.device)


def serial_exchange_pairs(manifest: Iterable[FeatureBatch]) -> tuple[SerialExchangePair, ...]:
    """Build deterministic pairs while never placing more than one item on GPU.

    The 8 GB runtime uses a micro-batch size of one.  The partner is therefore
    represented as a second serial model call rather than by concatenating two
    segments into a GPU batch.
    """
    batches = tuple(manifest)
    if not batches:
        raise ValueError("feature manifest must contain at least one segment")
    if any(not isinstance(batch, FeatureBatch) for batch in batches):
        raise TypeError("feature manifest entries must be FeatureBatch instances")
    if any(batch.timbre.shape[0] != 1 for batch in batches):
        raise ValueError("serial disentangler training requires batch_size=1 feature batches")
    if any(batch.timbre.device.type != "cpu" for batch in batches):
        raise ValueError("feature batches must remain on CPU before serial CUDA training")

    segment_ids = tuple(batch.segment_ids[0] for batch in batches)
    if len(set(segment_ids)) != len(segment_ids):
        raise ValueError("feature manifest contains duplicate segment identifiers")

    members_by_speaker: dict[str, list[FeatureBatch]] = {}
    for batch in batches:
        members_by_speaker.setdefault(batch.speaker_ids[0], []).append(batch)

    partners_by_segment: dict[str, FeatureBatch] = {}
    for speaker_id, members in members_by_speaker.items():
        if len(members) < 2:
            raise ValueError(
                f"speaker {speaker_id!r} must provide at least two segments for exchange pairing"
            )
        for offset, batch in enumerate(members):
            partners_by_segment[batch.segment_ids[0]] = members[(offset + 1) % len(members)]

    return tuple(
        SerialExchangePair(source=batch, partner=partners_by_segment[batch.segment_ids[0]])
        for batch in batches
    )
