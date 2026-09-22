import pytest
import torch

from backend.app.ml.datasets import FeatureBatch, exchange_partner_indices, serial_exchange_pairs


def _batch(
    speaker_ids: tuple[str, ...],
    segment_ids: tuple[str, ...],
    *,
    device: torch.device | str | None = None,
) -> FeatureBatch:
    batch_size = len(speaker_ids)
    return FeatureBatch(
        timbre=torch.randn(batch_size, 192, device=device),
        prosody=torch.randn(batch_size, 12, 1028, device=device),
        emotion=torch.randn(batch_size, 1024, device=device),
        speaker_ids=speaker_ids,
        segment_ids=segment_ids,
    )


def test_exchange_pairs_use_a_different_segment_from_the_same_speaker():
    batch = _batch(("speaker-a", "speaker-a", "speaker-b", "speaker-b"), ("a1", "a2", "b1", "b2"))

    partners = exchange_partner_indices(batch)

    assert partners.tolist() == [1, 0, 3, 2]
    for index, partner in enumerate(partners.tolist()):
        assert batch.speaker_ids[index] == batch.speaker_ids[partner]
        assert batch.segment_ids[index] != batch.segment_ids[partner]


def test_exchange_pairs_reject_a_speaker_without_another_segment():
    batch = _batch(("speaker-a", "speaker-b", "speaker-b"), ("a1", "b1", "b2"))

    with pytest.raises(ValueError, match="at least two segments"):
        exchange_partner_indices(batch)


def test_serial_exchange_pairs_keep_each_cuda_micro_batch_at_one():
    segments = [
        _batch(("speaker-a",), ("a1",)),
        _batch(("speaker-a",), ("a2",)),
        _batch(("speaker-b",), ("b1",)),
        _batch(("speaker-b",), ("b2",)),
    ]

    pairs = serial_exchange_pairs(segments)

    assert [
        (pair.source.segment_ids[0], pair.partner.segment_ids[0]) for pair in pairs
    ] == [("a1", "a2"), ("a2", "a1"), ("b1", "b2"), ("b2", "b1")]
    assert all(pair.source.timbre.shape[0] == pair.partner.timbre.shape[0] == 1 for pair in pairs)


def test_serial_exchange_pairs_reject_features_preloaded_on_an_accelerator():
    segments = [
        _batch(("speaker-a",), ("a1",), device="meta"),
        _batch(("speaker-a",), ("a2",), device="meta"),
    ]

    with pytest.raises(ValueError, match="remain on CPU"):
        serial_exchange_pairs(segments)
