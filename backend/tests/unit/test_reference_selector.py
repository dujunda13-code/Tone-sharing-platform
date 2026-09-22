from pathlib import Path

import pytest

from backend.app.schemas.common import EmotionLabel
from backend.app.services.disentanglement import (
    ReferenceCandidate,
    ReferenceControl,
    ReferenceSelector,
)


def _embedding(first: float, second: float = 0.0) -> tuple[float, ...]:
    return (first, second) + (0.0,) * 190


def _candidate(
    segment_id: str,
    embedding: tuple[float, ...],
    label: EmotionLabel,
    *,
    duration_seconds: float = 6.0,
    audio_path: Path | None = None,
    text: str | None = None,
) -> ReferenceCandidate:
    return ReferenceCandidate(
        profile_id="profile-a",
        segment_id=segment_id,
        audio_path=audio_path or Path(f"data/datasets/profile-a/{segment_id}.wav"),
        text=text if text is not None else f"text for {segment_id}",
        duration_seconds=duration_seconds,
        timbre_embedding=embedding,
        emotion_label=label,
        emotion_strength=0.8,
        f0_mean_hz=200.0,
        energy_mean=0.7,
    )


def test_selector_locks_timbre_center_before_matching_emotion_and_prosody():
    selector = ReferenceSelector(
        [
            _candidate("near-neutral", _embedding(1.0), EmotionLabel.NEUTRAL),
            _candidate("near-happy", _embedding(0.99, 0.1), EmotionLabel.HAPPY),
            _candidate("outlier-happy", _embedding(-1.0), EmotionLabel.HAPPY),
        ],
        center_candidate_limit=2,
    )

    selected = selector.select(
        "profile-a",
        ReferenceControl(
            emotion_label=EmotionLabel.HAPPY,
            emotion_strength=0.8,
            f0_mean_hz=200.0,
            energy_mean=0.7,
        ),
    )

    assert selected.segment_id == "near-happy"
    assert selected.audio_path.name == "near-happy.wav"
    assert selected.text == "text for near-happy"


def test_selector_rejects_profiles_without_a_five_to_ten_second_reference():
    selector = ReferenceSelector(
        [_candidate("too-short", _embedding(1.0), EmotionLabel.HAPPY, duration_seconds=4.9)]
    )

    with pytest.raises(ValueError, match="5 to 10 seconds"):
        selector.select("profile-a", ReferenceControl(emotion_label=EmotionLabel.HAPPY))


def test_reference_candidate_rejects_audio_outside_local_data():
    with pytest.raises(ValueError, match="local data/"):
        _candidate(
            "outside-data",
            _embedding(1.0),
            EmotionLabel.HAPPY,
            audio_path=Path("outside/reference.wav"),
        )


def test_reference_candidate_rejects_empty_transcript():
    with pytest.raises(ValueError, match="transcript"):
        _candidate("empty-text", _embedding(1.0), EmotionLabel.HAPPY, text=" \t")


def test_selector_rejects_duplicate_segment_ids_within_a_profile():
    duplicate = _candidate("duplicate", _embedding(1.0), EmotionLabel.HAPPY)

    with pytest.raises(ValueError, match="duplicate reference segment"):
        ReferenceSelector([duplicate, duplicate])
