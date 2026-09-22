from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from math import sqrt
import os
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Iterable, Iterator
from uuid import uuid4

import torch

from backend.app.ml.datasets import FeatureBatch, SerialExchangePair, serial_exchange_pairs
from backend.app.ml.disentangler import DualBranchDisentangler
from backend.app.ml.losses import compute_disentanglement_losses
from backend.app.schemas.common import EmotionLabel


LOSS_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        "adversarial": 1.0,
        "timbre_consistency": 3.0,
        "prosody": 2.0,
        "emotion": 1.0,
        "xcov": 0.2,
    }
)
GRADIENT_REVERSAL_LAMBDA = 0.2
MINIMUM_VRAM_BYTES = 8000 * 1024**2
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCAL_ARTIFACT_ROOTS = (PROJECT_ROOT / "data", PROJECT_ROOT / "models")
GPU_TRAINING_LOCK = Lock()


def validate_adapter_output_dir(output_dir: Path | str) -> Path:
    """Keep adapter weights inside the local-only data or models roots."""
    target_dir = Path(output_dir).resolve()
    for root in LOCAL_ARTIFACT_ROOTS:
        try:
            target_dir.relative_to(root.resolve())
        except ValueError:
            continue
        return target_dir
    raise ValueError("adapter weights must be saved under local data/ or models/")


def _resolve_project_path(path: Path) -> Path:
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def atomic_save_adapter_payload(destination: Path | str, payload: Mapping[str, object]) -> Path:
    """Write an adapter payload beside its target, then atomically publish it."""
    target = Path(destination).resolve()
    validate_adapter_output_dir(target.parent)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid4().hex}.tmp"
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


@dataclass(frozen=True)
class ReferenceCandidate:
    profile_id: str
    segment_id: str
    audio_path: Path
    text: str
    duration_seconds: float
    timbre_embedding: tuple[float, ...]
    emotion_label: EmotionLabel
    emotion_strength: float
    f0_mean_hz: float
    energy_mean: float

    def __post_init__(self) -> None:
        if len(self.timbre_embedding) != 192:
            raise ValueError("reference timbre embedding must contain 192 values")
        if self.duration_seconds <= 0:
            raise ValueError("reference duration must be positive")
        if not 0.0 <= self.emotion_strength <= 1.0:
            raise ValueError("reference emotion strength must be between 0 and 1")
        try:
            _resolve_project_path(self.audio_path).relative_to((PROJECT_ROOT / "data").resolve())
        except ValueError as exc:
            raise ValueError("reference audio must stay under local data/") from exc
        if not self.text.strip():
            raise ValueError("reference transcript must not be empty")


@dataclass(frozen=True)
class ReferenceControl:
    emotion_label: EmotionLabel
    emotion_strength: float = 0.65
    f0_mean_hz: float | None = None
    energy_mean: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.emotion_strength <= 1.0:
            raise ValueError("emotion strength must be between 0 and 1")


@dataclass(frozen=True)
class SelectedReference:
    profile_id: str
    segment_id: str
    audio_path: Path
    text: str
    duration_seconds: float
    timbre_similarity: float
    emotion_label: EmotionLabel
    emotion_strength: float
    f0_mean_hz: float
    energy_mean: float


def _normalize(values: tuple[float, ...]) -> tuple[float, ...]:
    norm = sqrt(sum(value * value for value in values))
    if norm == 0:
        raise ValueError("reference timbre embedding must not be zero")
    return tuple(value / norm for value in values)


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


class ReferenceSelector:
    """Select a local 5–10 second reference without changing the base model."""

    def __init__(self, candidates: Iterable[ReferenceCandidate], *, center_candidate_limit: int = 5) -> None:
        if center_candidate_limit < 1:
            raise ValueError("center candidate limit must be positive")
        self._candidates = tuple(candidates)
        candidate_keys = tuple(
            (candidate.profile_id, candidate.segment_id) for candidate in self._candidates
        )
        if len(set(candidate_keys)) != len(candidate_keys):
            raise ValueError("duplicate reference segment within profile")
        self._center_candidate_limit = center_candidate_limit

    def select(self, profile_id: str, control: ReferenceControl) -> SelectedReference:
        eligible = tuple(
            candidate
            for candidate in self._candidates
            if candidate.profile_id == profile_id and 5.0 <= candidate.duration_seconds <= 10.0
        )
        if not eligible:
            raise ValueError("profile has no eligible 5 to 10 seconds reference")

        normalized = {candidate.segment_id: _normalize(candidate.timbre_embedding) for candidate in eligible}
        center_values = tuple(
            sum(vector[index] for vector in normalized.values()) / len(normalized)
            for index in range(192)
        )
        center = _normalize(center_values)
        closest = sorted(
            eligible,
            key=lambda candidate: (-_dot(normalized[candidate.segment_id], center), candidate.segment_id),
        )[: self._center_candidate_limit]

        def rank(candidate: ReferenceCandidate) -> tuple[float, float, float, float, float, str]:
            emotion_match = float(candidate.emotion_label == control.emotion_label)
            strength_distance = abs(candidate.emotion_strength - control.emotion_strength)
            f0_distance = (
                abs(candidate.f0_mean_hz - control.f0_mean_hz) / max(control.f0_mean_hz, 1.0)
                if control.f0_mean_hz is not None
                else 0.0
            )
            energy_distance = (
                abs(candidate.energy_mean - control.energy_mean) / max(abs(control.energy_mean), 1e-6)
                if control.energy_mean is not None
                else 0.0
            )
            return (
                emotion_match,
                -strength_distance,
                -f0_distance,
                -energy_distance,
                _dot(normalized[candidate.segment_id], center),
                candidate.segment_id,
            )

        selected = max(closest, key=rank)
        return SelectedReference(
            profile_id=selected.profile_id,
            segment_id=selected.segment_id,
            audio_path=selected.audio_path,
            text=selected.text,
            duration_seconds=selected.duration_seconds,
            timbre_similarity=_dot(normalized[selected.segment_id], center),
            emotion_label=selected.emotion_label,
            emotion_strength=selected.emotion_strength,
            f0_mean_hz=selected.f0_mean_hz,
            energy_mean=selected.energy_mean,
        )


class DisentanglerRuntimeError(RuntimeError):
    """Raised when the fixed local CUDA adapter profile cannot safely run."""


@contextmanager
def exclusive_gpu_training_stage() -> Iterator[None]:
    """Reject overlapping local adapter stages before they contend for cuda:0."""
    if not GPU_TRAINING_LOCK.acquire(blocking=False):
        raise DisentanglerRuntimeError("another local cuda:0 training stage is already running")
    try:
        yield
    finally:
        GPU_TRAINING_LOCK.release()


@dataclass(frozen=True)
class AdapterWeights:
    """A locally saved dual-branch adapter artifact and its final objective terms."""

    path: Path
    losses: Mapping[str, float]
    steps: int


class DisentanglerTrainer:
    """Train the local adapter using serial CUDA:0 FP16 micro-batches only.

    GPT-SoVITS, CAM++, and Emotion2Vec are deliberately not instantiated here:
    this adapter consumes their already-frozen, local feature outputs.  Each
    feature manifest entry contains exactly one segment, while exchange pairs
    supply its same-speaker partner through a second serial model forward.
    """

    DEVICE = "cuda:0"
    BATCH_SIZE = 1
    FP16 = True

    def __init__(
        self,
        *,
        model: DualBranchDisentangler | None = None,
        learning_rate: float = 1e-4,
        epochs: int = 1,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning rate must be positive")
        if epochs < 1:
            raise ValueError("epochs must be at least one")
        self.model = model or DualBranchDisentangler()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device(self.DEVICE)
        self.batch_size = self.BATCH_SIZE
        self.fp16 = self.FP16
        self.grl_lambda = GRADIENT_REVERSAL_LAMBDA

    def _require_cuda_runtime(self) -> None:
        if self.device != torch.device("cuda:0"):
            raise DisentanglerRuntimeError("disentangler training is fixed to cuda:0")
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise DisentanglerRuntimeError("CUDA runtime is unavailable for cuda:0 disentangler training")
        total_memory = torch.cuda.get_device_properties(0).total_memory
        if total_memory < MINIMUM_VRAM_BYTES:
            raise DisentanglerRuntimeError("cuda:0 has less than the required 8000 MiB VRAM")

    @staticmethod
    def _require_finite_gradients(model: DualBranchDisentangler) -> None:
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                raise FloatingPointError(f"missing gradient for trainable disentangler parameter: {name}")
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"non-finite gradient for disentangler parameter: {name}")

    def _train_pair(
        self,
        pair: SerialExchangePair,
        *,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
    ) -> dict[str, float]:
        source = pair.source.to(self.device)
        partner = pair.partner.to(self.device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.fp16):
            source_condition = self.model(source)
            partner_condition = self.model(partner)
            same_segment_logits = self.model.independence_discriminator(
                source_condition.timbre,
                source_condition.prosody,
                self.grl_lambda,
            )
            exchanged_segment_logits = self.model.independence_discriminator(
                source_condition.timbre,
                partner_condition.prosody,
                self.grl_lambda,
            )
            discriminator_logits = torch.cat(
                (same_segment_logits.reshape(1), exchanged_segment_logits.reshape(1))
            )
            same_pair_targets = torch.tensor(
                (1.0, 0.0), device=self.device, dtype=discriminator_logits.dtype
            )
            losses = compute_disentanglement_losses(
                timbre=source_condition.timbre,
                timbre_target=partner_condition.timbre.detach(),
                prosody=self.model.reconstruct_frame_prosody(source_condition.prosody),
                prosody_target=source.prosody[..., :4],
                emotion=self.model.predict_emotion(source_condition.prosody),
                emotion_target=source.emotion,
                discriminator_logits=discriminator_logits,
                same_pair_targets=same_pair_targets,
                xcov_timbre=torch.cat(
                    (source_condition.timbre, partner_condition.timbre), dim=0
                ),
                xcov_prosody=torch.cat(
                    (
                        source_condition.prosody.mean(dim=1),
                        partner_condition.prosody.mean(dim=1),
                    ),
                    dim=0,
                ),
            )
            total_loss = sum(LOSS_WEIGHTS[name] * losses[name] for name in LOSS_WEIGHTS)

        if not torch.isfinite(total_loss):
            raise FloatingPointError("non-finite disentangler training loss")
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        self._require_finite_gradients(self.model)
        scaler.step(optimizer)
        scaler.update()
        return {name: float(value.detach().float().cpu()) for name, value in losses.items()}

    def fit(
        self,
        manifest: Iterable[FeatureBatch],
        output_dir: Path | str,
        *,
        max_steps: int | None = None,
    ) -> AdapterWeights:
        """Fit one or more serial passes over locally extracted feature batches."""
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be at least one when provided")
        target_dir = validate_adapter_output_dir(output_dir)
        pairs = serial_exchange_pairs(manifest)
        with exclusive_gpu_training_stage():
            self._require_cuda_runtime()
            torch.cuda.set_device(self.device)
            self.model.to(self.device).train()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
            scaler = torch.amp.GradScaler("cuda", enabled=self.fp16)

            last_losses: dict[str, float] | None = None
            steps = 0
            for _ in range(self.epochs):
                for pair in pairs:
                    last_losses = self._train_pair(pair, optimizer=optimizer, scaler=scaler)
                    steps += 1
                    if max_steps is not None and steps >= max_steps:
                        break
                if max_steps is not None and steps >= max_steps:
                    break

            if last_losses is None:
                raise RuntimeError("feature manifest produced no serial exchange pairs")
            weights_path = target_dir / "disentangler-adapter.pt"
            atomic_save_adapter_payload(
                weights_path,
                {
                    "state_dict": {
                        name: value.detach().cpu() for name, value in self.model.state_dict().items()
                    },
                    "metadata": {
                        "device": self.DEVICE,
                        "batch_size": self.batch_size,
                        "fp16": self.fp16,
                        "grl_lambda": self.grl_lambda,
                        "loss_weights": dict(LOSS_WEIGHTS),
                        "steps": steps,
                    },
                },
            )
            return AdapterWeights(
                path=weights_path,
                losses=MappingProxyType(dict(last_losses)),
                steps=steps,
            )
