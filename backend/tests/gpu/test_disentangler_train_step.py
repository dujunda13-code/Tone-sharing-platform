import shutil
from pathlib import Path

import pytest
import torch

from backend.app.ml.datasets import FeatureBatch
from backend.app.ml.disentangler import DualBranchDisentangler
from backend.app.services.disentanglement import DisentanglerTrainer


def _segment(segment_id: str) -> FeatureBatch:
    return FeatureBatch(
        timbre=torch.randn(1, 192),
        prosody=torch.randn(1, 12, 1028),
        emotion=torch.randn(1, 1024),
        speaker_ids=("speaker-a",),
        segment_ids=(segment_id,),
    )


@pytest.mark.gpu
def test_trainer_uses_serial_fp16_batch_one_step_with_finite_gradients(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("pinned CUDA runtime is unavailable")

    device = torch.device("cuda:0")
    total_memory = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.reset_peak_memory_stats(device)
    model = DualBranchDisentangler()
    trainer = DisentanglerTrainer(model=model)
    artifact_root = (Path("data") / "temp" / "pytest-task7-adapter").resolve()
    artifact_dir = (artifact_root / tmp_path.name).resolve()

    try:
        weights = trainer.fit(
            [_segment("segment-a"), _segment("segment-b")],
            artifact_dir,
            max_steps=1,
        )

        torch.cuda.synchronize(device)
        assert weights.path.is_file()
        assert weights.steps == 1
        assert set(weights.losses) == {
            "adversarial",
            "timbre_consistency",
            "prosody",
            "emotion",
            "xcov",
        }
        assert all(torch.isfinite(torch.tensor(value)) and value >= 0 for value in weights.losses.values())
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        assert torch.cuda.max_memory_allocated(device) < total_memory * 0.90
    finally:
        if artifact_dir.exists():
            assert artifact_dir.is_relative_to(artifact_root)
            shutil.rmtree(artifact_dir)
