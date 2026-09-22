import pytest

from backend.app.services import disentanglement


def test_exclusive_gpu_stage_rejects_a_second_concurrent_training_call():
    assert disentanglement.GPU_TRAINING_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(disentanglement.DisentanglerRuntimeError, match="already running"):
            with disentanglement.exclusive_gpu_training_stage():
                pass
    finally:
        disentanglement.GPU_TRAINING_LOCK.release()
