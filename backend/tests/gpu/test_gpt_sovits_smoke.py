from backend.app.services.gpt_sovits import GPTSoVITSAdapter
import pytest


@pytest.mark.gpu
def test_gpt_sovits_smoke_requires_pinned_checkout():
    info = GPTSoVITSAdapter().probe()
    if not info.available:
        total_mib = (
            "unavailable"
            if info.total_vram_bytes is None
            else f"{info.total_vram_bytes / 1024**2:.1f}"
        )
        required_mib = f"{info.required_vram_bytes / 1024**2:.1f}"
        missing_paths = ", ".join(path.as_posix() for path in info.missing_model_paths)
        raise AssertionError(
            "GPT-SoVITS smoke prerequisites are unavailable: "
            f"reason={info.reason}; total_mib={total_mib}; "
            f"required_mib={required_mib}; torch={info.torch_version}; "
            f"cuda={info.cuda_version}; missing_models={missing_paths or 'none'}"
        )
    assert info.tag == "20250606v2pro"
    assert info.device == "cuda:0"
