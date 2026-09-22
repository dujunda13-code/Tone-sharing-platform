from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_verification_script_has_an_explicit_auth_guard_check():
    script = (ROOT / "scripts" / "verify.ps1").read_text(encoding="utf-8")

    assert 'Invoke-Checked "AUTH_GUARD"' in script
    assert "test_protected_training_route_requires_login" in script


def test_verification_script_uses_short_reference_and_zero_shot_acceptance():
    script = (ROOT / "scripts" / "verify.ps1").read_text(encoding="utf-8-sig")

    assert "AuthorizedReference" in script
    assert "TIMBRE_AUTHORIZED_REFERENCE" in script
    assert "TIMBRE_AUTHORIZED_DATASET" not in script
    assert "test_end_to_end_gpu.py" in script
    assert "test_gpt_sovits_smoke.py" in script
    assert "test_watermark_robustness.py" in script
    assert "backend/tests/gpu -m gpu" not in script
    assert "test_disentangler_train_step.py" not in script
    assert "zero-shot" in script.lower()
    assert "TRAINING" not in script


def test_verification_script_does_not_use_shell_wrappers_or_skip_real_checks():
    script = (ROOT / "scripts" / "verify.ps1").read_text(encoding="utf-8-sig")

    assert "npm.cmd" not in script
    assert "cmd.exe" not in script
    assert "-SkipGpu" in script
    assert "CPU_ONLY" in script
    assert "FINAL_RESULT=PASS" in script
    assert '$report.final_result = "INCOMPLETE"' in script
