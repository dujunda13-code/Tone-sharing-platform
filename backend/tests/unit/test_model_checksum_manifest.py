import hashlib
from pathlib import Path
import shutil
import subprocess
from uuid import uuid4
import zipfile

from backend.app.services.gpt_sovits import validate_model_manifest


ROOT = Path(__file__).resolve().parents[3]


def test_empty_manifest_is_not_verified(tmp_path: Path):
    manifest = tmp_path / "checksums.sha256"
    manifest.write_text("# no models yet\n", encoding="utf-8")

    report = validate_model_manifest(manifest, tmp_path / "models")

    assert report.valid is False
    assert report.verified_count == 0
    assert report.reason == "model checksum manifest has no entries"


def test_manifest_rejects_path_outside_models_root(tmp_path: Path):
    manifest = tmp_path / "checksums.sha256"
    manifest.write_text(f"{'0' * 64} ../escape.pth\n", encoding="utf-8")

    report = validate_model_manifest(manifest, tmp_path / "models")

    assert report.valid is False
    assert report.reason == "model path escapes models root: ../escape.pth"


def test_manifest_reports_hash_mismatch_for_allowlisted_model(tmp_path: Path):
    models_root = tmp_path / "models"
    model_path = models_root / "gpt-sovits" / "base.pth"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"actual-model-content")
    manifest = tmp_path / "checksums.sha256"
    manifest.write_text(f"{'0' * 64} models/gpt-sovits/base.pth\n", encoding="utf-8")

    report = validate_model_manifest(manifest, models_root)

    assert report.valid is False
    assert report.verified_count == 0
    assert report.reason == "SHA-256 mismatch for models/gpt-sovits/base.pth"


def test_manifest_verifies_matching_allowlisted_model(tmp_path: Path):
    models_root = tmp_path / "models"
    model_path = models_root / "gpt-sovits" / "base.pth"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"trusted-model-content")
    expected = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest = tmp_path / "checksums.sha256"
    manifest.write_text(f"{expected} models/gpt-sovits/base.pth\n", encoding="utf-8")

    report = validate_model_manifest(manifest, models_root)

    assert report.valid is True
    assert report.verified_count == 1
    assert report.missing_paths == ()
    assert report.reason is None


def test_download_models_script_rejects_empty_manifest(tmp_path: Path):
    fixture_dir = ROOT / "models" / f".pytest-empty-manifest-{uuid4().hex}"
    manifest = fixture_dir / "checksums.sha256"
    fixture_dir.mkdir(parents=True)
    manifest.write_text("# no models yet\n", encoding="utf-8")

    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(ROOT / "scripts" / "download_models.ps1"),
                "-ChecksumFile",
                manifest.relative_to(ROOT).as_posix(),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    finally:
        shutil.rmtree(fixture_dir)

    assert result.returncode != 0
    assert "model checksum manifest has no entries" in result.stderr


def test_download_models_script_validates_temp_allowlisted_model_in_conda_environment():
    fixture_dir = ROOT / "models" / f".pytest-manifest-{uuid4().hex}"
    model_path = fixture_dir / "gpt-sovits" / "fixture.pth"
    manifest = fixture_dir / "checksums.sha256"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"trusted-model-content")
    expected = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest.write_text(
        f"{expected} {model_path.relative_to(ROOT).as_posix()}\n",
        encoding="utf-8",
    )

    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(ROOT / "scripts" / "download_models.ps1"),
                "-ChecksumFile",
                manifest.relative_to(ROOT).as_posix(),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    finally:
        shutil.rmtree(fixture_dir)

    assert result.returncode == 0, result.stderr
    assert "MODEL_CHECKSUMS_VERIFIED=1" in result.stdout


def test_download_models_script_rejects_tampered_g2pw_archive_before_extraction():
    fixture_dir = ROOT / "models" / f".pytest-g2pw-{uuid4().hex}"
    model_path = fixture_dir / "gpt-sovits" / "fixture.pth"
    manifest = fixture_dir / "checksums.sha256"
    archive = fixture_dir / "G2PWModel_1.1.zip"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"trusted-model-content")
    expected = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest.write_text(
        f"{expected} {model_path.relative_to(ROOT).as_posix()}\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("G2PWModel/g2pW.onnx", b"tampered-model")

    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(ROOT / "scripts" / "download_models.ps1"),
                "-ChecksumFile",
                manifest.relative_to(ROOT).as_posix(),
                "-InstallG2PW",
                "-G2PWArchivePath",
                archive.relative_to(ROOT).as_posix(),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    finally:
        shutil.rmtree(fixture_dir)

    assert result.returncode != 0
    assert "G2PW archive SHA-256 mismatch" in result.stderr


def test_g2pw_deployment_uses_the_official_archive_root_directory():
    script = (ROOT / "scripts" / "download_models.ps1").read_text(encoding="utf-8")

    assert '$stagedModel = Join-Path $stagingRoot "G2PWModel_1.1"' in script
    assert "Move-Item -LiteralPath $stagedModel -Destination $g2pwTarget" in script


def test_v2pro_speaker_vector_deployment_uses_verified_official_model():
    script = (ROOT / "scripts" / "download_models.ps1").read_text(encoding="utf-8")

    assert "InstallV2ProSpeakerVector" in script
    assert "speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common" in script
    assert "pretrained_eres2netv2w24s4ep4.ckpt" in script
    assert "740bb6584a99ee4cf910101536acba38c15a8017ea6a3a2813ec668fb62981f1" in script
    assert "V2ProSpeakerVectorPath" in script
