from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _script(name: str) -> str:
    return (ROOT / "scripts" / name).read_text(encoding="utf-8-sig")


def test_model_placement_links_vendor_pretrained_paths_into_checksum_tree():
    """The vendored zero-shot inference resolves pretrained paths from the
    pinned checkout working directory (sv.py hardcodes sv_path and the text
    frontend hardcodes the G2PW/fast-langdetect locations), so the placement
    script must link each checksum-verified models/ copy into that tree."""
    script = _script("download_models.ps1")

    assert "chinese-hubert-base" in script
    assert "chinese-roberta-wwm-ext-large" in script
    assert "pretrained_models\\sv" in script
    assert "pretrained_models\\fast_langdetect" in script
    assert "text\\G2PWModel" in script
    assert "Junction" in script
    # Fail-closed ordering: sources are checksum-verified before linking.
    assert script.index("MODEL_CHECKSUMS_VERIFIED") < script.index("Junction")


def test_requirements_pin_the_zero_shot_inference_import_chain():
    """inference_cli -> inference_webui imports fail closed when any of these
    are missing; pin exactly what the vendored entrypoint imports."""
    requirements = (ROOT / "requirements-app.txt").read_text(encoding="utf-8")

    for pinned in (
        "fast_langdetect==",
        "split-lang==",
        "wordsegment==",
        "pytorch-lightning==",
        "torchmetrics==1.5.0",
        "peft==",
        "g2p_en==",
        "psutil==",
    ):
        assert pinned in requirements, pinned
