from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from backend.app.schemas.synthesis import EmotionControl, LocalSynthesisOptions, SynthesisCreate
from backend.app.services.synthesis_pipeline import GPTSoVITSLocalSynthesizer, ZeroShotSynthesisReference
from backend.app.services.voice_base_registry import ResolvedVoiceBase


def test_local_webui_options_reach_the_official_bridge(tmp_path: Path) -> None:
    captured = []

    class RecordingBridge:
        def synthesize(self, spec):
            captured.append(spec)
            return spec.output_wav

    request = SynthesisCreate(
        voice_profile_id="voice-1",
        text="测试文本",
        text_lang="zh",
        emotion=EmotionControl(mode="auto"),
        consent_confirmed=True,
        local_options=LocalSynthesisOptions(
            cut_method="punctuation",
            speed=1.2,
            pause_seconds=0.25,
            top_k=30,
            top_p=0.8,
            temperature=0.7,
        ),
    )
    reference_wav = tmp_path / "reference.wav"
    base = ResolvedVoiceBase(
        id="base",
        source_tag="test",
        gpt_weight=tmp_path / "s1.ckpt",
        sovits_weight=tmp_path / "s2.pth",
    )
    reference = ZeroShotSynthesisReference(
        reference_audio=reference_wav,
        prompt_text="参考文本",
        prompt_lang="zh",
        emotion_label="neutral",
        base=base,
        primary_audio_path=reference_wav,
    )
    output = tmp_path / "output.wav"

    GPTSoVITSLocalSynthesizer(webui_bridge=RecordingBridge()).synthesize(
        request, SimpleNamespace(public_weight_dir=None), reference, output
    )

    spec = captured[0]
    assert spec.how_to_cut == "按标点符号切"
    assert spec.speed == 1.2
    assert spec.pause_second == 0.25
    assert spec.top_k == 30
    assert spec.top_p == 0.8
    assert spec.temperature == 0.7
