"""Official WebUI runner entrypoint executed as a child process inside vendor/GPT-SoVITS environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser(description="WebUI Inference Runner")
    parser.add_argument("--config", required=True, help="Path to JSON run config")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))

    # Set necessary environment variables before importing inference_webui
    os.environ["version"] = config.get("version", "v2ProPlus")
    os.environ["gpt_path"] = config["gpt_weight"]
    os.environ["sovits_path"] = config["sovits_weight"]

    # Import official WebUI module
    import inference_webui

    # Set random seed if requested
    seed = config.get("seed", -1)
    if seed != -1:
        inference_webui.set_seed(seed)

    # Map languages using dict_language
    prompt_lang_name = "中文" if config["prompt_lang"] == "zh" else "英文"
    target_lang_name = "中文" if config["text_lang"] == "zh" else "英文"

    # Map how_to_cut
    how_to_cut_str = inference_webui.i18n(config.get("how_to_cut", "不切"))

    # Build inp_refs objects (inference_webui expects each element in inp_refs to have a .name attribute)
    auxiliary_audios = config.get("auxiliary_audios", [])
    inp_refs = [SimpleNamespace(name=str(Path(p).resolve())) for p in auxiliary_audios] if auxiliary_audios else None

    # Call get_tts_wav generator
    generator = inference_webui.get_tts_wav(
        ref_wav_path=str(Path(config["reference_audio"]).resolve()),
        prompt_text=config["prompt_text"],
        prompt_language=prompt_lang_name,
        text=config["text"],
        text_language=target_lang_name,
        how_to_cut=how_to_cut_str,
        top_k=int(config.get("top_k", 15)),
        top_p=float(config.get("top_p", 1.0)),
        temperature=float(config.get("temperature", 1.0)),
        ref_free=False,
        speed=float(config.get("speed", 1.0)),
        if_freeze=False,
        inp_refs=inp_refs,
        sample_steps=int(config.get("sample_steps", 8)),
        if_sr=False,
        pause_second=float(config.get("pause_second", 0.3)),
    )

    # get_tts_wav yields (sample_rate, audio_numpy_array)
    sampling_rate, audio_data = next(generator)

    output_path = Path(config["output_wav"]).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio_data, sampling_rate, subtype="PCM_16")


if __name__ == "__main__":
    main()
