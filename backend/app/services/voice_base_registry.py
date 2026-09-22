"""Offline base-model registry for zero-shot voice synthesis.

Client synthesis always loads frozen GPT-SoVITS base weights selected
through this registry. Every weight must live under the local ``models/``
root and be listed in ``models/checksums.sha256``; the registry re-verifies
existence and SHA-256 on every resolve, so a missing, unregistered, or
tampered deployment fails closed with :class:`BaseModelUnavailable`.
Runtime model download never happens here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from backend.app.core.config import AppSettings


class BaseModelUnavailable(RuntimeError):
    """Raised when a configured base model cannot be verified locally."""


@dataclass(frozen=True)
class ResolvedVoiceBase:
    id: str
    source_tag: str
    gpt_weight: Path
    sovits_weight: Path


class VoiceBaseRegistry:
    def __init__(self, settings: AppSettings, models_root: Path, checksum_path: Path):
        self.settings = settings
        self.models_root = models_root.resolve()
        self.checksum_path = checksum_path.resolve()

    def resolve(self, base_model_id: str | None = None) -> ResolvedVoiceBase:
        selected_id = base_model_id or self.settings.models.active_voice_base
        configured = self.settings.models.voice_bases.get(selected_id)
        if configured is None:
            raise BaseModelUnavailable(f"未知基础模型: {selected_id}")
        return ResolvedVoiceBase(
            id=selected_id,
            source_tag=configured.source_tag,
            gpt_weight=self._verified_path(configured.gpt_weight),
            sovits_weight=self._verified_path(configured.sovits_weight),
        )

    def _verified_path(self, relative_path: Path) -> Path:
        resolved = (self.models_root / relative_path).resolve()
        try:
            manifest_relative = resolved.relative_to(self.models_root)
        except ValueError as exc:
            raise BaseModelUnavailable("基础模型权重必须位于 models/ 之内") from exc
        manifest_key = f"models/{manifest_relative.as_posix()}"
        expected = self._registered_digest(manifest_key)
        if not resolved.is_file():
            raise BaseModelUnavailable(f"基础模型权重文件缺失: {manifest_key}")
        if hashlib.sha256(resolved.read_bytes()).hexdigest() != expected:
            raise BaseModelUnavailable(f"基础模型权重 SHA-256 校验失败: {manifest_key}")
        return resolved

    def _registered_digest(self, manifest_key: str) -> str:
        if not self.checksum_path.is_file():
            raise BaseModelUnavailable(f"模型校验清单缺失: {self.checksum_path.name}")
        for raw_line in self.checksum_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            expected, relative_text = parts
            if relative_text.lstrip("*").strip() == manifest_key:
                return expected.lower()
        raise BaseModelUnavailable(f"基础模型权重未登记校验和: {manifest_key}")
