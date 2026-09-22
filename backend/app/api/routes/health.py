from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable, Literal

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.engine import Engine

from backend.app.core.config import AppSettings
from backend.app.services.gpt_sovits import GPTSoVITSAdapter
from backend.app.services.storage import LocalStorage


REQUIRED_CHECKS = (
    "database",
    "storage",
    "gpu",
    "gpt_sovits",
    "emotion2vec",
    "audioseal",
)

PUBLIC_READINESS_MESSAGES = {
    "database": ("本地数据库不可用", "本地数据库可用"),
    "storage": ("本地存储不可用", "本地存储可用"),
    "gpu": ("GPU cuda:0 未就绪", "GPU cuda:0 可用"),
    "gpt_sovits": ("GPT-SoVITS 未就绪", "GPT-SoVITS 可用"),
    "emotion2vec": ("Emotion2Vec 未就绪", "Emotion2Vec 可用"),
    "audioseal": ("AudioSeal 未就绪", "AudioSeal 可用"),
}


class HealthCheck(BaseModel):
    ok: bool
    message: str


class LiveResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, HealthCheck]


@dataclass(frozen=True)
class HealthDependencies:
    gpu: Callable[[], tuple[bool, str]] | None = None
    gpt_sovits: Callable[[], tuple[bool, str]] | None = None
    emotion2vec: Callable[[], tuple[bool, str]] | None = None
    audioseal: Callable[[], tuple[bool, str]] | None = None


class HealthService:
    """Report only local readiness signals; no remote health checks are allowed."""

    def __init__(
        self,
        *,
        engine: Engine,
        storage: LocalStorage,
        settings: AppSettings,
        project_root: Path | str | None = None,
        dependencies: HealthDependencies | None = None,
    ) -> None:
        self.engine = engine
        self.storage = storage
        self.settings = settings
        self.project_root = Path(project_root or Path(__file__).resolve().parents[4]).resolve()
        self.dependencies = dependencies or HealthDependencies()
        self._gpt_sovits_lock = Lock()
        self._gpt_sovits_signature: tuple[tuple[str, bool, int | None, int | None], ...] | None = None
        self._gpt_sovits_result: tuple[bool, str] | None = None

    @staticmethod
    def _result(ok: bool, message: str) -> tuple[bool, str]:
        return bool(ok), message

    def _database(self) -> tuple[bool, str]:
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except Exception as exc:  # pragma: no cover - exact driver errors vary by host
            return self._result(False, f"database unavailable: {exc}")
        return self._result(True, "local SQLite database is available")

    def _storage(self) -> tuple[bool, str]:
        try:
            root = self.storage.root
            root.mkdir(parents=True, exist_ok=True)
            probe = root / ".health-check"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            return self._result(False, f"local storage unavailable: {exc}")
        return self._result(True, f"local storage is available at {root}")

    def _gpu(self) -> tuple[bool, str]:
        if self.settings.runtime.device != "cuda:0":
            return self._result(False, "runtime device must be cuda:0")
        try:
            import torch

            if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
                return self._result(False, "cuda:0 is unavailable")
            total_mib = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        except (ImportError, OSError, RuntimeError) as exc:
            return self._result(False, f"cuda:0 probe failed: {exc}")
        required_mib = 8000
        if total_mib < required_mib:
            return self._result(False, f"cuda:0 has {total_mib:.1f} MiB; requires {required_mib} MiB")
        return self._result(True, f"cuda:0 is available with {total_mib:.1f} MiB")

    def _model_state_signature(self) -> tuple[tuple[str, bool, int | None, int | None], ...]:
        """Return cheap file metadata for all artifacts covered by the local manifest."""
        models_root = (self.project_root / "models").resolve()
        checksum_path = models_root / "checksums.sha256"
        paths = [checksum_path]

        if checksum_path.is_file():
            for raw_line in checksum_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                relative = Path(parts[1].lstrip("*").strip())
                candidate = (self.project_root / relative).resolve()
                try:
                    candidate.relative_to(models_root)
                except ValueError:
                    continue
                paths.append(candidate)

        states: list[tuple[str, bool, int | None, int | None]] = []
        for path in paths:
            try:
                stat = path.stat()
            except OSError:
                states.append((str(path.relative_to(self.project_root)), False, None, None))
            else:
                states.append((str(path.relative_to(self.project_root)), True, stat.st_size, stat.st_mtime_ns))
        return tuple(states)

    def _gpt_sovits(self) -> tuple[bool, str]:
        """Cache display-only readiness probes until the local model state changes."""
        with self._gpt_sovits_lock:
            signature = self._model_state_signature()
            if signature == self._gpt_sovits_signature and self._gpt_sovits_result is not None:
                return self._gpt_sovits_result

            adapter = GPTSoVITSAdapter(
                vendor_dir=self.project_root / "vendor" / "GPT-SoVITS",
                models_root=self.project_root / "models",
            )
            report = adapter.probe()
            if not report.available:
                result = self._result(False, report.reason or "GPT-SoVITS is unavailable")
            else:
                result = self._result(True, f"GPT-SoVITS {report.tag} is ready")
            self._gpt_sovits_signature = signature
            self._gpt_sovits_result = result
            return result

    def _emotion2vec(self) -> tuple[bool, str]:
        model_name = self.settings.models.emotion_model
        model_dir = self.project_root / "models" / "emotion2vec"
        if not model_dir.is_dir():
            return self._result(False, f"local Emotion2Vec weights are unavailable: {model_name}")
        return self._result(True, f"local Emotion2Vec weights are available: {model_dir}")

    def _audioseal(self) -> tuple[bool, str]:
        audioseal_root = self.project_root / "models" / "audioseal"
        required = (
            audioseal_root / "generator_base.pth",
            audioseal_root / "detector_base.pth",
        )
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            return self._result(False, f"local AudioSeal weights are missing: {', '.join(missing)}")
        return self._result(True, "local AudioSeal generator and detector are available")

    def _run(self, name: str, default: Callable[[], tuple[bool, str]]) -> HealthCheck:
        callback = getattr(self.dependencies, name) or default
        try:
            ok, message = callback()
        except Exception as exc:  # pragma: no cover - defensive boundary for a host probe
            ok, message = False, f"{name} probe failed: {exc}"
        return HealthCheck(ok=ok, message=message)

    def checks(self) -> dict[str, HealthCheck]:
        database_ok, database_message = self._database()
        storage_ok, storage_message = self._storage()
        return {
            "database": HealthCheck(ok=database_ok, message=database_message),
            "storage": HealthCheck(ok=storage_ok, message=storage_message),
            "gpu": self._run("gpu", self._gpu),
            "gpt_sovits": self._run("gpt_sovits", self._gpt_sovits),
            "emotion2vec": self._run("emotion2vec", self._emotion2vec),
            "audioseal": self._run("audioseal", self._audioseal),
        }

    def ready(self) -> ReadyResponse:
        internal_checks = self.checks()
        public_checks = {
            name: HealthCheck(
                ok=check.ok,
                message=PUBLIC_READINESS_MESSAGES[name][1 if check.ok else 0],
            )
            for name, check in internal_checks.items()
        }
        is_ready = all(check.ok for check in public_checks.values())
        return ReadyResponse(
            status="ready" if is_ready else "not_ready", checks=public_checks
        )


router = APIRouter(prefix="/api/health", tags=["health"])


def _service(request: Request) -> HealthService:
    return request.app.state.health_service


@router.get("/live", response_model=LiveResponse)
def live() -> LiveResponse:
    return LiveResponse(status="alive")


@router.get(
    "/ready",
    response_model=ReadyResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadyResponse}},
)
def ready(request: Request):
    report = _service(request).ready()
    code = status.HTTP_200_OK if report.status == "ready" else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=code, content=report.model_dump(mode="json"))
