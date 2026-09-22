from __future__ import annotations

from hashlib import sha256
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from backend.app.api.dependencies import get_current_user
from backend.app.schemas.auth import UserRecord
from backend.app.api.routes.synthesis import SynthesisService
from backend.app.schemas.synthesis import (
    WatermarkVerificationRequest,
    WatermarkVerificationResponse,
)
from backend.app.services.watermark import WatermarkService


class WatermarkVerificationError(ValueError):
    """Raised when an already-published local output no longer verifies."""


class SafetyService:
    """Recheck a published local synthesis output without accepting file-system paths."""

    def __init__(
        self,
        *,
        syntheses: SynthesisService | None = None,
        watermark: WatermarkService | None = None,
    ) -> None:
        self.syntheses = syntheses or SynthesisService()
        self.watermark = watermark or WatermarkService()

    def detect_watermark(
        self,
        job_id: str,
        owner_user_id: str | None = None,
    ) -> WatermarkVerificationResponse:
        detection = self.watermark.detect(self.syntheses.output_path(job_id, owner_user_id=owner_user_id))
        expected_payload = int.from_bytes(sha256(job_id.encode("utf-8")).digest()[:2], "big")
        if detection.probability < 0.80 or detection.payload != expected_payload:
            raise WatermarkVerificationError("local output did not pass watermark verification")
        return WatermarkVerificationResponse(
            job_id=job_id,
            probability=detection.probability,
            payload=detection.payload,
            payload_matches_job=detection.payload == expected_payload,
        )


router = APIRouter(prefix="/api/safety", tags=["safety"])


def _service(request: Request) -> SafetyService:
    service = getattr(request.app.state, "safety_service", None)
    if service is None:
        service = SafetyService()
        request.app.state.safety_service = service
    return service


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": {},
                "request_id": uuid4().hex,
            }
        },
    )


@router.post("/detect-watermark", response_model=WatermarkVerificationResponse)
def detect_watermark(
    body: WatermarkVerificationRequest,
    request: Request,
    current_user: UserRecord = Depends(get_current_user),
):
    try:
        return _service(request).detect_watermark(body.job_id, owner_user_id=current_user.id)
    except WatermarkVerificationError:
        return _error(409, "WATERMARK_SOURCE_UNCONFIRMED", "无法确认音频来源")
    except KeyError:
        return _error(404, "SYNTHESIS_NOT_FOUND", "合成任务不存在")
    except FileNotFoundError:
        return _error(404, "SYNTHESIS_AUDIO_NOT_READY", "已验证音频尚不可用")
