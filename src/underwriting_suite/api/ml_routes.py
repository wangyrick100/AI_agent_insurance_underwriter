"""ML scoring endpoints – explicit X2 trigger."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from underwriting_suite.agent.tools.x2_risk_model import x2_score

router = APIRouter(prefix="/v1/ml", tags=["ml"])


class ScoreRequest(BaseModel):
    applicant_id: Optional[str] = None
    feature_payload: Optional[dict[str, Any]] = None


@router.post("/score")
async def run_score(req: ScoreRequest):
    """Explicitly trigger X2 risk scoring."""
    result = await x2_score(req.model_dump())
    return result
