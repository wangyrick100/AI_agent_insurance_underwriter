"""Extraction endpoints – explicit X1 trigger."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from underwriting_suite.agent.skills.skill_extraction import skill_extraction

router = APIRouter(prefix="/v1/extraction", tags=["extraction"])


class ExtractionRequest(BaseModel):
    doc_ids: list[str] = []
    raw_text: Optional[str] = None
    applicant_id: Optional[str] = None


@router.post("/run")
async def run_extraction(req: ExtractionRequest):
    """Explicitly trigger X1 entity extraction."""
    result = await skill_extraction(req.model_dump())
    return result
