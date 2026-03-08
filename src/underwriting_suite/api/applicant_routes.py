"""Applicant endpoints – summary, list, search."""

from __future__ import annotations

import uuid
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from underwriting_suite.agent.schemas import ApplicantSummary
from underwriting_suite.agent.skills.skill_db_write import get_plans_for_applicant
from underwriting_suite.db.database import get_session
from underwriting_suite.models.applicant import Applicant
from underwriting_suite.models.document import Document
from underwriting_suite.models.extraction import Extraction
from underwriting_suite.models.risk_score import RiskScore

router = APIRouter(prefix="/v1/applicants", tags=["applicants"])


class ApplicantCreate(BaseModel):
    first_name: str
    last_name: str
    date_of_birth: Optional[date] = None
    gender: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    policy_number: Optional[str] = None


@router.get("")
async def list_applicants(
    q: Optional[str] = Query(None, description="Search by name or policy number"),
    limit: int = Query(20, le=100),
    session: AsyncSession = Depends(get_session),
):
    """List / search applicants."""
    query = select(Applicant).order_by(Applicant.created_at.desc()).limit(limit)
    if q:
        query = query.where(
            Applicant.first_name.ilike(f"%{q}%")
            | Applicant.last_name.ilike(f"%{q}%")
            | Applicant.policy_number.ilike(f"%{q}%")
        )
    result = await session.execute(query)
    applicants = result.scalars().all()
    return [
        {
            "id": a.id,
            "first_name": a.first_name,
            "last_name": a.last_name,
            "date_of_birth": str(a.date_of_birth) if a.date_of_birth else None,
            "status": a.status,
            "policy_number": a.policy_number,
        }
        for a in applicants
    ]


@router.post("")
async def create_applicant(
    data: ApplicantCreate,
    session: AsyncSession = Depends(get_session),
):
    """Create a new applicant."""
    applicant = Applicant(
        id=str(uuid.uuid4()),
        first_name=data.first_name,
        last_name=data.last_name,
        date_of_birth=data.date_of_birth,
        gender=data.gender,
        email=data.email,
        phone=data.phone,
        policy_number=data.policy_number,
    )
    session.add(applicant)
    await session.commit()
    return {"id": applicant.id, "status": "created"}


@router.get("/{applicant_id}/summary", response_model=ApplicantSummary)
async def get_applicant_summary(
    applicant_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get consolidated state for an applicant: docs, extractions, score, plans."""
    # Fetch applicant
    result = await session.execute(
        select(Applicant).where(Applicant.id == applicant_id)
    )
    applicant = result.scalar_one_or_none()
    if not applicant:
        raise HTTPException(status_code=404, detail="Applicant not found")

    # Fetch documents
    docs_result = await session.execute(
        select(Document).where(Document.applicant_id == applicant_id)
    )
    documents = [
        {
            "id": d.id,
            "filename": d.filename,
            "doc_type": d.doc_type,
            "status": d.status,
            "chunk_count": d.chunk_count,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in docs_result.scalars().all()
    ]

    # Fetch extractions
    ext_result = await session.execute(
        select(Extraction).where(Extraction.applicant_id == applicant_id)
    )
    extractions = [
        {
            "id": e.id,
            "entity_type": e.entity_type,
            "entity_name": e.entity_name,
            "entity_value": e.entity_value,
            "confidence": e.confidence,
            "evidence_snippet": e.evidence_snippet,
        }
        for e in ext_result.scalars().all()
    ]

    # Fetch latest risk score
    score_result = await session.execute(
        select(RiskScore)
        .where(RiskScore.applicant_id == applicant_id)
        .order_by(RiskScore.created_at.desc())
        .limit(1)
    )
    score_row = score_result.scalar_one_or_none()
    latest_score = None
    if score_row:
        latest_score = {
            "score": score_row.score,
            "confidence": score_row.confidence,
            "risk_class": score_row.risk_class,
            "feature_rationale": score_row.feature_rationale,
            "similar_cases": score_row.similar_cases,
            "model_version": score_row.model_version,
        }

    # Fetch pending write plans
    plans = get_plans_for_applicant(applicant_id)
    pending_plans = [p.model_dump() for p in plans if p.status == "pending"]

    return ApplicantSummary(
        applicant_id=applicant.id,
        first_name=applicant.first_name,
        last_name=applicant.last_name,
        status=applicant.status,
        documents=documents,
        extractions=extractions,
        latest_score=latest_score,
        pending_write_plans=pending_plans,
    )
