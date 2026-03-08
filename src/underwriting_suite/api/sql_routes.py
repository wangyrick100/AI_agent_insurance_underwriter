"""SQL read & write endpoints – X4 and X5."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from underwriting_suite.agent.skills.skill_sql_read import skill_sql_read
from underwriting_suite.agent.skills.skill_db_write import skill_db_write_commit, skill_db_write_plan

router = APIRouter(prefix="/v1/sql", tags=["sql"])


class SQLQueryRequest(BaseModel):
    question: str
    db_id: Optional[str] = None
    constraints: Optional[dict[str, Any]] = None


class WritePlanRequest(BaseModel):
    extraction_bundle: Optional[dict[str, Any]] = None
    applicant_id: str


class WriteCommitRequest(BaseModel):
    write_plan_id: str
    confirmation_token: str


@router.post("/query")
async def sql_query(req: SQLQueryRequest):
    """Execute a SELECT-only SQL query via X4 (text-to-SQL)."""
    result = await skill_sql_read(req.model_dump())
    return result


@router.post("/write/plan")
async def create_write_plan(req: WritePlanRequest):
    """Generate a write plan via X5 – does NOT execute."""
    result = await skill_db_write_plan(req.model_dump())
    return result


@router.post("/write/commit")
async def commit_write_plan(req: WriteCommitRequest):
    """Commit a write plan. Requires valid confirmation_token."""
    if not req.confirmation_token:
        raise HTTPException(status_code=400, detail="confirmation_token is required")
    result = await skill_db_write_commit(req.model_dump())
    return result
