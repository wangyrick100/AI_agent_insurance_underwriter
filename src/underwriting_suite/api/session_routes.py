"""Session trace endpoints – admin debug view."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from underwriting_suite.db.database import get_session
from underwriting_suite.models.session_trace import SessionTrace

router = APIRouter(prefix="/v1/sessions", tags=["sessions"])


@router.get("/{session_id}/trace")
async def get_session_trace(
    session_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Retrieve the full ReAct trace for a session (admin/debug)."""
    result = await session.execute(
        select(SessionTrace)
        .where(SessionTrace.session_id == session_id)
        .order_by(SessionTrace.step_index)
    )
    steps = result.scalars().all()
    if not steps:
        raise HTTPException(status_code=404, detail="Session trace not found")

    return {
        "session_id": session_id,
        "steps": [
            {
                "step_index": s.step_index,
                "step_type": s.step_type,
                "tool_name": s.tool_name,
                "tool_input_summary": s.tool_input_summary,
                "tool_output_summary": s.tool_output_summary,
                "thought_summary": s.thought_summary,
                "success": s.success,
                "duration_ms": s.duration_ms,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in steps
        ],
    }


@router.get("")
async def list_sessions(
    limit: int = 20,
    session: AsyncSession = Depends(get_session),
):
    """List recent sessions with trace counts."""
    from sqlalchemy import func, distinct

    result = await session.execute(
        select(
            SessionTrace.session_id,
            func.count(SessionTrace.id).label("step_count"),
            func.min(SessionTrace.created_at).label("started_at"),
        )
        .group_by(SessionTrace.session_id)
        .order_by(func.min(SessionTrace.created_at).desc())
        .limit(limit)
    )

    return [
        {
            "session_id": row.session_id,
            "step_count": row.step_count,
            "started_at": row.started_at.isoformat() if row.started_at else None,
        }
        for row in result.all()
    ]
