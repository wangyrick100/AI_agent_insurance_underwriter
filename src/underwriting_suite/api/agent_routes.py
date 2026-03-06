"""Agent chat endpoint – Supervisor ReAct loop."""

from __future__ import annotations

from fastapi import APIRouter

from underwriting_suite.agent.schemas import ChatRequest, ChatResponse
from underwriting_suite.agent.supervisor import run_supervisor

router = APIRouter(prefix="/v1/agent", tags=["agent"])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Run the Supervisor ReAct loop for an underwriter's request.

    The LLM dynamically plans which tools to invoke (X1–X6),
    executes them, reflects, and synthesizes a final answer.
    """
    result = await run_supervisor(
        user_message=req.message,
        session_id=req.session_id,
        applicant_id=req.applicant_id,
        confirmation_token=req.confirmation_token,
    )
    return ChatResponse(**result)
