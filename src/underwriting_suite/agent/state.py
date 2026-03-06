"""LangGraph state definition for the Supervisor ReAct loop."""

from __future__ import annotations

from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass
class SupervisorState:
    """Mutable working memory passed through the LangGraph loop.

    Every node reads/writes this state so the Supervisor can maintain
    context across plan → execute → reflect iterations.
    """

    # ── User request ────────────────────────
    user_message: str = ""
    session_id: str = ""
    applicant_id: Optional[str] = None

    # ── Conversation history (list of dicts) ─
    messages: list[dict[str, Any]] = field(default_factory=list)

    # ── Current plan step ───────────────────
    current_plan: Optional[dict[str, Any]] = None  # serialised PlanStep
    current_tool_result: Optional[dict[str, Any]] = None

    # ── Accumulated tool outputs ────────────
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)

    # ── Write-plan confirmation state ───────
    pending_write_plan_id: Optional[str] = None
    confirmation_token: Optional[str] = None

    # ── Loop control ────────────────────────
    iteration: int = 0
    max_iterations: int = 15
    should_stop: bool = False
    final_answer: str = ""

    # ── Trace log ───────────────────────────
    trace: list[dict[str, Any]] = field(default_factory=list)

    # ── Error recovery ──────────────────────
    last_error: Optional[str] = None
    consecutive_errors: int = 0
    max_consecutive_errors: int = 3
