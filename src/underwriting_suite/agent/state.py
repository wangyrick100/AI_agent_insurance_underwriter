"""LangGraph state definition for the Supervisor ReAct loop.

The state object acts as the **working memory** shared across every
node in the graph.  It tracks:

  • User request & session metadata
  • Conversation history with role-tagged messages
  • Current plan step and last tool result
  • Accumulated tool outputs (short-term memory)
  • Summarised long-term memory (compressed after N iterations)
  • Write-plan confirmation workflow
  • Token / cost budget tracking
  • Loop-control counters and error-recovery state
  • Full structured trace for observability & audit
"""

from __future__ import annotations

from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass
class SupervisorState:
    """Mutable working memory passed through the LangGraph loop.

    Every node reads/writes this state so the Supervisor can maintain
    context across plan → execute → reflect iterations.
    """

    # ── User request ────────────────────────────────
    user_message: str = ""
    session_id: str = ""
    applicant_id: Optional[str] = None

    # ── Conversation history (role-tagged dicts) ────
    messages: list[dict[str, Any]] = field(default_factory=list)

    # ── Current plan step ───────────────────────────
    current_plan: Optional[dict[str, Any]] = None  # serialised PlanStep
    current_tool_result: Optional[dict[str, Any]] = None

    # ── Accumulated tool outputs (short-term) ───────
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)

    # ── Long-term compressed memory ─────────────────
    #    After every ``memory_summarise_interval`` tool calls the
    #    supervisor compresses older outputs into a prose summary
    #    so the context window stays under budget.
    long_term_memory: str = ""
    memory_summarise_interval: int = 5

    # ── Goal / sub-goal stack ───────────────────────
    #    Enables multi-goal planning where the supervisor can
    #    push sub-goals and pop them as they are resolved.
    goal_stack: list[str] = field(default_factory=list)

    # ── Write-plan confirmation state ───────────────
    pending_write_plan_id: Optional[str] = None
    confirmation_token: Optional[str] = None

    # ── Token / cost budget tracking ────────────────
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens_used: int = 0
    estimated_cost_usd: float = 0.0
    token_budget: int = 120_000       # overridden from settings at init

    # ── Loop control ────────────────────────────────
    iteration: int = 0
    max_iterations: int = 15
    should_stop: bool = False
    stop_reason: str = ""             # human-readable reason for stopping
    final_answer: str = ""

    # ── Trace log ───────────────────────────────────
    trace: list[dict[str, Any]] = field(default_factory=list)

    # ── Error recovery ──────────────────────────────
    last_error: Optional[str] = None
    consecutive_errors: int = 0
    max_consecutive_errors: int = 3
    failed_tools: list[str] = field(default_factory=list)  # tools that errored

    # ── Quality signals ─────────────────────────────
    confidence_scores: dict[str, float] = field(default_factory=dict)
    #    tool_name → last confidence (0-1) reported by the tool

    # ── Parallel execution hints ────────────────────
    pending_parallel_calls: list[dict[str, Any]] = field(default_factory=list)

    # ═══════════════════════════════════════════════
    #  Helper properties
    # ═══════════════════════════════════════════════

    @property
    def is_over_budget(self) -> bool:
        """True when accumulated tokens exceed the session budget."""
        return self.total_tokens_used > self.token_budget

    @property
    def budget_remaining_pct(self) -> float:
        """Percentage of the token budget remaining."""
        if self.token_budget <= 0:
            return 0.0
        return max(0.0, 1.0 - self.total_tokens_used / self.token_budget) * 100

    @property
    def tools_used(self) -> list[str]:
        """Ordered list of tool names invoked so far."""
        return [o["tool_name"] for o in self.tool_outputs]

    @property
    def needs_memory_compression(self) -> bool:
        """True when short-term memory should be summarised."""
        return (
            len(self.tool_outputs) > 0
            and len(self.tool_outputs) % self.memory_summarise_interval == 0
        )
