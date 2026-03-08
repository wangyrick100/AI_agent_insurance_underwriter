"""Tool registry – exposes all X-agents as callable tools to the Supervisor.

Each tool has:
  - a stable async callable
  - a JSON schema describing its input
  - metadata for the Supervisor's tool-selection prompt
  - version identifier for tracking behaviour changes
  - estimated cost tier (low / medium / high) for budget-aware planning
  - health status tracking for circuit-breaker style degradation
  - category tags for structured tool discovery
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

from underwriting_suite.agent.skills.skill_extraction import skill_extraction
from underwriting_suite.agent.skills.skill_risk_model import skill_risk_model
from underwriting_suite.agent.skills.skill_web_research import skill_web_research
from underwriting_suite.agent.skills.skill_sql_read import skill_sql_read
from underwriting_suite.agent.skills.skill_db_write import skill_db_write_commit, skill_db_write_plan
from underwriting_suite.agent.skills.skill_rag import skill_rag

logger = logging.getLogger(__name__)

ToolCallable = Callable[[dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]


# ═══════════════════════════════════════════════
#  Tool metadata types
# ═══════════════════════════════════════════════

class CostTier(str, Enum):
    """Estimated LLM token cost per invocation."""
    low = "low"          # < 2k tokens
    medium = "medium"    # 2k – 10k tokens
    high = "high"        # > 10k tokens


class ToolCategory(str, Enum):
    """Functional category for tool grouping."""
    extraction = "extraction"
    scoring = "scoring"
    research = "research"
    database = "database"
    retrieval = "retrieval"


class HealthStatus(str, Enum):
    """Runtime health state of a tool."""
    healthy = "healthy"
    degraded = "degraded"
    unavailable = "unavailable"


@dataclass
class ToolHealth:
    """Tracks runtime health metrics for a registered tool."""
    status: HealthStatus = HealthStatus.healthy
    total_calls: int = 0
    total_failures: int = 0
    consecutive_failures: int = 0
    last_call_time: float | None = None
    last_error: str | None = None
    avg_duration_ms: float = 0.0
    _durations: list[float] = field(default_factory=list)

    def record_success(self, duration_ms: float) -> None:
        self.total_calls += 1
        self.consecutive_failures = 0
        self.last_call_time = time.time()
        self._durations.append(duration_ms)
        if len(self._durations) > 50:
            self._durations = self._durations[-50:]
        self.avg_duration_ms = sum(self._durations) / len(self._durations)
        self.status = HealthStatus.healthy

    def record_failure(self, error: str) -> None:
        self.total_calls += 1
        self.total_failures += 1
        self.consecutive_failures += 1
        self.last_call_time = time.time()
        self.last_error = error
        if self.consecutive_failures >= 5:
            self.status = HealthStatus.unavailable
        elif self.consecutive_failures >= 2:
            self.status = HealthStatus.degraded

    @property
    def failure_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_failures / self.total_calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "failure_rate": round(self.failure_rate, 3),
            "avg_duration_ms": round(self.avg_duration_ms, 1),
            "last_error": self.last_error,
        }


# ═══════════════════════════════════════════════
#  Registry entries
# ═══════════════════════════════════════════════

_TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "skill_extraction": {
        "fn": skill_extraction,
        "version": "2.0.0",
        "category": ToolCategory.extraction,
        "cost_tier": CostTier.high,
        "description": (
            "Extract medical/underwriting entities (medications, diagnoses, labs, vitals, "
            "procedures, demographics, family history, lifestyle) from uploaded documents "
            "or raw text. Supports multi-pass extraction with verification, entity "
            "normalisation (ICD-10/RxNorm/SNOMED), and conflict detection."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "doc_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of document IDs to extract from.",
                },
                "raw_text": {
                    "type": "string",
                    "description": "Raw document text (alternative to doc_ids).",
                },
                "applicant_id": {
                    "type": "string",
                    "description": "Applicant identifier.",
                },
            },
        },
        "health": ToolHealth(),
    },
    "skill_risk_model": {
        "fn": skill_risk_model,
        "version": "2.0.0",
        "category": ToolCategory.scoring,
        "cost_tier": CostTier.medium,
        "description": (
            "Compute risk score using ensemble scoring (LLM + rule-based) with "
            "SHAP-style feature importance, mortality rating, sub-scores "
            "(medical/lifestyle/financial/occupational), and 6-tier risk classification."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "applicant_id": {"type": "string"},
                "feature_payload": {
                    "type": "object",
                    "description": "Pre-built feature dictionary (optional).",
                },
                "extraction_bundle": {
                    "type": "object",
                    "description": "X1 extraction bundle for feature engineering.",
                },
                "scoring_method": {
                    "type": "string",
                    "enum": ["llm_only", "rule_based", "ensemble"],
                    "description": "Scoring method override (default: ensemble).",
                },
            },
        },
        "health": ToolHealth(),
    },
    "skill_web_research": {
        "fn": skill_web_research,
        "version": "2.0.0",
        "category": ToolCategory.research,
        "cost_tier": CostTier.medium,
        "description": (
            "Perform restricted web research on underwriting-relevant topics "
            "using allowlisted medical/regulatory domains. Supports Bing Search API, "
            "query expansion, source credibility tiering, and result caching."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Research question."},
                "topic_scope": {"type": "string", "description": "Topic narrower (optional)."},
                "max_sources": {"type": "integer", "description": "Max sources to include (default: 6)."},
                "require_tier1": {"type": "boolean", "description": "Require peer-reviewed sources."},
            },
            "required": ["query"],
        },
        "health": ToolHealth(),
    },
    "skill_sql_read": {
        "fn": skill_sql_read,
        "version": "2.0.0",
        "category": ToolCategory.database,
        "cost_tier": CostTier.medium,
        "description": (
            "Convert a natural-language question to a SELECT-only SQL query "
            "and execute it against the underwriting database. Supports multi-turn "
            "self-correction, LLM query review, and complexity analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Natural-language question."},
                "db_id": {"type": "string", "description": "Database identifier (default: primary)."},
                "constraints": {"type": "object", "description": "Extra constraints (optional)."},
                "max_rows": {"type": "integer", "description": "Maximum rows to return."},
                "enable_self_correction": {"type": "boolean", "description": "Enable multi-turn correction."},
            },
            "required": ["question"],
        },
        "health": ToolHealth(),
    },
    "skill_db_write_plan": {
        "fn": x5_write_plan,
        "version": "2.0.0",
        "category": ToolCategory.database,
        "cost_tier": CostTier.medium,
        "description": (
            "Generate a safe write plan (INSERT/UPDATE SQL) from extracted entities. "
            "Returns a plan with confirmation token, rollback statements, risk level, "
            "and validation checks. Does NOT execute until confirmed via skill_db_write_commit."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "extraction_bundle": {"type": "object", "description": "Extraction output."},
                "applicant_id": {"type": "string"},
                "dry_run": {"type": "boolean", "description": "Generate plan without allowing commit."},
            },
            "required": ["applicant_id"],
        },
        "health": ToolHealth(),
    },
    "skill_db_write_commit": {
        "fn": x5_write_commit,
        "version": "2.0.0",
        "category": ToolCategory.database,
        "cost_tier": CostTier.low,
        "description": (
            "Commit a previously generated write plan. REQUIRES a valid "
            "confirmation_token from the user to execute. Checks for token expiry "
            "and concurrent conflicts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "write_plan_id": {"type": "string"},
                "confirmation_token": {"type": "string"},
                "confirmed_by": {"type": "string", "description": "Identity of confirming user."},
            },
            "required": ["write_plan_id", "confirmation_token"],
        },
        "health": ToolHealth(),
    },
    "skill_rag": {
        "fn": x6_rag,
        "version": "2.0.0",
        "category": ToolCategory.retrieval,
        "cost_tier": CostTier.high,
        "description": (
            "Advanced RAG Q&A with citations over ingested underwriting documents. "
            "Supports query decomposition, hybrid retrieval, cross-encoder re-ranking, "
            "answer verification, and confidence self-assessment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Question to answer."},
                "applicant_id": {"type": "string", "description": "Scope to applicant docs."},
                "doc_scope": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific doc IDs to search within.",
                },
                "top_k": {"type": "integer", "description": "Number of chunks to retrieve."},
                "enable_query_decomposition": {
                    "type": "boolean",
                    "description": "Decompose complex questions into sub-questions.",
                },
            },
            "required": ["query"],
        },
        "health": ToolHealth(),
    },
}


# ═══════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════

def get_tool(name: str) -> ToolCallable | None:
    """Get a tool's callable by name.

    Returns None if tool is not registered or is currently unavailable
    (health status = unavailable).
    """
    entry = _TOOL_REGISTRY.get(name)
    if not entry:
        return None
    health: ToolHealth = entry.get("health")
    if health and health.status == HealthStatus.unavailable:
        logger.warning("Tool %s is unavailable (consecutive failures: %d)", name, health.consecutive_failures)
        return None
    return entry["fn"]


def get_tool_schemas() -> dict[str, dict[str, Any]]:
    """Return all tool schemas for the Supervisor's system prompt.

    Includes version, cost tier, and category metadata to help the
    Supervisor make budget-aware tool selections.
    """
    return {
        name: {
            "description": entry["description"],
            "input_schema": entry["input_schema"],
            "version": entry.get("version", "1.0.0"),
            "cost_tier": entry.get("cost_tier", CostTier.medium).value,
            "category": entry.get("category", "general").value
            if hasattr(entry.get("category"), "value")
            else str(entry.get("category", "general")),
        }
        for name, entry in _TOOL_REGISTRY.items()
    }


def list_tool_names() -> list[str]:
    """Return all registered tool names."""
    return list(_TOOL_REGISTRY.keys())


def record_tool_success(name: str, duration_ms: float) -> None:
    """Record a successful tool execution for health tracking."""
    entry = _TOOL_REGISTRY.get(name)
    if entry and "health" in entry:
        entry["health"].record_success(duration_ms)


def record_tool_failure(name: str, error: str) -> None:
    """Record a tool execution failure for health tracking."""
    entry = _TOOL_REGISTRY.get(name)
    if entry and "health" in entry:
        entry["health"].record_failure(error)
        logger.warning(
            "Tool %s failure recorded | consecutive=%d status=%s",
            name, entry["health"].consecutive_failures, entry["health"].status.value,
        )


def get_tool_health(name: str) -> dict[str, Any] | None:
    """Get health metrics for a specific tool."""
    entry = _TOOL_REGISTRY.get(name)
    if entry and "health" in entry:
        return entry["health"].to_dict()
    return None


def get_all_tool_health() -> dict[str, dict[str, Any]]:
    """Get health metrics for all registered tools."""
    return {
        name: entry["health"].to_dict()
        for name, entry in _TOOL_REGISTRY.items()
        if "health" in entry
    }


def get_tools_by_category(category: ToolCategory) -> list[str]:
    """Get tool names filtered by category."""
    return [
        name for name, entry in _TOOL_REGISTRY.items()
        if entry.get("category") == category
    ]


def get_tools_by_cost_tier(tier: CostTier) -> list[str]:
    """Get tool names filtered by cost tier."""
    return [
        name for name, entry in _TOOL_REGISTRY.items()
        if entry.get("cost_tier") == tier
    ]


def reset_tool_health(name: str) -> bool:
    """Reset a tool's health metrics (e.g., after a fix is deployed)."""
    entry = _TOOL_REGISTRY.get(name)
    if entry and "health" in entry:
        entry["health"] = ToolHealth()
        logger.info("Tool %s health metrics reset", name)
        return True
    return False
