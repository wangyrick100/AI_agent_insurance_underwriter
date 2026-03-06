"""Tool registry – exposes all X-agents as callable tools to the Supervisor.

Each tool has:
  - a stable async callable
  - a JSON schema describing its input
  - metadata for the Supervisor's tool-selection prompt
"""

from __future__ import annotations

from typing import Any, Callable, Coroutine

from underwriting_suite.agent.tools.x1_extraction import x1_extract
from underwriting_suite.agent.tools.x2_risk_model import x2_score
from underwriting_suite.agent.tools.x3_web_research import x3_web
from underwriting_suite.agent.tools.x4_sql_read import x4_sql_read
from underwriting_suite.agent.tools.x5_db_write import x5_write_commit, x5_write_plan
from underwriting_suite.agent.tools.x6_rag import x6_rag

ToolCallable = Callable[[dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]


_TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "x1_extract": {
        "fn": x1_extract,
        "description": (
            "Extract medical/underwriting entities (medications, diagnoses, labs, vitals, "
            "procedures, demographics) from uploaded documents or raw text."
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
    },
    "x2_score": {
        "fn": x2_score,
        "description": (
            "Compute ML risk score with confidence, feature rationale, and similar-case "
            "retrieval for an applicant."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "applicant_id": {"type": "string"},
                "feature_payload": {
                    "type": "object",
                    "description": "Pre-built feature dictionary (optional).",
                },
            },
        },
    },
    "x3_web": {
        "fn": x3_web,
        "description": (
            "Perform restricted web research on underwriting-relevant topics "
            "using only allowlisted medical/regulatory domains."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Research question."},
                "topic_scope": {"type": "string", "description": "Topic narrower (optional)."},
            },
            "required": ["query"],
        },
    },
    "x4_sql_read": {
        "fn": x4_sql_read,
        "description": (
            "Convert a natural-language question to a SELECT-only SQL query "
            "and execute it against the underwriting database."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Natural-language question."},
                "db_id": {"type": "string", "description": "Database identifier (default: primary)."},
                "constraints": {"type": "object", "description": "Extra constraints (optional)."},
            },
            "required": ["question"],
        },
    },
    "x5_write_plan": {
        "fn": x5_write_plan,
        "description": (
            "Generate a safe write plan (INSERT/UPDATE SQL) from extracted entities. "
            "Returns a plan with confirmation token – does NOT execute until confirmed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "extraction_bundle": {"type": "object", "description": "Extraction output."},
                "applicant_id": {"type": "string"},
            },
            "required": ["applicant_id"],
        },
    },
    "x5_write_commit": {
        "fn": x5_write_commit,
        "description": (
            "Commit a previously generated write plan. REQUIRES a valid "
            "confirmation_token from the user to execute."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "write_plan_id": {"type": "string"},
                "confirmation_token": {"type": "string"},
            },
            "required": ["write_plan_id", "confirmation_token"],
        },
    },
    "x6_rag": {
        "fn": x6_rag,
        "description": (
            "RAG Q&A with citations over ingested underwriting documents using "
            "Azure AI Search vector index."
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
            },
            "required": ["query"],
        },
    },
}


def get_tool(name: str) -> ToolCallable | None:
    """Get a tool's callable by name."""
    entry = _TOOL_REGISTRY.get(name)
    return entry["fn"] if entry else None


def get_tool_schemas() -> dict[str, dict[str, Any]]:
    """Return all tool schemas for the Supervisor's system prompt."""
    return {
        name: {"description": entry["description"], "input_schema": entry["input_schema"]}
        for name, entry in _TOOL_REGISTRY.items()
    }


def list_tool_names() -> list[str]:
    """Return all registered tool names."""
    return list(_TOOL_REGISTRY.keys())
