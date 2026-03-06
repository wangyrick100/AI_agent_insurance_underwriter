"""Pydantic schemas for the Supervisor ReAct loop & tool I/O contracts."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════
#  Supervisor ReAct step schema
# ═══════════════════════════════════════════════


class ToolName(str, Enum):
    x1_extract = "x1_extract"
    x2_score = "x2_score"
    x3_web = "x3_web"
    x4_sql_read = "x4_sql_read"
    x5_write_plan = "x5_write_plan"
    x5_write_commit = "x5_write_commit"
    x6_rag = "x6_rag"
    synthesize = "synthesize"
    ask_user = "ask_user"


class PlanStep(BaseModel):
    """Output of the Supervisor's plan-and-act LLM call."""

    thought_summary: str = Field(
        ..., max_length=300, description="Short reasoning (no chain-of-thought dump)"
    )
    next_tool: ToolName
    tool_input: dict[str, Any] = Field(default_factory=dict)
    user_message: Optional[str] = Field(
        None, description="Only populated when next_tool == ask_user"
    )
    stop: bool = False


class ReflectDecision(BaseModel):
    """Output of the reflect node after a tool execution."""

    observation_summary: str = Field(..., max_length=500)
    should_continue: bool = True
    revised_goal: Optional[str] = None


# ═══════════════════════════════════════════════
#  Tool input / output contracts
# ═══════════════════════════════════════════════


# ── X1 Extraction ────────────────────────────
class ExtractionEntity(BaseModel):
    entity_type: str  # medication, diagnosis, lab_result, vital, procedure, demographic
    entity_name: str
    entity_value: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_chunk_id: Optional[str] = None
    evidence_snippet: Optional[str] = None


class ExtractionBundle(BaseModel):
    applicant_id: Optional[str] = None
    doc_ids: list[str] = Field(default_factory=list)
    entities: list[ExtractionEntity] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    evidence_map: dict[str, str] = Field(
        default_factory=dict, description="field_name -> chunk_id"
    )
    disclaimers: list[str] = Field(
        default_factory=lambda: ["Decision support only. Not a final underwriting decision."]
    )


class X1Input(BaseModel):
    doc_ids: list[str] = Field(default_factory=list)
    raw_text: Optional[str] = None
    applicant_id: Optional[str] = None


# ── X2 Risk Scoring ─────────────────────────
class SimilarCase(BaseModel):
    case_id: str
    similarity: float
    risk_class: str
    reason: str


class RiskScoreBundle(BaseModel):
    applicant_id: str
    score: float = Field(ge=0.0, le=100.0)
    confidence: float = Field(ge=0.0, le=1.0)
    risk_class: str  # preferred, standard, substandard, decline
    feature_rationale: dict[str, float] = Field(default_factory=dict)
    similar_cases: list[SimilarCase] = Field(default_factory=list)
    model_version: str = "v1.0"
    disclaimers: list[str] = Field(
        default_factory=lambda: [
            "ML risk score is advisory. Final decision must be made by a licensed underwriter."
        ]
    )


class X2Input(BaseModel):
    applicant_id: Optional[str] = None
    feature_payload: Optional[dict[str, Any]] = None


# ── X3 Web Research ──────────────────────────
class WebSource(BaseModel):
    url: str
    title: str
    snippet: str


class UnderwriterBrief(BaseModel):
    query: str
    summary: str
    sources: list[WebSource] = Field(default_factory=list)
    disclaimers: list[str] = Field(
        default_factory=lambda: ["Research summary only. Verify with primary sources."]
    )


class X3Input(BaseModel):
    query: str
    topic_scope: Optional[str] = None


# ── X4 SQL Read ──────────────────────────────
class SQLResult(BaseModel):
    query: str
    db_id: str = "primary"
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = 0
    disclaimers: list[str] = Field(default_factory=list)


class X4Input(BaseModel):
    question: str
    db_id: Optional[str] = None
    constraints: Optional[dict[str, Any]] = None


# ── X5 DB Write ──────────────────────────────
class WritePlanSchema(BaseModel):
    plan_id: str
    applicant_id: str
    sql_statements: list[str]
    diff_preview: dict[str, Any] = Field(default_factory=dict)
    impacted_tables: list[str] = Field(default_factory=list)
    impacted_row_count: int = 0
    confirmation_token: str
    status: str = "pending"


class CommitResult(BaseModel):
    plan_id: str
    status: str  # committed, failed
    rows_affected: int = 0
    error_message: Optional[str] = None


class X5PlanInput(BaseModel):
    extraction_bundle: Optional[dict[str, Any]] = None
    applicant_id: str


class X5CommitInput(BaseModel):
    write_plan_id: str
    confirmation_token: str


# ── X6 RAG ───────────────────────────────────
class RAGCitation(BaseModel):
    doc_id: str
    chunk_id: str
    page: Optional[int] = None
    snippet: str
    relevance_score: float = 0.0


class RAGAnswerWithCitations(BaseModel):
    query: str
    answer: str
    citations: list[RAGCitation] = Field(default_factory=list)
    disclaimers: list[str] = Field(
        default_factory=lambda: ["Answer generated from indexed documents. Verify with originals."]
    )


class X6Input(BaseModel):
    query: str
    applicant_id: Optional[str] = None
    doc_scope: Optional[list[str]] = None


# ═══════════════════════════════════════════════
#  Session / trace schemas
# ═══════════════════════════════════════════════


class TraceEntry(BaseModel):
    step_index: int
    step_type: str
    tool_name: Optional[str] = None
    tool_input_summary: Optional[str] = None
    tool_output_summary: Optional[str] = None
    thought_summary: Optional[str] = None
    success: Optional[bool] = None
    duration_ms: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SessionTraceSchema(BaseModel):
    session_id: str
    steps: list[TraceEntry] = Field(default_factory=list)


# ═══════════════════════════════════════════════
#  API request / response schemas
# ═══════════════════════════════════════════════


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    applicant_id: Optional[str] = None
    confirmation_token: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    pending_write_plan: Optional[WritePlanSchema] = None
    trace: Optional[SessionTraceSchema] = None


class IngestRequest(BaseModel):
    applicant_id: str
    doc_type: str = "unknown"


class ApplicantSummary(BaseModel):
    applicant_id: str
    first_name: str
    last_name: str
    status: str
    documents: list[dict[str, Any]] = Field(default_factory=list)
    extractions: list[dict[str, Any]] = Field(default_factory=list)
    latest_score: Optional[dict[str, Any]] = None
    pending_write_plans: list[dict[str, Any]] = Field(default_factory=list)
