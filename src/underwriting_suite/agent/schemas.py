"""Pydantic schemas for the Supervisor ReAct loop & tool I/O contracts.

Every agent (X1–X6) has a strict input / output schema so that:
  • the Supervisor can reason about tool capabilities at planning time;
  • callers get predictable, validated payloads;
  • audit and traceability fields travel with every result.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════
#  Supervisor ReAct step schemas
# ═══════════════════════════════════════════════


class ToolName(str, Enum):
    skill_extraction = "skill_extraction"
    skill_risk_model = "skill_risk_model"
    skill_web_research = "skill_web_research"
    skill_sql_read = "skill_sql_read"
    skill_db_write_plan = "skill_db_write_plan"
    skill_db_write_commit = "skill_db_write_commit"
    skill_rag = "skill_rag"
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
    priority: int = Field(
        default=5, ge=1, le=10,
        description="Urgency hint for the supervisor scheduler (1=low, 10=critical)",
    )
    expected_output_fields: list[str] = Field(
        default_factory=list,
        description="Fields the planner expects to receive from this tool call",
    )
    stop: bool = False


class ReflectDecision(BaseModel):
    """Output of the reflect node after a tool execution."""

    observation_summary: str = Field(..., max_length=500)
    should_continue: bool = True
    revised_goal: Optional[str] = None
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Supervisor's self-assessed confidence in progress so far",
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Information gaps that remain after this iteration",
    )


# ═══════════════════════════════════════════════
#  Tool input / output contracts
# ═══════════════════════════════════════════════


# ── X1 Extraction ────────────────────────────

class NormalisedCode(BaseModel):
    """Standard medical code resolved during entity normalisation."""
    system: str = Field(description="Coding system: ICD-10 | RxNorm | SNOMED-CT | LOINC | CPT")
    code: str
    display: str = ""


class ExtractionEntity(BaseModel):
    entity_type: str  # medication, diagnosis, lab_result, vital, procedure, demographic
    entity_name: str
    entity_value: Optional[str] = None
    unit: Optional[str] = Field(None, description="Measurement unit (mg, mmHg, mg/dL, …)")
    confidence: float = Field(ge=0.0, le=1.0)
    normalised_codes: list[NormalisedCode] = Field(
        default_factory=list,
        description="ICD-10/RxNorm/SNOMED/LOINC codes resolved for this entity",
    )
    is_negated: bool = Field(
        default=False, description="True if the entity was negated in context ('no diabetes')"
    )
    temporality: Optional[str] = Field(
        None,
        description="Temporal qualifier: current | historical | family_history | planned",
    )
    evidence_chunk_id: Optional[str] = None
    evidence_snippet: Optional[str] = None
    evidence_page: Optional[int] = None
    source_doc_id: Optional[str] = None


class ExtractionConflict(BaseModel):
    """Records a contradiction between two extracted entities."""
    entity_a: str
    entity_b: str
    conflict_type: str = Field(
        description="Type: value_mismatch | temporal_contradiction | contradictory_negation"
    )
    description: str = ""
    resolution_suggestion: Optional[str] = None


class ExtractionBundle(BaseModel):
    applicant_id: Optional[str] = None
    doc_ids: list[str] = Field(default_factory=list)
    entities: list[ExtractionEntity] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    conflicts: list[ExtractionConflict] = Field(
        default_factory=list,
        description="Cross-document or intra-document contradictions detected",
    )
    evidence_map: dict[str, str] = Field(
        default_factory=dict, description="field_name -> chunk_id"
    )
    extraction_passes: int = Field(
        default=1, description="Number of LLM passes used (1 = single, 2 = verified)"
    )
    entity_count_by_type: dict[str, int] = Field(
        default_factory=dict, description="Counts per entity_type for quick summary"
    )
    disclaimers: list[str] = Field(
        default_factory=lambda: ["Decision support only. Not a final underwriting decision."]
    )


class X1Input(BaseModel):
    doc_ids: list[str] = Field(default_factory=list)
    raw_text: Optional[str] = None
    applicant_id: Optional[str] = None
    extract_normalised_codes: bool = Field(
        default=True,
        description="If True, attempt ICD-10/RxNorm/SNOMED normalisation for each entity",
    )


# ── X2 Risk Scoring ─────────────────────────

class FeatureImportance(BaseModel):
    """SHAP-style feature importance entry."""
    feature: str
    value: Any = None
    contribution: float = Field(description="Signed contribution to the risk score")
    direction: str = Field(default="neutral", description="increases_risk | decreases_risk | neutral")


class SimilarCase(BaseModel):
    case_id: str
    similarity: float
    risk_class: str
    reason: str
    key_features: dict[str, Any] = Field(
        default_factory=dict, description="Feature values that drove the match"
    )


class MortalityRating(BaseModel):
    """Insurance-specific mortality/morbidity rating."""
    table_rating: Optional[str] = Field(
        None, description="Table rating letter (A-P) or percentage (e.g. +50%)"
    )
    flat_extra: Optional[float] = Field(
        None, description="Flat extra premium per $1 000 of coverage"
    )
    duration_years: Optional[int] = Field(
        None, description="Duration for temporary ratings"
    )


class RiskScoreBundle(BaseModel):
    applicant_id: str
    score: float = Field(ge=0.0, le=100.0)
    confidence: float = Field(ge=0.0, le=1.0)
    risk_class: str  # preferred_plus, preferred, standard_plus, standard, substandard, decline
    sub_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Component sub-scores: medical, lifestyle, financial, occupational",
    )
    feature_importance: list[FeatureImportance] = Field(
        default_factory=list, description="Ranked feature contributions (SHAP-style)"
    )
    feature_rationale: dict[str, float] = Field(default_factory=dict)
    mortality_rating: Optional[MortalityRating] = None
    similar_cases: list[SimilarCase] = Field(default_factory=list)
    model_version: str = "v2.0"
    scoring_method: str = Field(
        default="ensemble",
        description="Method used: llm_only | rule_based | ensemble | ml_model",
    )
    calibration_note: Optional[str] = None
    disclaimers: list[str] = Field(
        default_factory=lambda: [
            "ML risk score is advisory. Final decision must be made by a licensed underwriter."
        ]
    )


class X2Input(BaseModel):
    applicant_id: Optional[str] = None
    feature_payload: Optional[dict[str, Any]] = None
    extraction_bundle: Optional[dict[str, Any]] = Field(
        None, description="Raw X1 extraction output – features will be engineered from it"
    )
    scoring_method: Optional[str] = Field(
        None, description="Override: llm_only | rule_based | ensemble"
    )


# ── X3 Web Research ──────────────────────────

class WebSource(BaseModel):
    url: str
    title: str
    snippet: str
    domain: str = ""
    credibility_tier: str = Field(
        default="tier_2",
        description="Source credibility: tier_1 (peer-reviewed), tier_2 (authoritative), tier_3 (general)"
    )
    retrieved_at: Optional[datetime] = None


class UnderwriterBrief(BaseModel):
    query: str
    summary: str
    key_findings: list[str] = Field(
        default_factory=list,
        description="Bullet-point key findings for quick scanning",
    )
    sources: list[WebSource] = Field(default_factory=list)
    search_strategy: str = Field(
        default="", description="Description of the search approach used"
    )
    source_agreement: Optional[str] = Field(
        None, description="Assessment of consistency across sources: consistent | mixed | contradictory"
    )
    disclaimers: list[str] = Field(
        default_factory=lambda: ["Research summary only. Verify with primary sources."]
    )


class X3Input(BaseModel):
    query: str
    topic_scope: Optional[str] = None
    max_sources: int = Field(default=8, ge=1, le=20)
    require_tier1: bool = Field(
        default=False,
        description="If True, only include tier-1 (peer-reviewed) sources",
    )


# ── X4 SQL Read ──────────────────────────────

class QueryPlan(BaseModel):
    """LLM-generated query with self-review metadata."""
    sql: str
    explanation: str = ""
    complexity: str = Field(
        default="simple", description="simple | moderate | complex | multi_join"
    )
    estimated_cost: Optional[str] = None
    review_notes: list[str] = Field(
        default_factory=list,
        description="Self-review notes (e.g. 'missing index hint', 'subquery alternative')"
    )


class SQLResult(BaseModel):
    query: str
    db_id: str = "primary"
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = Field(
        default=False, description="True if results were truncated to the row cap"
    )
    query_plan: Optional[QueryPlan] = None
    execution_time_ms: Optional[float] = None
    self_correction_attempts: int = Field(
        default=0, description="Number of SQL correction rounds performed"
    )
    disclaimers: list[str] = Field(default_factory=list)


class X4Input(BaseModel):
    question: str
    db_id: Optional[str] = None
    constraints: Optional[dict[str, Any]] = None
    max_rows: int = Field(default=500, ge=1, le=5000)
    enable_self_correction: bool = Field(
        default=True,
        description="Allow the agent to fix SQL errors and re-execute up to N times",
    )


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
    expires_at: Optional[datetime] = Field(
        None, description="UTC timestamp when the confirmation token expires"
    )
    rollback_statements: list[str] = Field(
        default_factory=list,
        description="SQL statements to reverse the write plan if needed",
    )
    risk_level: str = Field(
        default="low",
        description="Assessed risk: low | medium | high | critical",
    )
    validation_checks: list[str] = Field(
        default_factory=list,
        description="Pre-flight validation checks performed on the plan",
    )


class CommitResult(BaseModel):
    plan_id: str
    status: str  # committed, rolled_back, failed, expired
    rows_affected: int = 0
    error_message: Optional[str] = None
    committed_at: Optional[datetime] = None
    audit_id: Optional[str] = Field(
        None, description="Unique audit trail identifier for this commit"
    )


class X5PlanInput(BaseModel):
    extraction_bundle: Optional[dict[str, Any]] = None
    applicant_id: str
    dry_run: bool = Field(
        default=False,
        description="If True, generate the plan but mark as dry_run (no commit allowed)",
    )


class X5CommitInput(BaseModel):
    write_plan_id: str
    confirmation_token: str
    confirmed_by: Optional[str] = Field(
        None, description="Identity of the human approver (for audit trail)"
    )


# ── X6 RAG ───────────────────────────────────

class RAGCitation(BaseModel):
    doc_id: str
    chunk_id: str
    page: Optional[int] = None
    snippet: str
    relevance_score: float = 0.0
    rerank_score: Optional[float] = Field(
        None, description="Score from the cross-encoder re-ranking pass"
    )


class RAGAnswerWithCitations(BaseModel):
    query: str
    answer: str
    citations: list[RAGCitation] = Field(default_factory=list)
    sub_questions: list[str] = Field(
        default_factory=list,
        description="Decomposed sub-questions if complex query decomposition was used",
    )
    answer_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="LLM's self-assessed confidence in the generated answer"
    )
    retrieval_strategy: str = Field(
        default="hybrid", description="sparse | dense | hybrid | hybrid+rerank"
    )
    chunks_retrieved: int = Field(default=0, description="Total chunks before re-ranking")
    chunks_after_rerank: int = Field(default=0, description="Chunks after re-ranking filter")
    disclaimers: list[str] = Field(
        default_factory=lambda: ["Answer generated from indexed documents. Verify with originals."]
    )


class X6Input(BaseModel):
    query: str
    applicant_id: Optional[str] = None
    doc_scope: Optional[list[str]] = None
    top_k: int = Field(default=8, ge=1, le=50)
    enable_query_decomposition: bool = Field(
        default=True,
        description="Decompose complex questions into sub-queries for better retrieval",
    )


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
    tokens_used: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SessionTraceSchema(BaseModel):
    session_id: str
    steps: list[TraceEntry] = Field(default_factory=list)
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    total_duration_ms: float = 0.0


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
    usage: Optional[dict[str, Any]] = Field(
        None,
        description="Token usage and cost summary for this session turn",
    )


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
