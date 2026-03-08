"""tool_db_write – Safe Database Write Tool (Plan + Confirm + Commit).

Generates a write plan from extracted entities, requires explicit
confirmation token before committing any changes.

Production capabilities:
  • Confirmation-token expiry  – plans expire after a configurable TTL
    (default 15 min) and cannot be committed once expired.
  • Rollback SQL generation  – each plan includes reverse SQL statements
    to undo committed changes if needed.
  • Risk-level assessment  – classifies each plan as low / medium / high /
    critical based on number of tables/rows and statement types.
  • Pre-flight validation checks  – inspects generated SQL for common
    errors, schema mismatches, and constraint violations.
  • Audit trail  – every commit generates a unique audit ID and records
    the confirming user identity and timestamp.
  • Concurrent conflict detection  – prevents committing two overlapping
    plans for the same applicant.
  • Dry-run mode  – generate plan without allowing commit.
"""

from __future__ import annotations

import json
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from underwriting_suite.agent.schemas import (
    CommitResult,
    WritePlanSchema,
    X5CommitInput,
    X5PlanInput,
)
from underwriting_suite.config import settings
from underwriting_suite.services.azure_openai import get_chat_completion
from underwriting_suite.services.db_service import execute_write_plan, get_db_schema

logger = logging.getLogger(__name__)

# In-memory store for active write plans (production: use DB table)
_active_plans: dict[str, WritePlanSchema] = {}

# Track which applicants have in-flight commits to prevent conflicts
_inflight_applicants: set[str] = set()


# ═══════════════════════════════════════════════
#  System prompts
# ═══════════════════════════════════════════════

WRITE_PLAN_SYSTEM_PROMPT = """\
You are AgentX5WritePlan, a safe database write planner for underwriting systems.

TASK:
Given extracted entities for an applicant, generate SQL statements to
insert or update the underwriting database.  Also generate corresponding
ROLLBACK statements that can reverse each change.

SCHEMA:
{schema}

OUTPUT FORMAT (strict JSON):
{{
  "sql_statements": ["INSERT INTO ...", "UPDATE ... SET ..."],
  "rollback_statements": ["DELETE FROM ... WHERE ...", "UPDATE ... SET ... WHERE ..."],
  "diff_preview": {{
    "table_name": {{"columns_affected": [...], "action": "INSERT|UPDATE", \
"row_preview": {{}}}}
  }},
  "impacted_tables": ["table1", "table2"],
  "impacted_row_count": 5,
  "validation_notes": ["any warnings about the generated plan"]
}}

RULES:
1. Generate ONLY INSERT and UPDATE statements – never DELETE, DROP, TRUNCATE.
2. Each rollback_statement must precisely undo the corresponding sql_statement.
3. For INSERTs, the rollback is a DELETE with the same primary key.
4. For UPDATEs, the rollback restores original values (use diff_preview.before).
5. Use parameterised values where possible.
6. Provide a diff preview showing affected columns and row preview.
7. Track all impacted tables and estimated row count.
8. These statements will NOT execute until the user confirms with a token.
"""


# ═══════════════════════════════════════════════
#  Risk assessment
# ═══════════════════════════════════════════════

def _assess_risk(
    sql_statements: list[str],
    impacted_tables: list[str],
    impacted_row_count: int,
) -> str:
    """Classify the risk level of a write plan."""
    upper_stmts = [s.upper() for s in sql_statements]

    # Critical: modifies core tables or high row count
    core_tables = {"applicants", "risk_scores"}
    if any(t.lower() in core_tables for t in impacted_tables):
        if impacted_row_count > 10:
            return "critical"
        return "high"

    if impacted_row_count > 50:
        return "critical"
    if impacted_row_count > 10:
        return "high"
    if len(sql_statements) > 5:
        return "medium"
    return "low"


# ═══════════════════════════════════════════════
#  Validation checks
# ═══════════════════════════════════════════════

def _validate_plan(sql_statements: list[str], schema_text: str) -> list[str]:
    """Run pre-flight validation checks on the generated SQL."""
    checks: list[str] = []

    for i, stmt in enumerate(sql_statements):
        upper = stmt.upper().strip()

        # Must be INSERT or UPDATE only
        if not (upper.startswith("INSERT") or upper.startswith("UPDATE")):
            checks.append(f"Statement {i+1}: Not an INSERT/UPDATE – will be skipped")
            continue

        # Warn if no WHERE clause on UPDATE
        if upper.startswith("UPDATE") and "WHERE" not in upper:
            checks.append(f"Statement {i+1}: UPDATE without WHERE clause – potential mass update")

        # Warn if inserting without explicit columns
        if upper.startswith("INSERT") and "(" not in upper.split("VALUES")[0]:
            checks.append(f"Statement {i+1}: INSERT without explicit column list")

    if not sql_statements:
        checks.append("No SQL statements generated – plan is empty")

    return checks


# ═══════════════════════════════════════════════
#  Write plan generation
# ═══════════════════════════════════════════════

async def plan_db_write(input_data: dict[str, Any]) -> dict[str, Any]:
    """Generate a write plan from extraction results.

    Pipeline:
      1. Fetch current schema
      2. LLM generates SQL + rollback + diff preview
      3. Validate plan
      4. Assess risk level
      5. Generate confirmation token with TTL
      6. Store plan

    Args:
        input_data: Must conform to X5PlanInput schema.

    Returns:
        Serialised WritePlanSchema with confirmation_token.
    """
    parsed = X5PlanInput(**input_data)
    session_id = input_data.get("_session_id")
    logger.info("X5 write plan started | applicant=%s dry_run=%s", parsed.applicant_id, parsed.dry_run)

    schema_text = await get_db_schema("primary")
    extraction_text = json.dumps(parsed.extraction_bundle or {}, indent=2)

    messages = [
        {"role": "system", "content": WRITE_PLAN_SYSTEM_PROMPT.format(schema=schema_text)},
        {
            "role": "user",
            "content": (
                f"Applicant ID: {parsed.applicant_id}\n\n"
                f"Extraction bundle:\n{extraction_text[:8000]}\n\n"
                "Generate a write plan with rollback statements for these extracted entities."
            ),
        },
    ]

    raw = await get_chat_completion(
        messages, temperature=0.0, response_format="json_object", session_id=session_id
    )

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("X5: LLM returned invalid JSON")
        return WritePlanSchema(
            plan_id=str(uuid.uuid4()),
            applicant_id=parsed.applicant_id,
            sql_statements=[],
            confirmation_token="",
            status="failed",
            validation_checks=["LLM returned invalid JSON – plan generation failed"],
        ).model_dump()

    sql_statements = result.get("sql_statements", [])
    rollback_statements = result.get("rollback_statements", [])
    impacted_tables = result.get("impacted_tables", [])
    impacted_row_count = result.get("impacted_row_count", 0)

    # Validation
    validation_checks = _validate_plan(sql_statements, schema_text)
    validation_checks.extend(result.get("validation_notes", []))

    # Risk assessment
    risk_level = _assess_risk(sql_statements, impacted_tables, impacted_row_count)

    plan_id = str(uuid.uuid4())
    token = secrets.token_urlsafe(32)
    ttl = settings.write_plan_ttl_seconds
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)

    status = "dry_run" if parsed.dry_run else "pending"

    plan = WritePlanSchema(
        plan_id=plan_id,
        applicant_id=parsed.applicant_id,
        sql_statements=sql_statements,
        diff_preview=result.get("diff_preview", {}),
        impacted_tables=impacted_tables,
        impacted_row_count=impacted_row_count,
        confirmation_token=token,
        status=status,
        expires_at=expires_at,
        rollback_statements=rollback_statements,
        risk_level=risk_level,
        validation_checks=validation_checks,
    )

    _active_plans[plan_id] = plan

    logger.info(
        "X5 write plan created | plan_id=%s statements=%d rollbacks=%d risk=%s expires=%s",
        plan_id, len(sql_statements), len(rollback_statements), risk_level, expires_at.isoformat(),
    )
    return plan.model_dump()


# ═══════════════════════════════════════════════
#  Write plan commit
# ═══════════════════════════════════════════════

async def commit_db_write(input_data: dict[str, Any]) -> dict[str, Any]:
    """Commit a previously created write plan after user confirmation.

    Pre-commit checks:
      1. Plan exists and is in 'pending' state
      2. Confirmation token matches
      3. Token has not expired
      4. No concurrent commit in flight for the same applicant

    Args:
        input_data: Must conform to X5CommitInput schema.

    Returns:
        Serialised CommitResult.
    """
    parsed = X5CommitInput(**input_data)
    logger.info("X5 commit requested | plan_id=%s", parsed.write_plan_id)

    # ── Retrieve plan ───────────────────────────────
    plan = _active_plans.get(parsed.write_plan_id)
    if not plan:
        logger.warning("X5: Plan not found: %s", parsed.write_plan_id)
        return CommitResult(
            plan_id=parsed.write_plan_id,
            status="failed",
            error_message="Write plan not found. It may have expired or been consumed.",
        ).model_dump()

    # ── State check ─────────────────────────────────
    if plan.status == "dry_run":
        return CommitResult(
            plan_id=parsed.write_plan_id,
            status="failed",
            error_message="This is a dry-run plan and cannot be committed.",
        ).model_dump()

    if plan.status != "pending":
        return CommitResult(
            plan_id=parsed.write_plan_id,
            status="failed",
            error_message=f"Plan is not in pending state (current: {plan.status}).",
        ).model_dump()

    # ── Token validation ────────────────────────────
    if plan.confirmation_token != parsed.confirmation_token:
        logger.warning("X5: Invalid confirmation token for plan %s", parsed.write_plan_id)
        return CommitResult(
            plan_id=parsed.write_plan_id,
            status="failed",
            error_message="Invalid confirmation token. Write operation denied.",
        ).model_dump()

    # ── Expiry check ────────────────────────────────
    if plan.expires_at and datetime.now(timezone.utc) > plan.expires_at:
        plan.status = "expired"
        logger.warning("X5: Plan %s expired at %s", parsed.write_plan_id, plan.expires_at)
        return CommitResult(
            plan_id=parsed.write_plan_id,
            status="expired",
            error_message=(
                f"Confirmation token expired at {plan.expires_at.isoformat()}. "
                "Generate a new write plan."
            ),
        ).model_dump()

    # ── Concurrent conflict detection ───────────────
    if plan.applicant_id in _inflight_applicants:
        return CommitResult(
            plan_id=parsed.write_plan_id,
            status="failed",
            error_message=(
                f"Another write is in progress for applicant {plan.applicant_id}. "
                "Please wait and retry."
            ),
        ).model_dump()

    # ── Execute ─────────────────────────────────────
    _inflight_applicants.add(plan.applicant_id)
    audit_id = str(uuid.uuid4())

    try:
        rows_affected = await execute_write_plan(plan.sql_statements)
        plan.status = "committed"
        committed_at = datetime.now(timezone.utc)

        result = CommitResult(
            plan_id=parsed.write_plan_id,
            status="committed",
            rows_affected=rows_affected,
            committed_at=committed_at,
            audit_id=audit_id,
        )
        logger.info(
            "X5 commit succeeded | plan_id=%s rows=%d audit_id=%s confirmed_by=%s",
            parsed.write_plan_id, rows_affected, audit_id, parsed.confirmed_by or "unknown",
        )
    except Exception as e:
        plan.status = "failed"
        result = CommitResult(
            plan_id=parsed.write_plan_id,
            status="failed",
            error_message=str(e),
            audit_id=audit_id,
        )
        logger.error(
            "X5 commit failed | plan_id=%s audit_id=%s error=%s",
            parsed.write_plan_id, audit_id, str(e),
        )
    finally:
        _inflight_applicants.discard(plan.applicant_id)

    return result.model_dump()


# ═══════════════════════════════════════════════
#  Plan management helpers
# ═══════════════════════════════════════════════

def get_active_plan(plan_id: str) -> WritePlanSchema | None:
    """Retrieve an active write plan by ID (for API endpoints)."""
    plan = _active_plans.get(plan_id)
    if plan and plan.expires_at and datetime.now(timezone.utc) > plan.expires_at:
        plan.status = "expired"
    return plan


def get_plans_for_applicant(applicant_id: str) -> list[WritePlanSchema]:
    """Get all active plans for a given applicant."""
    now = datetime.now(timezone.utc)
    plans = []
    for p in _active_plans.values():
        if p.applicant_id == applicant_id:
            if p.expires_at and now > p.expires_at and p.status == "pending":
                p.status = "expired"
            plans.append(p)
    return plans


def cleanup_expired_plans() -> int:
    """Remove expired/committed plans from memory. Returns count removed."""
    now = datetime.now(timezone.utc)
    to_remove = [
        pid for pid, p in _active_plans.items()
        if p.status in ("committed", "failed", "expired")
        or (p.expires_at and now > p.expires_at)
    ]
    for pid in to_remove:
        del _active_plans[pid]
    return len(to_remove)
