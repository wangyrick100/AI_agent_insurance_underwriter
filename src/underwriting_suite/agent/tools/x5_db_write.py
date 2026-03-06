"""X5 – Safe Database Write Agent (Plan + Confirm + Commit).

Generates a write plan from extracted entities, requires explicit
confirmation token before committing any changes.
"""

from __future__ import annotations

import json
import logging
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any

from underwriting_suite.agent.schemas import (
    CommitResult,
    WritePlanSchema,
    X5CommitInput,
    X5PlanInput,
)
from underwriting_suite.services.azure_openai import get_chat_completion
from underwriting_suite.services.db_service import execute_write_plan, get_db_schema

logger = logging.getLogger(__name__)

# In-memory store for active write plans (production: use DB table)
_active_plans: dict[str, WritePlanSchema] = {}

WRITE_PLAN_SYSTEM_PROMPT = """\
You are AgentX5WritePlan, a safe database write planner for underwriting systems.

TASK:
Given extracted entities for an applicant, generate SQL statements to
insert or update the underwriting database.

SCHEMA:
{schema}

OUTPUT FORMAT (JSON):
{{
  "sql_statements": ["INSERT INTO ...", "UPDATE ... SET ..."],
  "diff_preview": {{"table_name": {{"before": {{}}, "after": {{}}}}}},
  "impacted_tables": ["table1", "table2"],
  "impacted_row_count": 5
}}

RULES:
1. Generate ONLY INSERT and UPDATE statements – never DELETE or DROP.
2. Use parameterised values where possible.
3. Provide a diff preview showing before/after for each affected record.
4. Track all impacted tables and estimated row count.
5. These statements will NOT execute until the user confirms with a token.
"""


async def x5_write_plan(input_data: dict[str, Any]) -> dict[str, Any]:
    """Generate a write plan from extraction results.

    Args:
        input_data: Must conform to X5PlanInput schema.

    Returns:
        Serialised WritePlanSchema with confirmation_token.
    """
    parsed = X5PlanInput(**input_data)
    logger.info("X5 write plan started | applicant=%s", parsed.applicant_id)

    schema_text = await get_db_schema("primary")
    extraction_text = json.dumps(parsed.extraction_bundle or {}, indent=2)

    messages = [
        {"role": "system", "content": WRITE_PLAN_SYSTEM_PROMPT.format(schema=schema_text)},
        {
            "role": "user",
            "content": (
                f"Applicant ID: {parsed.applicant_id}\n\n"
                f"Extraction bundle:\n{extraction_text}\n\n"
                "Generate a write plan for these extracted entities."
            ),
        },
    ]

    raw = await get_chat_completion(messages, temperature=0.0, response_format="json_object")

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
        ).model_dump()

    plan_id = str(uuid.uuid4())
    token = secrets.token_urlsafe(32)

    plan = WritePlanSchema(
        plan_id=plan_id,
        applicant_id=parsed.applicant_id,
        sql_statements=result.get("sql_statements", []),
        diff_preview=result.get("diff_preview", {}),
        impacted_tables=result.get("impacted_tables", []),
        impacted_row_count=result.get("impacted_row_count", 0),
        confirmation_token=token,
        status="pending",
    )

    # Store plan for later commit
    _active_plans[plan_id] = plan

    logger.info(
        "X5 write plan created | plan_id=%s statements=%d",
        plan_id,
        len(plan.sql_statements),
    )
    return plan.model_dump()


async def x5_write_commit(input_data: dict[str, Any]) -> dict[str, Any]:
    """Commit a previously created write plan after user confirmation.

    Args:
        input_data: Must conform to X5CommitInput schema.

    Returns:
        Serialised CommitResult.
    """
    parsed = X5CommitInput(**input_data)
    logger.info("X5 commit requested | plan_id=%s", parsed.write_plan_id)

    # Retrieve plan
    plan = _active_plans.get(parsed.write_plan_id)
    if not plan:
        logger.warning("X5: Plan not found: %s", parsed.write_plan_id)
        return CommitResult(
            plan_id=parsed.write_plan_id,
            status="failed",
            error_message="Write plan not found. It may have expired.",
        ).model_dump()

    # Validate confirmation token
    if plan.confirmation_token != parsed.confirmation_token:
        logger.warning("X5: Invalid confirmation token for plan %s", parsed.write_plan_id)
        return CommitResult(
            plan_id=parsed.write_plan_id,
            status="failed",
            error_message="Invalid confirmation token. Write operation denied.",
        ).model_dump()

    if plan.status != "pending":
        return CommitResult(
            plan_id=parsed.write_plan_id,
            status="failed",
            error_message=f"Plan is not in pending state (current: {plan.status}).",
        ).model_dump()

    # Execute the write plan
    try:
        rows_affected = await execute_write_plan(plan.sql_statements)
        plan.status = "committed"
        result = CommitResult(
            plan_id=parsed.write_plan_id, status="committed", rows_affected=rows_affected
        )
        logger.info("X5 commit succeeded | plan_id=%s rows=%d", parsed.write_plan_id, rows_affected)
    except Exception as e:
        plan.status = "failed"
        result = CommitResult(
            plan_id=parsed.write_plan_id,
            status="failed",
            error_message=str(e),
        )
        logger.error("X5 commit failed | plan_id=%s error=%s", parsed.write_plan_id, str(e))

    return result.model_dump()


def get_active_plan(plan_id: str) -> WritePlanSchema | None:
    """Retrieve an active write plan by ID (for API endpoints)."""
    return _active_plans.get(plan_id)


def get_plans_for_applicant(applicant_id: str) -> list[WritePlanSchema]:
    """Get all active plans for a given applicant."""
    return [p for p in _active_plans.values() if p.applicant_id == applicant_id]
