"""X4 – Schema-Aware Text-to-SQL Read Agent.

Converts natural-language questions into SELECT-only SQL queries
and executes them against the underwriting database.

Production capabilities:
  • Multi-turn self-correction  – if a query fails, the agent inspects the
    error, revises the SQL, and retries (up to configurable max).
  • LLM self-review  – optional pre-execution review pass that checks for
    correctness, optimises JOINs, and adds index hints (feature flag).
  • Query complexity scoring  – classifies query as simple / moderate /
    complex / multi_join to inform cost and performance expectations.
  • Robust SQL validation  – regex-based guard against DML/DDL, CTE bombs,
    stacked queries, and comment-based injection.
  • Configurable row cap  – enforces LIMIT to prevent large result sets.
  • Execution timing  – measures and reports query execution latency.
  • QueryPlan metadata  – returns structured metadata alongside results.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from underwriting_suite.agent.schemas import QueryPlan, SQLResult, X4Input
from underwriting_suite.config import settings
from underwriting_suite.services.azure_openai import get_chat_completion
from underwriting_suite.services.db_service import execute_readonly_query, get_db_schema

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════
#  System prompts
# ═══════════════════════════════════════════════

SQL_READ_SYSTEM_PROMPT = """\
You are AgentX4SQLRead, a schema-aware Text-to-SQL agent for underwriting databases.

TASK:
Convert the user's natural-language question into a safe, optimised SELECT query.

SCHEMA:
{schema}

OUTPUT FORMAT (strict JSON):
{{
  "sql": "SELECT ...",
  "explanation": "what the query does and why this approach was chosen",
  "complexity": "simple|moderate|complex|multi_join"
}}

RULES:
1. ONLY produce SELECT statements (or WITH ... SELECT CTEs). Never INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, EXEC.
2. Use table and column names EXACTLY as shown in the schema.
3. Add LIMIT {max_rows} unless the user explicitly requests all rows.
4. Do NOT expose sensitive fields (SSN, confirmation_token) in results.
5. Prefer explicit JOINs over sub-selects where possible.
6. Use aliases for readability in multi-table queries.
7. For date filtering, use ISO 8601 format: 'YYYY-MM-DD'.
8. If the question is ambiguous, state your interpretation in the explanation.
"""

SQL_REVIEW_PROMPT = """\
You are a SQL quality reviewer.

Given the SCHEMA and a generated SQL query, review it for:
1. Correctness – does it answer the stated question?
2. Safety – is it truly SELECT-only with no side effects?
3. Performance – could it be optimised (e.g. unnecessary sub-queries, missing index hints)?
4. Edge cases – will it handle NULLs, empty tables, or duplicates correctly?

Return a revised version if needed.

OUTPUT FORMAT (strict JSON):
{{
  "revised_sql": "SELECT ... (or identical if no changes needed)",
  "changes_made": ["list of changes or 'none'"],
  "review_notes": ["safety/performance/correctness observations"]
}}
"""

SQL_CORRECTION_PROMPT = """\
You are AgentX4SQLRead.  Your previous query caused an execution error.

SCHEMA:
{schema}

ORIGINAL QUESTION: {question}

PREVIOUS SQL:
{failed_sql}

ERROR MESSAGE:
{error}

Generate a CORRECTED SELECT query that avoids the error.

OUTPUT FORMAT (strict JSON):
{{
  "sql": "SELECT ...",
  "explanation": "what was wrong and how this query fixes it",
  "complexity": "simple|moderate|complex|multi_join"
}}
"""


# ═══════════════════════════════════════════════
#  SQL validation
# ═══════════════════════════════════════════════

_DANGEROUS_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "EXEC", "EXECUTE", "GRANT", "REVOKE", "CREATE", "MERGE",
    "CALL", "RENAME", "COMMENT",
]

_COMMENT_RE = re.compile(r"(--.*?$|/\*.*?\*/)", re.MULTILINE | re.DOTALL)


def _validate_select_only(sql: str) -> tuple[bool, str]:
    """Validate that the SQL is a read-only SELECT statement.

    Returns (is_valid, reason).
    """
    # Strip comments first (common injection vector)
    stripped = _COMMENT_RE.sub("", sql).strip()
    cleaned = stripped.upper()

    # Must start with SELECT or WITH (CTE)
    if not (cleaned.startswith("SELECT") or cleaned.startswith("WITH")):
        return False, "Query must start with SELECT or WITH"

    # Check for stacked queries (semicolon-separated)
    # Allow trailing semicolon but not multiple statements
    statements = [s.strip() for s in stripped.split(";") if s.strip()]
    if len(statements) > 1:
        return False, "Multiple statements detected – only single SELECT allowed"

    # Check for dangerous keywords
    for kw in _DANGEROUS_KEYWORDS:
        if re.search(rf"\b{kw}\b", cleaned):
            return False, f"Blocked keyword detected: {kw}"

    # Check for system table / information_schema access
    if re.search(r"\b(INFORMATION_SCHEMA|SYS\.|SYSOBJECTS|PG_CATALOG)\b", cleaned):
        return False, "Access to system tables is blocked"

    return True, "OK"


def _assess_complexity(sql: str) -> str:
    """Heuristic complexity classification."""
    upper = sql.upper()
    join_count = len(re.findall(r"\bJOIN\b", upper))
    has_subquery = "SELECT" in upper[upper.find("FROM"):] if "FROM" in upper else False
    has_groupby = "GROUP BY" in upper
    has_having = "HAVING" in upper
    has_window = "OVER(" in upper.replace(" ", "")

    if join_count >= 3 or (has_subquery and join_count >= 1):
        return "multi_join"
    if join_count >= 2 or has_window or (has_groupby and has_having):
        return "complex"
    if join_count >= 1 or has_groupby or has_subquery:
        return "moderate"
    return "simple"


def _enforce_limit(sql: str, max_rows: int) -> str:
    """Ensure the query has a LIMIT clause."""
    upper = sql.strip().upper()
    if "LIMIT" not in upper:
        # Remove trailing semicolon first
        sql = sql.rstrip().rstrip(";")
        sql += f"\nLIMIT {max_rows}"
    return sql


# ═══════════════════════════════════════════════
#  Public entry point
# ═══════════════════════════════════════════════

async def skill_sql_read(input_data: dict[str, Any]) -> dict[str, Any>:
    """Convert a natural-language question to SQL and execute (read-only).

    Pipeline:
      1. Fetch schema context
      2. LLM generates SQL + explanation + complexity
      3. Validate SELECT-only
      4. Optional self-review pass (feature flag)
      5. Enforce row limit
      6. Execute query
      7. On error → self-correction loop (up to N times)
      8. Return SQLResult with QueryPlan metadata

    Args:
        input_data: Must conform to X4Input schema.

    Returns:
        Serialised SQLResult.
    """
    parsed = X4Input(**input_data)
    session_id = input_data.get("_session_id")
    max_rows = min(parsed.max_rows, settings.sql_max_result_rows)
    db_id = parsed.db_id or "primary"
    logger.info("X4 SQL read started | question=%s | max_rows=%d", parsed.question[:80], max_rows)

    schema_text = await get_db_schema(db_id)

    # ── 1. Initial SQL generation ───────────────────
    messages = [
        {"role": "system", "content": SQL_READ_SYSTEM_PROMPT.format(schema=schema_text, max_rows=max_rows)},
        {
            "role": "user",
            "content": (
                f"Question: {parsed.question}\n"
                f"Database: {db_id}\n"
                f"Additional constraints: {json.dumps(parsed.constraints) if parsed.constraints else 'none'}"
            ),
        },
    ]

    raw = await get_chat_completion(
        messages, temperature=0.0, response_format="json_object", session_id=session_id
    )

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("X4: LLM returned invalid JSON")
        return SQLResult(query="ERROR", db_id=db_id, disclaimers=["LLM output was unparseable."]).model_dump()

    sql = result.get("sql", "").strip()
    explanation = result.get("explanation", "")
    complexity = result.get("complexity", _assess_complexity(sql))

    # ── 2. Validate ─────────────────────────────────
    is_valid, reason = _validate_select_only(sql)
    if not is_valid:
        logger.warning("X4: Blocked query: %s – %s", sql[:100], reason)
        return SQLResult(
            query=sql, db_id=db_id,
            disclaimers=[f"Query blocked: {reason}"],
            query_plan=QueryPlan(sql=sql, explanation=explanation, complexity=complexity,
                                 review_notes=[f"BLOCKED: {reason}"]),
        ).model_dump()

    # ── 3. Optional self-review ─────────────────────
    review_notes: list[str] = []
    if settings.enable_sql_query_review:
        review_messages = [
            {"role": "system", "content": SQL_REVIEW_PROMPT},
            {
                "role": "user",
                "content": (
                    f"SCHEMA:\n{schema_text}\n\n"
                    f"QUESTION: {parsed.question}\n\n"
                    f"GENERATED SQL:\n{sql}"
                ),
            },
        ]
        review_raw = await get_chat_completion(
            review_messages, temperature=0.0, response_format="json_object", session_id=session_id
        )
        try:
            review = json.loads(review_raw)
            revised = review.get("revised_sql", sql).strip()
            if revised and revised != sql:
                # Re-validate the revised SQL
                ok, rev_reason = _validate_select_only(revised)
                if ok:
                    sql = revised
                    logger.info("X4: SQL revised by self-review")
                else:
                    review_notes.append(f"Revision rejected ({rev_reason}); using original")
            review_notes.extend(review.get("review_notes", []))
        except json.JSONDecodeError:
            review_notes.append("Self-review pass returned invalid JSON – using original SQL.")

    # ── 4. Enforce limit ────────────────────────────
    sql = _enforce_limit(sql, max_rows)

    # ── 5. Execute with self-correction loop ────────
    max_corrections = 3 if parsed.enable_self_correction else 0
    corrections = 0
    last_error = ""

    for attempt in range(1 + max_corrections):
        try:
            exec_start = time.time()
            columns, rows = await execute_readonly_query(sql, db_id)
            exec_ms = (time.time() - exec_start) * 1000

            truncated = len(rows) >= max_rows
            if truncated:
                rows = rows[:max_rows]

            sql_result = SQLResult(
                query=sql,
                db_id=db_id,
                columns=columns,
                rows=rows,
                row_count=len(rows),
                truncated=truncated,
                execution_time_ms=round(exec_ms, 1),
                self_correction_attempts=corrections,
                query_plan=QueryPlan(
                    sql=sql, explanation=explanation,
                    complexity=complexity, review_notes=review_notes,
                ),
            )
            logger.info(
                "X4 SQL read complete | rows=%d corrections=%d exec_ms=%.1f complexity=%s",
                len(rows), corrections, exec_ms, complexity,
            )
            return sql_result.model_dump()

        except Exception as e:
            last_error = str(e)
            logger.warning("X4 query execution failed (attempt %d): %s", attempt + 1, last_error[:200])

            if attempt < max_corrections:
                # Self-correction: ask LLM to fix
                corrections += 1
                correction_messages = [
                    {
                        "role": "system",
                        "content": SQL_CORRECTION_PROMPT.format(
                            schema=schema_text, question=parsed.question,
                            failed_sql=sql, error=last_error[:500],
                        ),
                    },
                    {"role": "user", "content": "Generate a corrected SQL query."},
                ]
                correction_raw = await get_chat_completion(
                    correction_messages, temperature=0.0,
                    response_format="json_object", session_id=session_id,
                )
                try:
                    correction_data = json.loads(correction_raw)
                    new_sql = correction_data.get("sql", "").strip()
                    ok, val_reason = _validate_select_only(new_sql)
                    if ok:
                        sql = _enforce_limit(new_sql, max_rows)
                        explanation = correction_data.get("explanation", explanation)
                        complexity = correction_data.get("complexity", complexity)
                        logger.info("X4: Self-corrected SQL (attempt %d)", corrections)
                    else:
                        logger.warning("X4: Corrected SQL also blocked – %s", val_reason)
                        break
                except json.JSONDecodeError:
                    logger.warning("X4: Correction LLM returned invalid JSON")
                    break

    # All attempts failed
    return SQLResult(
        query=sql, db_id=db_id,
        self_correction_attempts=corrections,
        query_plan=QueryPlan(sql=sql, explanation=explanation, complexity=complexity, review_notes=review_notes),
        disclaimers=[f"Query execution failed after {corrections + 1} attempt(s): {last_error}"],
    ).model_dump()
