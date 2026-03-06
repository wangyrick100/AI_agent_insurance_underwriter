"""X4 – Schema-Aware Text-to-SQL Read Agent.

Converts natural-language questions into SELECT-only SQL queries
and executes them against the underwriting database.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from underwriting_suite.agent.schemas import SQLResult, X4Input
from underwriting_suite.services.azure_openai import get_chat_completion
from underwriting_suite.services.db_service import execute_readonly_query, get_db_schema

logger = logging.getLogger(__name__)

SQL_READ_SYSTEM_PROMPT = """\
You are AgentX4SQLRead, a schema-aware Text-to-SQL agent for underwriting databases.

TASK:
Convert the user's natural-language question into a safe SELECT query.

SCHEMA:
{schema}

OUTPUT FORMAT (JSON):
{{
  "sql": "SELECT ...",
  "explanation": "brief explanation of what the query does"
}}

RULES:
1. ONLY produce SELECT statements. Never INSERT, UPDATE, DELETE, DROP, ALTER, or TRUNCATE.
2. Use table and column names exactly as shown in the schema.
3. Add LIMIT 100 if no limit is specified.
4. Do not expose sensitive fields like SSN in results.
5. Use parameterised patterns where possible for safety.
"""


def _validate_select_only(sql: str) -> bool:
    """Ensure the SQL is a SELECT-only statement."""
    cleaned = sql.strip().upper()
    # Must start with SELECT or WITH (CTE)
    if not (cleaned.startswith("SELECT") or cleaned.startswith("WITH")):
        return False
    # Must not contain dangerous keywords
    dangerous = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "EXEC", "EXECUTE"]
    for kw in dangerous:
        if re.search(rf"\b{kw}\b", cleaned):
            return False
    return True


async def x4_sql_read(input_data: dict[str, Any]) -> dict[str, Any]:
    """Convert a natural-language question to SQL and execute (read-only).

    Args:
        input_data: Must conform to X4Input schema.

    Returns:
        Serialised SQLResult.
    """
    parsed = X4Input(**input_data)
    logger.info("X4 SQL read started | question=%s", parsed.question[:80])

    db_id = parsed.db_id or "primary"
    schema_text = await get_db_schema(db_id)

    messages = [
        {"role": "system", "content": SQL_READ_SYSTEM_PROMPT.format(schema=schema_text)},
        {
            "role": "user",
            "content": (
                f"Question: {parsed.question}\n"
                f"Database: {db_id}\n"
                f"Additional constraints: {json.dumps(parsed.constraints) if parsed.constraints else 'none'}"
            ),
        },
    ]

    raw = await get_chat_completion(messages, temperature=0.0, response_format="json_object")

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("X4: LLM returned invalid JSON")
        return SQLResult(query="ERROR", db_id=db_id).model_dump()

    sql = result.get("sql", "").strip()

    # Validate SELECT-only
    if not _validate_select_only(sql):
        logger.warning("X4: Blocked non-SELECT query: %s", sql[:100])
        return SQLResult(
            query=sql,
            db_id=db_id,
            disclaimers=["Query blocked: only SELECT statements are allowed."],
        ).model_dump()

    # Execute query
    try:
        columns, rows = await execute_readonly_query(sql, db_id)
        sql_result = SQLResult(
            query=sql,
            db_id=db_id,
            columns=columns,
            rows=rows,
            row_count=len(rows),
        )
    except Exception as e:
        logger.error("X4: Query execution failed: %s", str(e))
        sql_result = SQLResult(
            query=sql,
            db_id=db_id,
            disclaimers=[f"Query execution failed: {str(e)}"],
        )

    logger.info("X4 SQL read complete | rows=%d", sql_result.row_count)
    return sql_result.model_dump()
