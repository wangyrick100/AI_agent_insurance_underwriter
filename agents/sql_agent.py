"""SQL Agent: translate natural-language queries to schema-aware SQL and execute them."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from llm.base import BaseLLM
from utils.database import execute_query, get_schema_ddl


class SQLAgent:
    """Converts natural-language questions to SQL queries and executes them.

    The agent injects the database schema into every LLM prompt so the model
    can produce syntactically and semantically correct SQL without needing
    any database-specific fine-tuning.

    Parameters
    ----------
    llm:
        Language model used to generate SQL.
    engine:
        Optional SQLAlchemy engine.  Defaults to the global engine from
        :mod:`utils.database`.
    max_rows:
        Maximum number of result rows returned to the caller.
    """

    SYSTEM_PROMPT = (
        "You are a senior data analyst who writes precise, read-only SQL queries. "
        "You must ONLY write SELECT statements. "
        "Return ONLY the SQL query, with no explanation or markdown fences."
    )

    def __init__(
        self,
        llm: BaseLLM,
        engine=None,
        max_rows: int = 50,
    ):
        self.llm = llm
        self.engine = engine
        self.max_rows = max_rows
        self._schema_ddl: Optional[str] = None

    def query(self, natural_language_query: str) -> Dict[str, Any]:
        """Execute *natural_language_query* and return results.

        Returns
        -------
        dict
            ``sql`` (generated SQL string), ``rows`` (list of row dicts),
            ``row_count`` (int), ``error`` (str or None).
        """
        schema = self._get_schema()
        sql = self._generate_sql(natural_language_query, schema)
        sql = self._sanitise_sql(sql)

        try:
            rows = execute_query(sql, engine=self.engine)
            rows = rows[: self.max_rows]
            return {"sql": sql, "rows": rows, "row_count": len(rows), "error": None}
        except Exception as exc:  # noqa: BLE001
            return {"sql": sql, "rows": [], "row_count": 0, "error": str(exc)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_schema(self) -> str:
        if self._schema_ddl is None:
            self._schema_ddl = get_schema_ddl()
        return self._schema_ddl

    def _generate_sql(self, question: str, schema: str) -> str:
        prompt = (
            f"Database schema:\n{schema}\n\n"
            f"Write a SQL query to answer:\n{question}"
        )
        return self.llm.complete(prompt, system=self.SYSTEM_PROMPT)

    @staticmethod
    def _sanitise_sql(raw: str) -> str:
        """Strip markdown fences and trailing semicolons that cause issues."""
        # Remove ```sql ... ``` fences
        raw = re.sub(r"```(?:sql)?\s*", "", raw, flags=re.IGNORECASE)
        raw = raw.replace("```", "").strip()
        # Ensure it ends with a semicolon for clarity
        if not raw.endswith(";"):
            raw += ";"
        return raw
