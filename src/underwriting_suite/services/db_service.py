"""Database service – read-only queries and safe write execution."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text

from underwriting_suite.db.database import async_session, engine

logger = logging.getLogger(__name__)


async def get_db_schema(db_id: str = "primary") -> str:
    """Return the database schema as a text description for the LLM.

    Args:
        db_id: Database identifier (currently only 'primary' supported).

    Returns:
        Multi-line string describing tables and columns.
    """
    schema_description = """\
DATABASE SCHEMA (primary):

TABLE applicants:
  id          VARCHAR(36)   PRIMARY KEY
  first_name  VARCHAR(100)  NOT NULL
  last_name   VARCHAR(100)  NOT NULL
  date_of_birth DATE
  gender      VARCHAR(20)
  email       VARCHAR(200)
  phone       VARCHAR(30)
  policy_number VARCHAR(50)
  status      VARCHAR(30)   DEFAULT 'pending_review'
  created_at  DATETIME
  updated_at  DATETIME

TABLE documents:
  id            VARCHAR(36)  PRIMARY KEY
  applicant_id  VARCHAR(36)  FOREIGN KEY -> applicants.id
  filename      VARCHAR(500) NOT NULL
  doc_type      VARCHAR(50)  -- aps, meds, labs, vitals, tele_interview, paramedics, application_form
  storage_url   TEXT
  chunk_count   INTEGER
  status        VARCHAR(30)  DEFAULT 'uploaded'
  created_at    DATETIME

TABLE extractions:
  id                VARCHAR(36)   PRIMARY KEY
  applicant_id      VARCHAR(36)   FOREIGN KEY -> applicants.id
  document_id       VARCHAR(36)
  entity_type       VARCHAR(50)   -- medication, diagnosis, lab_result, vital, procedure, demographic
  entity_name       VARCHAR(300)  NOT NULL
  entity_value      TEXT
  confidence        FLOAT
  evidence_chunk_id VARCHAR(36)
  evidence_snippet  TEXT
  metadata_json     JSON
  created_at        DATETIME

TABLE risk_scores:
  id                VARCHAR(36)  PRIMARY KEY
  applicant_id      VARCHAR(36)  FOREIGN KEY -> applicants.id
  score             FLOAT        NOT NULL
  confidence        FLOAT
  risk_class        VARCHAR(50)  -- preferred, standard, substandard, decline
  feature_rationale JSON
  similar_cases     JSON
  model_version     VARCHAR(50)
  disclaimers       TEXT
  created_at        DATETIME

TABLE write_plans:
  id                  VARCHAR(36)  PRIMARY KEY
  applicant_id        VARCHAR(36)  FOREIGN KEY -> applicants.id
  status              VARCHAR(30)  DEFAULT 'pending'
  sql_statements      JSON
  diff_preview        JSON
  impacted_tables     JSON
  impacted_row_count  INTEGER
  confirmation_token  VARCHAR(64)
  confirmed_by        VARCHAR(100)
  error_message       TEXT
  created_at          DATETIME
  committed_at        DATETIME
"""
    return schema_description


async def execute_readonly_query(
    sql: str, db_id: str = "primary"
) -> tuple[list[str], list[list[Any]]]:
    """Execute a read-only SQL query and return results.

    Args:
        sql: SELECT-only SQL statement.
        db_id: Database identifier.

    Returns:
        Tuple of (column_names, rows).
    """
    logger.info("Executing read-only query: %s", sql[:100])

    async with async_session() as session:
        result = await session.execute(text(sql))
        columns = list(result.keys())
        rows = [list(row) for row in result.fetchall()]
        return columns, rows


async def execute_write_plan(sql_statements: list[str]) -> int:
    """Execute a list of write SQL statements within a transaction.

    Args:
        sql_statements: List of INSERT/UPDATE SQL statements.

    Returns:
        Total rows affected.
    """
    total_rows = 0

    async with async_session() as session:
        async with session.begin():
            for stmt in sql_statements:
                logger.info("Executing write statement: %s", stmt[:100])
                result = await session.execute(text(stmt))
                total_rows += result.rowcount
            await session.commit()

    logger.info("Write plan executed: %d rows affected", total_rows)
    return total_rows
