---
name: read_sql
version: "2.0.0"
tool: read_sql
category: database
cost_tier: medium
tags:
  - sql
  - read
  - text-to-sql
  - database
  - select
---

# Skill: read_sql

## Purpose

Convert a natural-language question into a **SELECT-only** SQL query and execute
it against the underwriting database. Supports schema-aware query generation,
LLM self-correction on errors, and complexity analysis to guard against
expensive full-table scans.

## When to Invoke

- Structured underwriting data (applicant records, risk scores, write plans,
  extraction results) must be retrieved from the relational database.
- The question involves filtering, aggregation, joining, or sorting over
  database tables.
- The Supervisor needs counts, summaries, or lookups not available in the
  vector index.

## Do NOT Invoke When

- The answer can be found in ingested PDF/document chunks – use `query_rag`.
- A write operation is needed – use `plan_db_write` + `commit_db_write`.
- The question is about external medical guidelines – use `research_web`.

## Inputs

| Field                    | Type     | Required | Description                                                  |
|--------------------------|----------|----------|--------------------------------------------------------------|
| `question`               | `str`    | Yes      | Natural-language question to answer from the database        |
| `db_id`                  | `str`    | No       | Database identifier (default: primary underwriting DB)       |
| `constraints`            | `dict`   | No       | Extra SQL constraints (e.g. `{"applicant_id": "APL-001"}`)   |
| `max_rows`               | `int`    | No       | Hard cap on result rows (default from config)                |
| `enable_self_correction` | `bool`   | No       | Enable multi-turn LLM correction on SQL errors (default: true)|

## Outputs

```json
{
  "question": "How many applicants were scored this month?",
  "sql": "SELECT COUNT(*) AS total FROM risk_scores WHERE created_at >= '2024-05-01'",
  "rows": [{ "total": 142 }],
  "row_count": 1,
  "execution_time_ms": 34,
  "confidence": 0.95
}
```

## Usage Example (Supervisor ReAct JSON)

```json
{
  "thought_summary": "Need the existing risk score for APL-001 from the DB.",
  "next_tool": "read_sql",
  "tool_input": {
    "question": "What is the latest risk score and tier for applicant APL-001?",
    "constraints": { "applicant_id": "APL-001" }
  }
}
```

## Notes

- Only `SELECT` statements are permitted; any generated SQL containing DML
  (`INSERT`, `UPDATE`, `DELETE`, `DROP`, etc.) is rejected before execution.
- The LLM is given the full schema (`INFORMATION_SCHEMA`) to generate accurate
  joins and column references.
- Self-correction re-sends the failing SQL + database error to the LLM for
  up to 3 correction attempts.
