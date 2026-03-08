---
name: plan_db_write
version: "2.0.0"
tool: plan_db_write
category: database
cost_tier: medium
tags:
  - sql
  - write
  - plan
  - insert
  - update
  - two-phase-commit
---

# Skill: plan_db_write

## Purpose

Generate a safe, reviewable write plan for persisting extracted underwriting
entities to the database. Returns a plan containing the SQL statements,
validation checks, rollback statements, risk level, and a confirmation token
required for the subsequent `commit_db_write` call.

**This tool does NOT execute any writes.** It only produces a plan for human
or Supervisor review.

## When to Invoke

- Extracted entities (from `extract_entities`) need to be persisted to the
  database for the first time.
- An applicant record, risk score, or extraction bundle must be updated.
- The Supervisor is ready to write data but has not yet obtained a confirmation
  token from the user.

## Do NOT Invoke When

- You want to immediately commit without review – the two-phase write is
  mandatory for data integrity.
- Only a read is needed – use `read_sql`.

## Inputs

| Field               | Type   | Required | Description                                             |
|---------------------|--------|----------|---------------------------------------------------------|
| `applicant_id`      | `str`  | Yes      | Target applicant identifier                             |
| `extraction_bundle` | `dict` | No       | Output from `extract_entities` to persist               |
| `dry_run`           | `bool` | No       | Generate plan without allowing a subsequent commit      |

## Outputs

```json
{
  "plan_id": "plan-uuid-1234",
  "applicant_id": "APL-001",
  "statements": [
    "INSERT INTO extractions (applicant_id, ...) VALUES ('APL-001', ...)",
    "UPDATE applicants SET last_extraction_at = NOW() WHERE id = 'APL-001'"
  ],
  "rollback_statements": [
    "DELETE FROM extractions WHERE id = 'ext-uuid'",
    "UPDATE applicants SET last_extraction_at = NULL WHERE id = 'APL-001'"
  ],
  "risk_level": "low",
  "validation_checks": ["applicant exists", "no duplicate extraction"],
  "confirmation_token": "tok-abc789",
  "expires_at": "2024-06-01T12:30:00Z",
  "dry_run": false
}
```

## Usage Example (Supervisor ReAct JSON)

```json
{
  "thought_summary": "Extraction complete. Generating write plan for APL-001.",
  "next_tool": "plan_db_write",
  "tool_input": {
    "applicant_id": "APL-001",
    "extraction_bundle": { "...": "output from extract_entities" }
  }
}
```

## Notes

- The `confirmation_token` expires after the window configured in
  `settings.WRITE_PLAN_TTL_SECONDS` (default: 300 s).
- High-risk plans (bulk updates, cascading deletes) require an additional
  supervisor approval flag before `commit_db_write` will proceed.
- If `dry_run` is `true`, the returned token is non-committal and
  `commit_db_write` will reject it.
