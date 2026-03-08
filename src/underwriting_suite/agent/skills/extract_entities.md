---
name: extract_entities
version: "2.0.0"
tool: extract_entities
category: extraction
cost_tier: high
tags:
  - nlp
  - medical
  - entities
  - icd10
  - rxnorm
  - snomed
---

# Skill: extract_entities

## Purpose

Extract structured medical and underwriting entities from uploaded documents or
raw text. Returns a normalised bundle of findings that downstream tools
(e.g. `score_risk`, `plan_db_write`, `query_rag`) can consume without further
parsing.

## When to Invoke

- An applicant document has just been ingested and entities are not yet present
  in the database.
- The user asks for a medical summary, diagnosis list, or medication history.
- Before running `score_risk` when no pre-built `feature_payload` is available.
- Any time structured clinical data is needed from unstructured text.

## Do NOT Invoke When

- Entity records for this applicant already exist and are current (use `read_sql`
  or `query_rag` instead).
- The input contains only numeric or tabular data with no clinical narrative.

## Inputs

| Field        | Type            | Required | Description                                          |
|--------------|-----------------|----------|------------------------------------------------------|
| `doc_ids`    | `list[str]`     | No†      | Document IDs to extract from (fetched via doc store) |
| `raw_text`   | `str`           | No†      | Raw document text (alternative to `doc_ids`)         |
| `applicant_id` | `str`         | No       | Scopes results and enables auto-save to DB           |

† At least one of `doc_ids` or `raw_text` must be provided.

## Outputs

```json
{
  "applicant_id": "APL-001",
  "extraction_id": "ext-uuid",
  "entities": {
    "diagnoses":    [{ "code": "I10", "label": "Essential hypertension", "source": "ICD-10" }],
    "medications":  [{ "name": "Lisinopril", "dose": "10 mg", "rxnorm": "29046" }],
    "labs":         [{ "test": "HbA1c", "value": 6.8, "unit": "%", "date": "2024-01-15" }],
    "vitals":       [{ "type": "BMI", "value": 27.4 }],
    "procedures":   [],
    "family_history": [],
    "lifestyle":    [{ "factor": "smoking", "status": "never" }]
  },
  "confidence": 0.92,
  "conflicts": [],
  "processing_notes": []
}
```

## Usage Example (Supervisor ReAct JSON)

```json
{
  "thought_summary": "No entities found in DB for APL-001 – must extract first.",
  "next_tool": "extract_entities",
  "tool_input": {
    "doc_ids": ["doc-abc123", "doc-def456"],
    "applicant_id": "APL-001"
  }
}
```

## Notes

- Entity normalisation maps to ICD-10 (diagnoses), RxNorm (medications), and
  SNOMED-CT (procedures/findings).
- Multi-pass extraction is used: first pass for breadth, second pass for
  verification and deduplication.
- Prompt-injection content embedded in documents is silently discarded.
