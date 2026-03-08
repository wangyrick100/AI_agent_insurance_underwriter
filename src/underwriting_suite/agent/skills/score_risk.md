---
name: score_risk
version: "2.0.0"
tool: score_risk
category: scoring
cost_tier: medium
tags:
  - risk
  - ml
  - scoring
  - ensemble
  - shap
  - mortality
---

# Skill: score_risk

## Purpose

Compute a comprehensive underwriting risk score for an applicant using an
ensemble of LLM reasoning and deterministic rule-based logic. Returns a
6-tier risk classification, mortality rating, feature importance (SHAP-style),
and four sub-scores covering medical, lifestyle, financial, and occupational
dimensions.

## When to Invoke

- After `extract_entities` has produced an extraction bundle for the applicant.
- The user asks to assess, classify, or quantify risk for an applicant.
- A risk score is needed to decide preliminary underwriting outcome.

## Do NOT Invoke When

- No clinical data exists yet – run `extract_entities` first.
- Only a single sub-dimension is needed (tailor `feature_payload` instead).

## Inputs

| Field               | Type              | Required | Description                                              |
|---------------------|-------------------|----------|----------------------------------------------------------|
| `applicant_id`      | `str`             | Yes      | Applicant identifier                                     |
| `extraction_bundle` | `dict`            | No       | Direct output from `extract_entities`                    |
| `feature_payload`   | `dict`            | No       | Pre-built feature dictionary (overrides bundle)          |
| `scoring_method`    | `str`             | No       | `"llm_only"` \| `"rule_based"` \| `"ensemble"` (default)|

## Outputs

```json
{
  "applicant_id": "APL-001",
  "risk_score": 72,
  "risk_tier": "elevated",
  "mortality_rating": 175,
  "sub_scores": {
    "medical": 78,
    "lifestyle": 60,
    "financial": 55,
    "occupational": 40
  },
  "top_factors": [
    { "feature": "hypertension", "impact": +12.3, "direction": "adverse" },
    { "feature": "non_smoker",   "impact":  -5.1, "direction": "favorable" }
  ],
  "confidence": 0.87,
  "scoring_method": "ensemble",
  "model_version": "2.0.0"
}
```

## Risk Tiers

| Tier         | Score Range | Mortality Rating |
|--------------|------------|-----------------|
| preferred    | 0 – 25     | < 100           |
| standard     | 26 – 45    | 100 – 125       |
| rated_mild   | 46 – 60    | 126 – 150       |
| elevated     | 61 – 74    | 151 – 200       |
| high         | 75 – 89    | 201 – 300       |
| decline      | 90 – 100   | > 300           |

## Usage Example (Supervisor ReAct JSON)

```json
{
  "thought_summary": "Entities extracted. Now scoring risk for APL-001.",
  "next_tool": "score_risk",
  "tool_input": {
    "applicant_id": "APL-001",
    "extraction_bundle": { "...": "output from extract_entities" }
  }
}
```

## Notes

- The ensemble method averages LLM confidence-weighted output with rule engine
  scores and flags disagreements > 15 points for human review.
- SHAP-style factor attribution is approximated via Shapley value decomposition
  on the rule sub-scores.
