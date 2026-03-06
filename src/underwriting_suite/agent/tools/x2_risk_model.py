"""X2 – ML Risk Scoring Agent.

Computes risk score with confidence, feature rationale, and similar-case
retrieval for underwriter review.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from underwriting_suite.agent.schemas import (
    RiskScoreBundle,
    SimilarCase,
    X2Input,
)
from underwriting_suite.services.azure_openai import get_chat_completion

logger = logging.getLogger(__name__)

SCORING_SYSTEM_PROMPT = """\
You are AgentX2RiskModel, an underwriting risk scoring engine.

TASK:
Given applicant features (extracted medical entities, demographics, etc.),
produce a risk assessment score.

OUTPUT FORMAT (JSON):
{
  "score": 0-100,
  "confidence": 0.0-1.0,
  "risk_class": "preferred|standard|substandard|decline",
  "feature_rationale": {"feature_name": contribution_weight, ...},
  "similar_cases": [
    {"case_id": "ANON-xxx", "similarity": 0.0-1.0, "risk_class": "...", "reason": "..."}
  ]
}

RULES:
1. Score 0-30 = preferred, 31-55 = standard, 56-75 = substandard, 76-100 = decline.
2. Provide at least 3 feature contributions in rationale.
3. Provide 2-3 similar anonymised cases.
4. This is ADVISORY only – never make a final underwriting decision.
"""


async def x2_score(input_data: dict[str, Any]) -> dict[str, Any]:
    """Compute risk score for an applicant.

    Args:
        input_data: Must conform to X2Input schema.

    Returns:
        Serialised RiskScoreBundle.
    """
    parsed = X2Input(**input_data)
    logger.info("X2 scoring started | applicant=%s", parsed.applicant_id)

    # If feature_payload is provided, use it; else build from DB
    features = parsed.feature_payload or {}

    # Build prompt with features
    features_text = json.dumps(features, indent=2) if features else "No pre-built features available."

    messages = [
        {"role": "system", "content": SCORING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Applicant ID: {parsed.applicant_id or 'unknown'}\n\n"
                f"Features / extracted entities:\n{features_text}\n\n"
                "Produce a risk score assessment."
            ),
        },
    ]

    raw = await get_chat_completion(messages, temperature=0.1, response_format="json_object")

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("X2: LLM returned invalid JSON")
        return RiskScoreBundle(
            applicant_id=parsed.applicant_id or "unknown",
            score=50.0,
            confidence=0.0,
            risk_class="standard",
        ).model_dump()

    similar = [
        SimilarCase(**c) for c in result.get("similar_cases", [])
    ]

    bundle = RiskScoreBundle(
        applicant_id=parsed.applicant_id or "unknown",
        score=float(result.get("score", 50)),
        confidence=float(result.get("confidence", 0.5)),
        risk_class=result.get("risk_class", "standard"),
        feature_rationale=result.get("feature_rationale", {}),
        similar_cases=similar,
    )
    logger.info("X2 scoring complete | score=%.1f class=%s", bundle.score, bundle.risk_class)
    return bundle.model_dump()
