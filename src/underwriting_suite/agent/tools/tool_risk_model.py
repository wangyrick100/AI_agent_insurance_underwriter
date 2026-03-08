"""tool_risk_model – ML Risk Scoring Tool.

Computes risk score with confidence, feature rationale, and similar-case
retrieval for underwriter review.

Production capabilities:
  • Feature engineering pipeline  – converts raw X1 extraction bundles into
    structured feature vectors used by both rule-based and LLM scorers.
  • Rule-based debit/credit scoring engine aligned with insurance
    mortality/morbidity tables (SOA 2015 VBT-style).
  • LLM-based scoring for nuanced, multi-factor assessment where rules
    are insufficient.
  • Ensemble combiner  – weighted blend of rule-based and LLM scores
    with configurable strategy (llm_only | rule_based | ensemble).
  • SHAP-style feature-importance ranking with directional indicators.
  • Mortality table rating suggestion (table letter + flat extra).
  • Similar-case retrieval with richer match context.
  • Component sub-scores (medical, lifestyle, financial, occupational).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from underwriting_suite.agent.schemas import (
    FeatureImportance,
    MortalityRating,
    RiskScoreBundle,
    SimilarCase,
    X2Input,
)
from underwriting_suite.services.azure_openai import get_chat_completion

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
#  Feature Engineering
# ═══════════════════════════════════════════════

def _engineer_features(
    feature_payload: dict[str, Any] | None,
    extraction_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a normalised feature dictionary from raw inputs.

    Priority: uses ``extraction_bundle`` entities to populate medical
    features, then overlays any values from ``feature_payload``.
    """
    features: dict[str, Any] = {}

    if extraction_bundle:
        for ent in extraction_bundle.get("entities", []):
            etype = ent.get("entity_type", "")
            ename = ent.get("entity_name", "").lower().replace(" ", "_")
            evalue = ent.get("entity_value")
            is_neg = ent.get("is_negated", False)

            if etype == "vital":
                features[f"vital_{ename}"] = evalue
            elif etype == "lab_result":
                features[f"lab_{ename}"] = evalue
            elif etype == "diagnosis":
                key = f"dx_{ename}"
                features[key] = "absent" if is_neg else (evalue or "present")
            elif etype == "medication":
                features[f"med_{ename}"] = evalue or "prescribed"
            elif etype == "demographic":
                features[ename] = evalue
            elif etype == "lifestyle":
                features[f"lifestyle_{ename}"] = evalue
            elif etype == "family_history":
                features[f"fhx_{ename}"] = evalue or "positive"
            elif etype == "procedure":
                features[f"proc_{ename}"] = evalue or "yes"

    # Overlay explicit feature_payload
    if feature_payload:
        features.update(feature_payload)

    return features


# ═══════════════════════════════════════════════
#  Rule-Based Scoring Engine
# ═══════════════════════════════════════════════

# Debit/credit table inspired by SOA 2015 VBT & standard
# underwriting manuals.  Each rule returns (debit_points, reason).

_DEBIT_RULES: list[tuple[str, Any, float, str]] = [
    # (feature_prefix_or_key, trigger_value, debit, reason)
    ("dx_diabetes", "present", 20.0, "Diabetes diagnosis +20"),
    ("dx_hypertension", "present", 12.0, "Hypertension +12"),
    ("dx_coronary_artery_disease", "present", 30.0, "CAD +30"),
    ("dx_cancer", "present", 35.0, "Cancer history +35"),
    ("dx_depression", "present", 8.0, "Depression +8"),
    ("dx_copd", "present", 18.0, "COPD +18"),
    ("dx_sleep_apnea", "present", 6.0, "Sleep apnoea +6"),
    ("lifestyle_smoking_status", "current", 25.0, "Current smoker +25"),
    ("lifestyle_smoking_status", "former", 8.0, "Former smoker +8"),
    ("fhx_", None, 5.0, "Family history item +5"),
    ("lifestyle_alcohol_use", "heavy", 15.0, "Heavy alcohol use +15"),
    ("lifestyle_drug_use", "current", 20.0, "Current drug use +20"),
]


def _rule_based_score(features: dict[str, Any]) -> tuple[float, list[FeatureImportance]]:
    """Apply deterministic debit/credit rules and return (score, importances)."""
    base_score = 25.0  # baseline (preferred territory)
    importances: list[FeatureImportance] = []

    for key, trigger, debit, reason in _DEBIT_RULES:
        for fkey, fval in features.items():
            if not fkey.startswith(key):
                continue
            fval_str = str(fval).lower().strip()
            if trigger is None or fval_str == str(trigger).lower():
                base_score += debit
                importances.append(FeatureImportance(
                    feature=fkey, value=fval, contribution=debit, direction="increases_risk"
                ))

    # BMI debit
    bmi = features.get("vital_bmi") or features.get("bmi")
    if bmi:
        try:
            bmi_f = float(bmi)
            if bmi_f >= 40:
                d = 20.0
            elif bmi_f >= 35:
                d = 12.0
            elif bmi_f >= 30:
                d = 6.0
            elif bmi_f < 18.5:
                d = 8.0
            else:
                d = 0.0
            if d:
                base_score += d
                importances.append(FeatureImportance(
                    feature="bmi", value=bmi_f, contribution=d, direction="increases_risk"
                ))
        except (ValueError, TypeError):
            pass

    # Clamp 0-100
    return max(0.0, min(100.0, base_score)), importances


# ═══════════════════════════════════════════════
#  LLM Scoring
# ═══════════════════════════════════════════════

SCORING_SYSTEM_PROMPT = """\
You are AgentX2RiskModel, an insurance-underwriting risk scoring engine.

TASK:
Given applicant features (extracted medical entities, demographics, etc.),
produce a risk assessment.

OUTPUT FORMAT (strict JSON):
{{
  "score": 0-100,
  "confidence": 0.0-1.0,
  "risk_class": "preferred_plus|preferred|standard_plus|standard|substandard|decline",
  "sub_scores": {{
    "medical": 0-100,
    "lifestyle": 0-100,
    "financial": 0-100,
    "occupational": 0-100
  }},
  "feature_importance": [
    {{"feature": "...", "value": "...", "contribution": ±float, \
"direction": "increases_risk|decreases_risk|neutral"}}
  ],
  "mortality_rating": {{
    "table_rating": "A-P or +pct or null",
    "flat_extra": null,
    "duration_years": null
  }},
  "similar_cases": [
    {{"case_id": "ANON-xxx", "similarity": 0.0-1.0, "risk_class": "...", \
"reason": "...", "key_features": {{}}}}
  ],
  "calibration_note": "brief note on scoring confidence or caveats"
}}

SCORING GUIDE:
  0-20 = preferred_plus, 21-30 = preferred, 31-45 = standard_plus,
  46-55 = standard, 56-75 = substandard, 76-100 = decline.

RULES:
1. Provide ≥5 feature-importance entries ranked by absolute contribution.
2. Provide 2-4 similar anonymised comparison cases.
3. Sub-scores capture component risk dimensions independently.
4. mortality_rating follows life-insurance Table Rating convention.
5. This is ADVISORY only – never make a final underwriting decision.
"""


async def _llm_score(
    features: dict[str, Any],
    applicant_id: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Get risk scores from the LLM."""
    features_text = json.dumps(features, indent=2, default=str)
    messages = [
        {"role": "system", "content": SCORING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Applicant ID: {applicant_id}\n\n"
                f"Features / extracted entities:\n{features_text}\n\n"
                "Produce a comprehensive risk score assessment."
            ),
        },
    ]
    raw = await get_chat_completion(
        messages, temperature=0.1, response_format="json_object", session_id=session_id
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error("X2: LLM returned invalid JSON")
        return {}


# ═══════════════════════════════════════════════
#  Ensemble Combiner
# ═══════════════════════════════════════════════

_RISK_CLASS_THRESHOLDS = [
    (20, "preferred_plus"),
    (30, "preferred"),
    (45, "standard_plus"),
    (55, "standard"),
    (75, "substandard"),
    (100, "decline"),
]


def _classify(score: float) -> str:
    for threshold, label in _RISK_CLASS_THRESHOLDS:
        if score <= threshold:
            return label
    return "decline"


# ═══════════════════════════════════════════════
#  Public entry point
# ═══════════════════════════════════════════════

async def score_risk(input_data: dict[str, Any]) -> dict[str, Any]:
    """Compute risk score for an applicant.

    Pipeline:
      1. Feature engineering from extraction bundle + payload
      2. Rule-based scoring (debit/credit)
      3. LLM-based scoring (contextual)
      4. Ensemble blend (configurable)
      5. Assemble RiskScoreBundle

    Args:
        input_data: Must conform to X2Input schema.

    Returns:
        Serialised RiskScoreBundle.
    """
    parsed = X2Input(**input_data)
    session_id = input_data.get("_session_id")
    applicant_id = parsed.applicant_id or "unknown"
    method = parsed.scoring_method or "ensemble"
    logger.info("X2 scoring started | applicant=%s method=%s", applicant_id, method)

    # ── 1. Feature engineering ──────────────────────
    features = _engineer_features(parsed.feature_payload, parsed.extraction_bundle)
    if not features:
        features = parsed.feature_payload or {}

    # ── 2. Rule-based score ─────────────────────────
    rule_score, rule_importances = _rule_based_score(features)

    # ── 3. LLM score ───────────────────────────────
    if method in ("llm_only", "ensemble"):
        llm_result = await _llm_score(features, applicant_id, session_id)
    else:
        llm_result = {}

    # ── 4. Ensemble blend ───────────────────────────
    llm_score_val = float(llm_result.get("score", rule_score))
    if method == "ensemble":
        final_score = round(0.4 * rule_score + 0.6 * llm_score_val, 1)
        scoring_label = "ensemble"
    elif method == "rule_based":
        final_score = rule_score
        scoring_label = "rule_based"
    else:
        final_score = llm_score_val
        scoring_label = "llm_only"

    final_class = _classify(final_score)
    confidence = float(llm_result.get("confidence", 0.6 if method == "rule_based" else 0.5))

    # ── Sub-scores ──────────────────────────────────
    sub_scores = llm_result.get("sub_scores", {
        "medical": round(rule_score, 1),
        "lifestyle": 0.0,
        "financial": 0.0,
        "occupational": 0.0,
    })

    # ── Feature importance (merge) ──────────────────
    llm_importances = [
        FeatureImportance(**fi) for fi in llm_result.get("feature_importance", [])
    ]
    # Merge: LLM first, then rule-based (deduplicated)
    seen = {fi.feature for fi in llm_importances}
    combined_importances = list(llm_importances)
    for ri in rule_importances:
        if ri.feature not in seen:
            combined_importances.append(ri)
    combined_importances.sort(key=lambda x: abs(x.contribution), reverse=True)

    # ── Mortality rating ────────────────────────────
    mort_data = llm_result.get("mortality_rating")
    mortality = MortalityRating(**mort_data) if mort_data else None

    # ── Similar cases ───────────────────────────────
    similar = [SimilarCase(**c) for c in llm_result.get("similar_cases", [])]

    # Legacy feature_rationale for backward compat
    feature_rationale = {fi.feature: fi.contribution for fi in combined_importances[:10]}

    bundle = RiskScoreBundle(
        applicant_id=applicant_id,
        score=final_score,
        confidence=confidence,
        risk_class=final_class,
        sub_scores=sub_scores,
        feature_importance=combined_importances,
        feature_rationale=feature_rationale,
        mortality_rating=mortality,
        similar_cases=similar,
        scoring_method=scoring_label,
        calibration_note=llm_result.get("calibration_note"),
    )
    logger.info(
        "X2 scoring complete | score=%.1f class=%s method=%s confidence=%.2f",
        bundle.score, bundle.risk_class, scoring_label, confidence,
    )
    return bundle.model_dump()
