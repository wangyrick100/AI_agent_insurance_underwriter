"""Risk Scoring Agent: combine ML model output with LLM explanation."""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from llm.base import BaseLLM
from models.risk_model import RiskModel, RiskPrediction
from utils.database import execute_query, get_engine, risk_scores


class RiskScoringAgent:
    """Scores insurance applications using the ML model and the LLM.

    Workflow
    --------
    1. Extract features from the applicant dict.
    2. Call :class:`~models.risk_model.RiskModel` to get a numeric score.
    3. (Optionally) Look up prior claims from the database.
    4. Use the LLM to generate a human-readable risk explanation.
    5. Persist the score to the ``risk_scores`` table.

    Parameters
    ----------
    llm:
        Language model used for natural-language explanations.
    risk_model:
        Pre-instantiated (and trained) :class:`RiskModel`.
    engine:
        Optional SQLAlchemy engine.
    """

    def __init__(
        self,
        llm: BaseLLM,
        risk_model: Optional[RiskModel] = None,
        engine=None,
    ):
        self.llm = llm
        self.risk_model = risk_model or RiskModel()
        self.engine = engine

    def score(self, applicant: Dict[str, Any]) -> Dict[str, Any]:
        """Compute risk for *applicant* and return a full risk report.

        Parameters
        ----------
        applicant:
            Dict with any subset of: ``age``, ``credit_score``,
            ``annual_income``, ``years_insured``, ``num_claims``,
            ``total_claimed``, ``coverage_amount``, ``deductible``.
            Additional keys (e.g. ``name``, ``id``) are passed through.

        Returns
        -------
        dict
            Keys: ``applicant``, ``prediction`` (RiskPrediction as dict),
            ``explanation`` (str), ``scored_at`` (ISO timestamp).
        """
        # Augment features from DB if applicant_id is present
        features = self._extract_features(applicant)
        if "applicant_id" in applicant or "id" in applicant:
            db_features = self._fetch_db_features(applicant.get("applicant_id") or applicant.get("id"))
            features.update({k: v for k, v in db_features.items() if k not in features})

        prediction: RiskPrediction = self.risk_model.predict(features)
        explanation = self._explain(applicant, prediction)
        scored_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Persist to DB (best-effort)
        self._persist_score(applicant, prediction, scored_at)

        return {
            "applicant": applicant,
            "prediction": prediction.as_dict(),
            "explanation": explanation,
            "scored_at": scored_at,
        }

    def batch_score(self, applicants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score multiple applicants and return a list of reports."""
        return [self.score(a) for a in applicants]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_features(self, applicant: Dict[str, Any]) -> Dict[str, float]:
        feature_keys = [
            "age", "credit_score", "annual_income", "years_insured",
            "num_claims", "total_claimed", "coverage_amount", "deductible",
        ]
        return {k: float(applicant[k]) for k in feature_keys if k in applicant}

    def _fetch_db_features(self, applicant_id: int) -> Dict[str, float]:
        """Pull claims history from the database for the applicant."""
        from sqlalchemy import text

        try:
            engine = self.engine or get_engine()
            with engine.connect() as conn:
                result = conn.execute(
                    text(
                        "SELECT COUNT(c.id) AS num_claims, "
                        "COALESCE(SUM(c.amount), 0) AS total_claimed "
                        "FROM claims c WHERE c.applicant_id = :aid"
                    ),
                    {"aid": int(applicant_id)},
                )
                row = result.fetchone()
            if row:
                return {
                    "num_claims": float(row[0] or 0),
                    "total_claimed": float(row[1] or 0),
                }
        except Exception:  # noqa: BLE001
            pass
        return {}

    def _explain(self, applicant: Dict[str, Any], prediction: RiskPrediction) -> str:
        prompt = (
            f"Explain the risk score for the following insurance applicant.\n\n"
            f"Applicant: {applicant}\n"
            f"Risk score: {prediction.score:.2f} (tier: {prediction.risk_tier})\n"
            f"Top features by importance: "
            f"{sorted(prediction.feature_importances.items(), key=lambda x: x[1], reverse=True)[:4]}\n\n"
            "Provide a concise underwriting explanation and recommendation."
        )
        return self.llm.complete(prompt)

    def _persist_score(
        self,
        applicant: Dict[str, Any],
        prediction: RiskPrediction,
        scored_at: str,
    ) -> None:
        applicant_id = applicant.get("applicant_id") or applicant.get("id")
        if applicant_id is None:
            return
        try:
            engine = self.engine or get_engine()
            with engine.connect() as conn:
                conn.execute(
                    risk_scores.insert().values(
                        applicant_id=int(applicant_id),
                        score=prediction.score,
                        risk_tier=prediction.risk_tier,
                        scored_at=scored_at,
                        model_version=prediction.model_version,
                    )
                )
                conn.commit()
        except Exception:  # noqa: BLE001
            pass
