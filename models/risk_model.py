"""Gradient-boosted ML model for insurance risk scoring.

The model is trained on synthetic data the first time it is instantiated
(a few milliseconds) so no pre-trained artefacts need to be shipped.
A serialised copy can optionally be saved to and loaded from disk.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# Feature schema  (order must stay consistent)
# ------------------------------------------------------------------

FEATURES: List[str] = [
    "age",               # int,   18-80
    "credit_score",      # int,   300-850
    "annual_income",     # float, dollars
    "years_insured",     # int,   0-40
    "num_claims",        # int,   0-10
    "total_claimed",     # float, dollars, all-time
    "coverage_amount",   # float, requested coverage
    "deductible",        # float, requested deductible
]

RISK_THRESHOLDS = {"LOW": 0.35, "MEDIUM": 0.65}  # score < LOW → LOW tier, etc.


@dataclass
class RiskPrediction:
    """Result of a single risk scoring call."""

    score: float          # probability of HIGH-risk class, 0–1
    risk_tier: str        # LOW / MEDIUM / HIGH
    feature_importances: Dict[str, float]
    model_version: str = "1.0"

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "risk_tier": self.risk_tier,
            "feature_importances": {k: round(v, 4) for k, v in self.feature_importances.items()},
            "model_version": self.model_version,
        }


class RiskModel:
    """Gradient-boosted classifier for underwriting risk assessment."""

    def __init__(self, model_path: Optional[str] = None):
        self._clf = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            random_state=42,
        )
        self._scaler = StandardScaler()
        self._trained = False
        self.model_version = "1.0"

        if model_path and Path(model_path).exists():
            self.load(model_path)
        else:
            self._train_on_synthetic_data()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, features: Dict[str, float]) -> RiskPrediction:
        """Return a :class:`RiskPrediction` for *features*.

        Parameters
        ----------
        features:
            Dict with keys matching :data:`FEATURES`.  Missing keys default
            to 0.
        """
        if not self._trained:
            raise RuntimeError("Model has not been trained yet.")

        x = self._dict_to_array(features)
        x_scaled = self._scaler.transform(x)
        prob = float(self._clf.predict_proba(x_scaled)[0, 1])
        tier = self._tier(prob)

        importances = dict(zip(FEATURES, self._clf.feature_importances_))
        return RiskPrediction(
            score=prob,
            risk_tier=tier,
            feature_importances=importances,
            model_version=self.model_version,
        )

    def save(self, path: str) -> None:
        """Persist the trained model to *path* (pickle)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"clf": self._clf, "scaler": self._scaler, "version": self.model_version}, fh)

    def load(self, path: str) -> None:
        """Load a previously saved model from *path*."""
        with open(path, "rb") as fh:
            data = pickle.load(fh)  # noqa: S301 — trusted local file
        self._clf = data["clf"]
        self._scaler = data["scaler"]
        self.model_version = data.get("version", "unknown")
        self._trained = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_on_synthetic_data(self) -> None:
        """Generate ~2 000 synthetic applicants and fit the model."""
        rng = np.random.default_rng(42)
        n = 2000

        age = rng.integers(18, 81, size=n).astype(float)
        credit_score = rng.integers(300, 851, size=n).astype(float)
        annual_income = rng.uniform(20_000, 200_000, size=n)
        years_insured = rng.integers(0, 41, size=n).astype(float)
        num_claims = rng.integers(0, 11, size=n).astype(float)
        total_claimed = num_claims * rng.uniform(500, 15_000, size=n)
        coverage_amount = rng.uniform(10_000, 1_000_000, size=n)
        deductible = rng.uniform(250, 10_000, size=n)

        X = np.column_stack([
            age, credit_score, annual_income, years_insured,
            num_claims, total_claimed, coverage_amount, deductible,
        ])

        # Risk label: high if credit low, many claims, low income
        risk_score = (
            0.35 * (1 - (credit_score - 300) / 550)
            + 0.30 * (num_claims / 10)
            + 0.15 * (1 - np.clip(annual_income / 200_000, 0, 1))
            + 0.10 * (total_claimed / total_claimed.max())
            + 0.10 * rng.uniform(0, 1, size=n)
        )
        y = (risk_score > 0.55).astype(int)

        X_scaled = self._scaler.fit_transform(X)
        self._clf.fit(X_scaled, y)
        self._trained = True

    def _dict_to_array(self, features: Dict[str, float]) -> np.ndarray:
        return np.array([[features.get(f, 0.0) for f in FEATURES]], dtype=float)

    @staticmethod
    def _tier(prob: float) -> str:
        if prob < RISK_THRESHOLDS["LOW"]:
            return "LOW"
        if prob < RISK_THRESHOLDS["MEDIUM"]:
            return "MEDIUM"
        return "HIGH"
