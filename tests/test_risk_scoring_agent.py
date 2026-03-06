"""Tests for the RiskScoringAgent and the underlying RiskModel."""

import pytest
from sqlalchemy import create_engine

from llm.mock_llm import MockLLM
from models.risk_model import RiskModel, FEATURES
from utils.database import init_db, seed_database, reset_engine
from agents.risk_scoring_agent import RiskScoringAgent


LOW_RISK_APPLICANT = {
    "age": 35,
    "credit_score": 800,
    "annual_income": 120_000,
    "years_insured": 10,
    "num_claims": 0,
    "total_claimed": 0,
    "coverage_amount": 300_000,
    "deductible": 2_500,
}

HIGH_RISK_APPLICANT = {
    "age": 55,
    "credit_score": 420,
    "annual_income": 28_000,
    "years_insured": 1,
    "num_claims": 7,
    "total_claimed": 85_000,
    "coverage_amount": 900_000,
    "deductible": 250,
}


@pytest.fixture()
def risk_model():
    return RiskModel()


@pytest.fixture()
def db_engine(tmp_path):
    reset_engine()
    url = f"sqlite:///{tmp_path}/test_risk.db"
    engine = create_engine(url)
    init_db(url)
    seed_database(engine)
    yield engine
    engine.dispose()
    reset_engine()


@pytest.fixture()
def agent(db_engine):
    return RiskScoringAgent(llm=MockLLM(), engine=db_engine)


class TestRiskModel:
    def test_predict_returns_prediction(self, risk_model):
        pred = risk_model.predict(LOW_RISK_APPLICANT)
        assert 0.0 <= pred.score <= 1.0
        assert pred.risk_tier in ("LOW", "MEDIUM", "HIGH")

    def test_low_risk_applicant_scores_lower(self, risk_model):
        low = risk_model.predict(LOW_RISK_APPLICANT)
        high = risk_model.predict(HIGH_RISK_APPLICANT)
        assert low.score < high.score, "High-risk applicant should score higher than low-risk."

    def test_feature_importances_sum_near_one(self, risk_model):
        pred = risk_model.predict(LOW_RISK_APPLICANT)
        total = sum(pred.feature_importances.values())
        assert abs(total - 1.0) < 1e-6

    def test_all_features_present_in_importances(self, risk_model):
        pred = risk_model.predict(LOW_RISK_APPLICANT)
        for feat in FEATURES:
            assert feat in pred.feature_importances

    def test_as_dict_has_expected_keys(self, risk_model):
        pred = risk_model.predict(LOW_RISK_APPLICANT)
        d = pred.as_dict()
        assert set(d.keys()) == {"score", "risk_tier", "feature_importances", "model_version"}

    def test_save_and_load(self, risk_model, tmp_path):
        path = str(tmp_path / "model.pkl")
        risk_model.save(path)
        loaded = RiskModel.__new__(RiskModel)
        loaded._clf = None
        loaded._scaler = None
        loaded._trained = False
        loaded.model_version = ""
        loaded.load(path)
        assert loaded._trained
        pred = loaded.predict(LOW_RISK_APPLICANT)
        assert 0 <= pred.score <= 1


class TestRiskScoringAgent:
    def test_score_returns_full_report(self, agent):
        result = agent.score(LOW_RISK_APPLICANT)
        assert "applicant" in result
        assert "prediction" in result
        assert "explanation" in result
        assert "scored_at" in result

    def test_score_prediction_structure(self, agent):
        result = agent.score(HIGH_RISK_APPLICANT)
        pred = result["prediction"]
        assert "score" in pred
        assert "risk_tier" in pred
        assert pred["risk_tier"] in ("LOW", "MEDIUM", "HIGH")

    def test_explanation_is_string(self, agent):
        result = agent.score(LOW_RISK_APPLICANT)
        assert isinstance(result["explanation"], str)
        assert len(result["explanation"]) > 0

    def test_batch_score(self, agent):
        applicants = [LOW_RISK_APPLICANT, HIGH_RISK_APPLICANT]
        results = agent.batch_score(applicants)
        assert len(results) == 2

    def test_score_with_db_lookup(self, agent):
        applicant = {**LOW_RISK_APPLICANT, "id": 2}  # Bob Martinez has claims
        result = agent.score(applicant)
        assert result["prediction"]["score"] >= 0
