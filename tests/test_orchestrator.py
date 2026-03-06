"""Tests for the UnderwritingOrchestrator (integration-level)."""

import pytest
from sqlalchemy import create_engine

from llm.mock_llm import MockLLM
from utils.database import reset_engine
from utils.vector_store import VectorStore
from agents.orchestrator import UnderwritingOrchestrator


SAMPLE_APPLICATION = {
    "name": "Test Applicant",
    "age": 40,
    "credit_score": 720,
    "annual_income": 75_000,
    "years_insured": 5,
    "num_claims": 1,
    "total_claimed": 3_500,
    "coverage_amount": 400_000,
    "deductible": 2_000,
    "policy_type": "Homeowners",
}


@pytest.fixture()
def orchestrator(tmp_path):
    reset_engine()
    db_url = f"sqlite:///{tmp_path}/orch_test.db"
    policies_dir = str(tmp_path / "policies")
    import os
    os.makedirs(policies_dir)
    # Create one policy document
    (tmp_path / "policies" / "sample.txt").write_text(
        "HOMEOWNERS POLICY. Coverage A: $400,000. Deductible: $2,000. "
        "Exclusions: Flood, Earthquake."
    )
    orch = UnderwritingOrchestrator(
        llm=MockLLM(),
        vector_store_path=str(tmp_path / "vs"),
        database_url=db_url,
        sample_policies_dir=policies_dir,
    )
    yield orch
    reset_engine()


class TestUnderwritingOrchestrator:
    def test_process_application_returns_report(self, orchestrator):
        report = orchestrator.process_application(SAMPLE_APPLICATION)
        assert "application" in report
        assert "policy_context" in report
        assert "database_context" in report
        assert "risk_assessment" in report
        assert "underwriting_decision" in report

    def test_policy_context_has_answer(self, orchestrator):
        report = orchestrator.process_application(SAMPLE_APPLICATION)
        assert report["policy_context"]["answer"]

    def test_risk_assessment_has_score(self, orchestrator):
        report = orchestrator.process_application(SAMPLE_APPLICATION)
        score = report["risk_assessment"]["prediction"]["score"]
        assert 0.0 <= score <= 1.0

    def test_underwriting_decision_present(self, orchestrator):
        report = orchestrator.process_application(SAMPLE_APPLICATION)
        decision = report["underwriting_decision"]
        assert isinstance(decision, dict)

    def test_answer_question(self, orchestrator):
        answer = orchestrator.answer_question("What are the policy exclusions?")
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_run_sql_query(self, orchestrator):
        result = orchestrator.run_sql_query("Show all applicants")
        assert "sql" in result
        assert "rows" in result

    def test_ingest_policies_returns_count(self, orchestrator, tmp_path):
        extra_dir = tmp_path / "extra_policies"
        extra_dir.mkdir()
        (extra_dir / "extra.txt").write_text("Extra policy document.")
        count = orchestrator.ingest_policies(str(extra_dir))
        assert count == 1

    def test_database_is_seeded(self, orchestrator):
        result = orchestrator.run_sql_query("Show all applicants")
        assert result["row_count"] > 0
