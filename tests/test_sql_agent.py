"""Tests for the SQLAgent."""

import pytest
from sqlalchemy import create_engine

from llm.mock_llm import MockLLM
from utils.database import init_db, seed_database, reset_engine
from agents.sql_agent import SQLAgent


@pytest.fixture()
def db_engine(tmp_path):
    """Fresh in-memory SQLite DB for each test."""
    reset_engine()
    url = f"sqlite:///{tmp_path}/test.db"
    engine = create_engine(url)
    init_db(url)
    seed_database(engine)
    yield engine
    engine.dispose()
    reset_engine()


@pytest.fixture()
def mock_llm():
    return MockLLM()


@pytest.fixture()
def sql_agent(mock_llm, db_engine):
    return SQLAgent(llm=mock_llm, engine=db_engine)


class TestSQLAgent:
    def test_query_returns_dict_structure(self, sql_agent):
        result = sql_agent.query("Show all applicants")
        assert "sql" in result
        assert "rows" in result
        assert "row_count" in result
        assert "error" in result

    def test_successful_query_has_no_error(self, sql_agent):
        result = sql_agent.query("List all applicants with their risk tier")
        assert result["error"] is None

    def test_rows_are_returned(self, sql_agent):
        result = sql_agent.query("Show all applicants")
        # The mock LLM generates a SELECT on applicants
        assert isinstance(result["rows"], list)
        assert result["row_count"] == len(result["rows"])

    def test_max_rows_limit(self, mock_llm, db_engine):
        agent = SQLAgent(llm=mock_llm, engine=db_engine, max_rows=3)
        result = agent.query("Show all applicants")
        assert result["row_count"] <= 3

    def test_schema_is_injected(self, mock_llm, db_engine):
        agent = SQLAgent(llm=mock_llm, engine=db_engine)
        schema = agent._get_schema()
        assert "applicants" in schema
        assert "policies" in schema
        assert "claims" in schema
        assert "risk_scores" in schema

    def test_sanitise_removes_markdown_fences(self):
        raw = "```sql\nSELECT * FROM applicants;\n```"
        cleaned = SQLAgent._sanitise_sql(raw)
        assert "```" not in cleaned
        assert cleaned.startswith("SELECT")

    def test_sanitise_adds_semicolon(self):
        raw = "SELECT * FROM applicants"
        cleaned = SQLAgent._sanitise_sql(raw)
        assert cleaned.endswith(";")

    def test_non_select_raises_error(self, sql_agent):
        # Inject a non-SELECT query directly to execute_query
        from utils.database import execute_query
        with pytest.raises(ValueError, match="Only SELECT"):
            execute_query("DROP TABLE applicants", engine=sql_agent.engine)
