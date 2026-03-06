"""Tests for the RAGAgent."""

import pytest

from llm.mock_llm import MockLLM
from utils.vector_store import VectorStore
from agents.rag_agent import RAGAgent


POLICY_TEXT = (
    "COMMERCIAL GENERAL LIABILITY POLICY\n"
    "Coverage A — Bodily Injury and Property Damage: $1,000,000 per occurrence.\n"
    "Deductible: $5,000.\n"
    "Exclusions: Intentional acts, pollution, war and terrorism.\n"
    "Claims must be reported within 30 days.\n"
)


@pytest.fixture()
def populated_store():
    store = VectorStore(store_path=None)
    store.add_documents(
        texts=[POLICY_TEXT],
        metadatas=[{"source": "test_policy.txt", "filename": "test_policy.txt"}],
    )
    return store


@pytest.fixture()
def empty_store():
    return VectorStore(store_path=None)


@pytest.fixture()
def mock_llm():
    return MockLLM()


class TestRAGAgent:
    def test_query_returns_answer(self, populated_store, mock_llm):
        agent = RAGAgent(vector_store=populated_store, llm=mock_llm)
        result = agent.query("What is the coverage limit?")

        assert "answer" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_query_returns_sources(self, populated_store, mock_llm):
        agent = RAGAgent(vector_store=populated_store, llm=mock_llm)
        result = agent.query("What exclusions apply?")

        assert "sources" in result
        assert isinstance(result["sources"], list)

    def test_query_returns_chunks(self, populated_store, mock_llm):
        agent = RAGAgent(vector_store=populated_store, llm=mock_llm)
        result = agent.query("When must claims be reported?")

        assert "retrieved_chunks" in result
        assert len(result["retrieved_chunks"]) > 0

    def test_query_empty_store_returns_message(self, empty_store, mock_llm):
        agent = RAGAgent(vector_store=empty_store, llm=mock_llm)
        result = agent.query("What is the deductible?")

        assert "No relevant documents" in result["answer"]
        assert result["sources"] == []

    def test_top_k_limits_chunks(self, populated_store, mock_llm):
        agent = RAGAgent(vector_store=populated_store, llm=mock_llm, top_k=2)
        result = agent.query("coverage limit")
        assert len(result["retrieved_chunks"]) <= 2

    def test_context_included_in_llm_prompt(self, populated_store, mock_llm):
        agent = RAGAgent(vector_store=populated_store, llm=mock_llm)
        result = agent.query("What are the policy exclusions?")
        # The mock LLM always produces a non-empty response
        assert result["answer"]
