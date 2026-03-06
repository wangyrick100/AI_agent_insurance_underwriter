"""Tests for the IngestionAgent."""

import os
import tempfile

import pytest

from llm.mock_llm import MockLLM
from utils.vector_store import VectorStore
from agents.ingestion_agent import IngestionAgent


@pytest.fixture()
def tmp_vector_store():
    """In-memory vector store for each test."""
    return VectorStore(store_path=None)


@pytest.fixture()
def mock_llm():
    return MockLLM()


@pytest.fixture()
def sample_text_file(tmp_path):
    p = tmp_path / "policy_test.txt"
    p.write_text(
        "HOMEOWNERS INSURANCE POLICY\n"
        "Coverage A — Dwelling: $300,000\n"
        "Deductible: $1,000\n"
        "Exclusions: Flood, Earthquake\n"
    )
    return str(p)


@pytest.fixture()
def sample_directory(tmp_path):
    (tmp_path / "doc1.txt").write_text("Commercial policy. Coverage limit $500,000.")
    (tmp_path / "doc2.txt").write_text("Auto policy. Collision deductible $500.")
    return str(tmp_path)


class TestIngestionAgent:
    def test_ingest_file_adds_to_vector_store(self, tmp_vector_store, mock_llm, sample_text_file):
        agent = IngestionAgent(vector_store=tmp_vector_store, llm=mock_llm)
        doc = agent.ingest_file(sample_text_file)

        assert doc.content  # content is not empty
        assert tmp_vector_store.document_count > 0

    def test_ingest_file_is_idempotent(self, tmp_vector_store, mock_llm, sample_text_file):
        agent = IngestionAgent(vector_store=tmp_vector_store, llm=mock_llm)
        agent.ingest_file(sample_text_file)
        count_after_first = tmp_vector_store.document_count

        agent.ingest_file(sample_text_file)
        assert tmp_vector_store.document_count == count_after_first, (
            "Ingesting the same file twice should not duplicate chunks."
        )

    def test_ingest_directory(self, tmp_vector_store, mock_llm, sample_directory):
        agent = IngestionAgent(vector_store=tmp_vector_store, llm=mock_llm)
        docs = agent.ingest_directory(sample_directory)

        assert len(docs) == 2
        assert agent.ingested_count == 2

    def test_ingested_count_tracks_files(self, tmp_vector_store, mock_llm, sample_directory):
        agent = IngestionAgent(vector_store=tmp_vector_store, llm=mock_llm)
        assert agent.ingested_count == 0
        agent.ingest_directory(sample_directory)
        assert agent.ingested_count == 2

    def test_summarise_without_llm(self, tmp_vector_store, sample_text_file):
        agent = IngestionAgent(vector_store=tmp_vector_store, llm=None)
        doc = agent.ingest_file(sample_text_file)
        summary = agent.summarise(doc)
        assert len(summary) <= 303  # 300 chars + "…"

    def test_summarise_with_mock_llm(self, tmp_vector_store, mock_llm, sample_text_file):
        agent = IngestionAgent(vector_store=tmp_vector_store, llm=mock_llm)
        doc = agent.ingest_file(sample_text_file)
        summary = agent.summarise(doc)
        assert isinstance(summary, str)
        assert len(summary) > 0
