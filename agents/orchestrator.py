"""Multi-agent orchestrator for the insurance underwriting pipeline.

The :class:`UnderwritingOrchestrator` coordinates four specialised agents:

1. :class:`~agents.ingestion_agent.IngestionAgent` — document ingestion
2. :class:`~agents.rag_agent.RAGAgent` — retrieval-augmented Q&A
3. :class:`~agents.sql_agent.SQLAgent` — schema-aware SQL queries
4. :class:`~agents.risk_scoring_agent.RiskScoringAgent` — ML risk scoring

The main entry point is :meth:`process_application`, which executes the full
underwriting pipeline for a new insurance application.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import config
from llm.base import BaseLLM
from llm.factory import create_llm
from utils.database import get_engine, init_db, seed_database
from utils.vector_store import VectorStore

from .ingestion_agent import IngestionAgent
from .rag_agent import RAGAgent
from .risk_scoring_agent import RiskScoringAgent
from .sql_agent import SQLAgent


class UnderwritingOrchestrator:
    """Top-level coordinator for the underwriting multi-agent pipeline.

    Parameters
    ----------
    llm:
        Shared LLM instance.  Defaults to the one returned by
        :func:`~llm.factory.create_llm`.
    vector_store_path:
        Directory for the persistent vector index.
    database_url:
        SQLAlchemy database URL.
    sample_policies_dir:
        Directory of sample policy documents to ingest on first run.
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        vector_store_path: Optional[str] = None,
        database_url: Optional[str] = None,
        sample_policies_dir: Optional[str] = None,
    ):
        self.llm = llm or create_llm()

        # Vector store
        vs_path = vector_store_path or config.VECTOR_STORE_PATH
        self.vector_store = VectorStore(store_path=vs_path)

        # Database
        self.engine = get_engine(database_url)
        init_db(self.engine.url.render_as_string(hide_password=False))
        seed_database(self.engine)

        # Agents
        self.ingestion_agent = IngestionAgent(
            vector_store=self.vector_store,
            llm=self.llm,
        )
        self.rag_agent = RAGAgent(
            vector_store=self.vector_store,
            llm=self.llm,
        )
        self.sql_agent = SQLAgent(
            llm=self.llm,
            engine=self.engine,
        )
        self.risk_scoring_agent = RiskScoringAgent(
            llm=self.llm,
            engine=self.engine,
        )

        # Auto-ingest sample policies if provided and vector store is empty
        if sample_policies_dir and self.vector_store.document_count == 0:
            self.ingestion_agent.ingest_directory(sample_policies_dir)

    # ------------------------------------------------------------------
    # High-level workflow
    # ------------------------------------------------------------------

    def process_application(self, application: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full underwriting pipeline for *application*.

        Steps
        -----
        1. Retrieve similar policy documents from the vector store.
        2. Answer key underwriting questions via RAG.
        3. Pull applicant history from the structured database.
        4. Score risk using the ML model.
        5. Use the LLM to produce a final underwriting decision.

        Parameters
        ----------
        application:
            Dict with applicant data.  Required keys: ``name``,
            ``coverage_amount``, ``policy_type``.  Optional: ``age``,
            ``credit_score``, ``annual_income``, ``years_insured``,
            ``deductible``, ``applicant_id``.

        Returns
        -------
        dict
            Comprehensive underwriting report.
        """
        name = application.get("name", "Applicant")
        policy_type = application.get("policy_type", "General")

        # Step 1: RAG — find similar policy context
        rag_result = self.rag_agent.query(
            f"What are the standard coverage terms and exclusions for a {policy_type} insurance policy?"
        )

        # Step 2: SQL — pull any existing history from the database
        sql_result = self.sql_agent.query(
            f"Show the claims history and risk tier for applicant named '{name}' if they exist, "
            "otherwise show the 5 most recent claims across all applicants."
        )

        # Step 3: ML risk scoring
        risk_result = self.risk_scoring_agent.score(application)

        # Step 4: Final LLM decision
        decision = self._make_decision(application, rag_result, sql_result, risk_result)

        return {
            "application": application,
            "policy_context": {
                "answer": rag_result["answer"],
                "sources": rag_result["sources"],
            },
            "database_context": {
                "sql": sql_result["sql"],
                "rows": sql_result["rows"],
            },
            "risk_assessment": risk_result,
            "underwriting_decision": decision,
        }

    def answer_question(self, question: str) -> str:
        """Answer a free-form question about ingested policies."""
        return self.rag_agent.query(question)["answer"]

    def run_sql_query(self, natural_language: str) -> Dict[str, Any]:
        """Execute a natural-language SQL query against the database."""
        return self.sql_agent.query(natural_language)

    def ingest_policies(self, directory: str) -> int:
        """Ingest all policy documents from *directory*.

        Returns
        -------
        int
            Number of documents ingested in this call.
        """
        before = self.ingestion_agent.ingested_count
        self.ingestion_agent.ingest_directory(directory)
        return self.ingestion_agent.ingested_count - before

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_decision(
        self,
        application: Dict[str, Any],
        rag_result: Dict[str, Any],
        sql_result: Dict[str, Any],
        risk_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ask the LLM for a final underwriting decision."""
        prompt = (
            "You are a senior insurance underwriter. "
            "Based on the following information, provide a final underwriting decision.\n\n"
            f"Application:\n{json.dumps(application, default=str, indent=2)}\n\n"
            f"Policy context summary:\n{rag_result['answer'][:500]}\n\n"
            f"Database history (recent claims):\n{json.dumps(sql_result['rows'][:5], default=str)}\n\n"
            f"ML Risk Score: {risk_result['prediction']['score']:.3f} "
            f"(Tier: {risk_result['prediction']['risk_tier']})\n"
            f"Risk explanation: {risk_result['explanation'][:400]}\n\n"
            "Provide an underwriting decision as JSON with keys: "
            "decision (APPROVED / APPROVED_WITH_CONDITIONS / DECLINED), "
            "premium_loading, conditions (list), rationale."
        )
        raw = self.llm.complete(prompt)
        try:
            # Try to parse as JSON; fall back to wrapping in a dict
            # Strip any markdown fences first
            import re
            cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            return {"decision": "REVIEW_REQUIRED", "rationale": raw}
