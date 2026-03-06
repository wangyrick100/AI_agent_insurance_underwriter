"""Deterministic mock LLM used when no API key is available.

The mock inspects keywords in the prompt to route to appropriate canned
responses, making the full pipeline runnable without network access.
"""

import json
import re
from typing import Optional

from .base import BaseLLM


class MockLLM(BaseLLM):
    """A keyword-driven mock that returns plausible structured responses."""

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        prompt_lower = prompt.lower()

        if self._is_sql_request(prompt_lower):
            return self._sql_response(prompt)

        if self._is_risk_explanation(prompt_lower):
            return self._risk_explanation_response(prompt)

        if self._is_document_extraction(prompt_lower):
            return self._document_extraction_response(prompt)

        if self._is_rag_qa(prompt_lower):
            return self._rag_qa_response(prompt)

        if self._is_underwriting_decision(prompt_lower):
            return self._underwriting_decision_response(prompt)

        return self._generic_response(prompt)

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def _is_sql_request(self, p: str) -> bool:
        return any(kw in p for kw in ("sql", "select", "query the database", "write a query"))

    def _is_risk_explanation(self, p: str) -> bool:
        return any(kw in p for kw in ("explain the risk", "risk score", "risk level", "why is the risk"))

    def _is_document_extraction(self, p: str) -> bool:
        return any(kw in p for kw in ("extract", "summarize the document", "key information", "parse"))

    def _is_rag_qa(self, p: str) -> bool:
        return any(kw in p for kw in ("context:", "based on the following", "according to the documents"))

    def _is_underwriting_decision(self, p: str) -> bool:
        return any(kw in p for kw in ("underwriting decision", "approve", "decline", "final recommendation"))

    # ------------------------------------------------------------------
    # Canned responses
    # ------------------------------------------------------------------

    def _sql_response(self, prompt: str) -> str:
        """Return a simple SELECT query appropriate for the schema."""
        if "claim" in prompt.lower():
            return (
                "SELECT a.name, COUNT(c.id) AS claim_count, SUM(c.amount) AS total_claimed\n"
                "FROM applicants a\n"
                "LEFT JOIN claims c ON a.id = c.applicant_id\n"
                "GROUP BY a.id, a.name\n"
                "ORDER BY total_claimed DESC\n"
                "LIMIT 10;"
            )
        if "high risk" in prompt.lower() or "risk_tier" in prompt.lower():
            return (
                "SELECT id, name, age, credit_score, risk_tier\n"
                "FROM applicants\n"
                "WHERE risk_tier = 'HIGH'\n"
                "ORDER BY credit_score ASC;"
            )
        return (
            "SELECT id, name, age, credit_score, annual_income, risk_tier\n"
            "FROM applicants\n"
            "ORDER BY id\n"
            "LIMIT 10;"
        )

    def _risk_explanation_response(self, prompt: str) -> str:
        score_match = re.search(r"(\d+\.\d+|\d+)%?", prompt)
        score = float(score_match.group(1)) if score_match else 0.42
        level = "HIGH" if score > 0.6 else ("MEDIUM" if score > 0.3 else "LOW")
        return (
            f"Risk Level: {level}\n\n"
            f"The applicant's risk score of {score:.2f} is driven by the following factors:\n"
            "1. Credit score below the preferred threshold increases default probability.\n"
            "2. Multiple prior claims in the last 3 years indicate above-average loss exposure.\n"
            "3. The requested coverage amount is high relative to the declared property value.\n\n"
            f"Recommendation: {'Decline or refer to specialist underwriter.' if level == 'HIGH' else 'Standard terms with possible premium loading.' if level == 'MEDIUM' else 'Accept at standard rates.'}"
        )

    def _document_extraction_response(self, prompt: str) -> str:
        return json.dumps(
            {
                "policy_type": "Commercial General Liability",
                "insured_name": "Acme Corp",
                "coverage_limit": 1000000,
                "deductible": 5000,
                "effective_date": "2025-01-01",
                "expiry_date": "2026-01-01",
                "exclusions": ["Intentional acts", "War and terrorism"],
                "key_clauses": ["Occurrence basis", "Cross-liability"],
            },
            indent=2,
        )

    def _rag_qa_response(self, prompt: str) -> str:
        # Extract the question from the prompt if possible
        q_match = re.search(r"question[:\s]+(.+?)(?:\n|context|$)", prompt, re.IGNORECASE)
        question = q_match.group(1).strip() if q_match else "the query"
        return (
            f"Based on the retrieved policy documents, here is the answer to {question}:\n\n"
            "The policy provides coverage for property damage and bodily injury arising from "
            "business operations. The aggregate limit is $1,000,000 per occurrence and "
            "$2,000,000 in the aggregate. Claims must be reported within 30 days of the "
            "incident. The policy excludes intentional acts, contractual liability unless "
            "specifically endorsed, and losses arising from professional services."
        )

    def _underwriting_decision_response(self, prompt: str) -> str:
        return json.dumps(
            {
                "decision": "APPROVED_WITH_CONDITIONS",
                "premium_loading": "15%",
                "conditions": [
                    "Risk management survey required within 90 days",
                    "Maximum coverage limit capped at $500,000",
                ],
                "rationale": (
                    "Applicant meets minimum eligibility criteria. Elevated claims history "
                    "warrants a premium loading and conditional approval."
                ),
                "reviewed_by": "AI Underwriting Agent v1.0",
            },
            indent=2,
        )

    def _generic_response(self, prompt: str) -> str:
        return (
            "I have reviewed the submitted information. Based on the available data, "
            "the application appears to be within acceptable risk parameters. "
            "Please refer to the detailed risk scoring report for specific metrics."
        )
