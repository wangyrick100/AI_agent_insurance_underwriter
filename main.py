"""Demo entry point for the AI Insurance Underwriting Pipeline.

Run:
    python main.py

The script demonstrates the full pipeline end-to-end using the mock LLM so no
API key is required.  Set OPENAI_API_KEY in your .env file (or environment) and
remove USE_MOCK_LLM to switch to the real OpenAI model.
"""

import json
import os
import sys

# Ensure the project root is on sys.path when running directly
sys.path.insert(0, os.path.dirname(__file__))

import config
from agents.orchestrator import UnderwritingOrchestrator


def banner(title: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def section(title: str) -> None:
    print(f"\n--- {title} ---")


def main() -> None:
    banner("AI Insurance Underwriting Pipeline — Demo")

    llm_mode = "MOCK (no API key required)" if config.USE_MOCK_LLM else f"OpenAI ({config.OPENAI_MODEL})"
    print(f"\nLLM mode : {llm_mode}")
    print(f"Database : {config.DATABASE_URL}")
    print(f"Vec store: {config.VECTOR_STORE_PATH}")

    # ------------------------------------------------------------------ #
    # 1. Initialise orchestrator (seeds DB + ingests sample policies)     #
    # ------------------------------------------------------------------ #
    banner("Step 1 — Initialising Orchestrator")
    orch = UnderwritingOrchestrator(
        sample_policies_dir=str(config.SAMPLE_POLICIES_DIR),
    )
    doc_count = orch.vector_store.document_count
    print(f"  Vector store chunks : {doc_count}")
    print(f"  Documents ingested  : {orch.ingestion_agent.ingested_count}")

    # ------------------------------------------------------------------ #
    # 2. RAG Q&A                                                          #
    # ------------------------------------------------------------------ #
    banner("Step 2 — RAG Q&A over Policy Documents")
    questions = [
        "What exclusions are common across commercial and homeowners policies?",
        "What are the reporting requirements when a claim occurs?",
        "What optional coverages are available for auto policies?",
    ]
    for q in questions:
        section(f"Q: {q}")
        answer = orch.answer_question(q)
        print(f"A: {answer[:400]}")

    # ------------------------------------------------------------------ #
    # 3. Schema-aware SQL queries                                         #
    # ------------------------------------------------------------------ #
    banner("Step 3 — Schema-Aware SQL Queries")
    nl_queries = [
        "Show all applicants with their risk tier ordered by credit score",
        "Which applicants have filed more than one claim?",
    ]
    for nq in nl_queries:
        section(f"NL: {nq}")
        result = orch.run_sql_query(nq)
        print(f"SQL: {result['sql']}")
        if result["error"]:
            print(f"Error: {result['error']}")
        else:
            print(f"Rows ({result['row_count']}):")
            for row in result["rows"][:5]:
                print(f"  {row}")

    # ------------------------------------------------------------------ #
    # 4. Full application processing (multi-agent workflow)               #
    # ------------------------------------------------------------------ #
    banner("Step 4 — Full Underwriting Application")
    application = {
        "name": "Jordan Kim",
        "age": 42,
        "occupation": "Small Business Owner",
        "annual_income": 84_000,
        "credit_score": 645,
        "years_insured": 4,
        "num_claims": 2,
        "total_claimed": 14_200,
        "coverage_amount": 750_000,
        "deductible": 5_000,
        "policy_type": "Commercial",
    }
    print("\nApplication:")
    for k, v in application.items():
        print(f"  {k}: {v}")

    report = orch.process_application(application)

    section("Risk Assessment")
    pred = report["risk_assessment"]["prediction"]
    print(f"  Score     : {pred['score']:.4f}")
    print(f"  Risk Tier : {pred['risk_tier']}")
    print(f"  Explanation:\n    {report['risk_assessment']['explanation'][:300]}")

    section("Policy Context (RAG)")
    print(f"  {report['policy_context']['answer'][:400]}")

    section("Database Context")
    print(f"  SQL: {report['database_context']['sql']}")
    for row in report["database_context"]["rows"][:3]:
        print(f"  {row}")

    section("Underwriting Decision")
    decision = report["underwriting_decision"]
    print(json.dumps(decision, indent=4, default=str))

    banner("Demo Complete")


if __name__ == "__main__":
    main()
