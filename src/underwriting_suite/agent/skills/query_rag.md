---
name: query_rag
version: "2.0.0"
tool: query_rag
category: retrieval
cost_tier: high
tags:
  - rag
  - retrieval
  - vector-search
  - hybrid-search
  - citations
  - azure-ai-search
---

# Skill: query_rag

## Purpose

Answer questions over ingested underwriting documents using Advanced Retrieval-
Augmented Generation (RAG). Supports multi-hop query decomposition, hybrid
vector + keyword retrieval, cross-encoder re-ranking, answer verification, and
confidence self-assessment. Returns an answer with inline citations to source
document chunks.

## When to Invoke

- The question can be answered from ingested applicant documents (APS, labs,
  physician letters, policy forms).
- Factual lookup over document content is needed without re-extracting entities.
- The user asks "what did the doctor say about …" or "find the policy clause
  regarding …".

## Do NOT Invoke When

- The question requires current web knowledge not in the index – use `research_web`.
- Structured relational queries are needed – use `read_sql`.
- Full entity extraction of a new document is required – use `extract_entities`.

## Inputs

| Field                        | Type        | Required | Description                                               |
|------------------------------|-------------|----------|-----------------------------------------------------------|
| `query`                      | `str`       | Yes      | Question to answer                                        |
| `applicant_id`               | `str`       | No       | Scopes retrieval to this applicant's document index       |
| `doc_scope`                  | `list[str]` | No       | Specific document IDs to search within                    |
| `top_k`                      | `int`       | No       | Number of chunks to retrieve before re-ranking (default: 10)|
| `enable_query_decomposition` | `bool`      | No       | Decompose complex questions into sub-questions (default: true)|

## Outputs

```json
{
  "query": "What medications was the applicant taking as of the last APS?",
  "answer": "According to the APS dated 2024-01-10, the applicant was prescribed Lisinopril 10 mg daily and Metformin 500 mg twice daily.",
  "citations": [
    {
      "doc_id": "doc-abc123",
      "doc_name": "APS_2024_01_10.pdf",
      "chunk_id": "chunk-7",
      "page": 3,
      "excerpt": "Patient is currently on Lisinopril 10 mg OD and Metformin 500 mg BD."
    }
  ],
  "sub_queries": [],
  "answer_confidence": 0.93,
  "retrieval_strategy": "hybrid"
}
```

## Usage Example (Supervisor ReAct JSON)

```json
{
  "thought_summary": "Need medication history from APS documents for APL-001.",
  "next_tool": "query_rag",
  "tool_input": {
    "query": "What medications is applicant APL-001 currently taking?",
    "applicant_id": "APL-001",
    "enable_query_decomposition": false
  }
}
```

## Notes

- Hybrid retrieval combines dense embeddings (Azure AI Search vector index) with
  BM25 keyword scoring; results are merged via Reciprocal Rank Fusion (RRF).
- The cross-encoder re-ranker scores each retrieved chunk against the query
  before passing the top-`k` chunks to the LLM for synthesis.
- Prompt-injection content embedded in document chunks is stripped before answer
  synthesis.
- `answer_confidence` below 0.7 will include a disclaimer recommending human
  document review.
