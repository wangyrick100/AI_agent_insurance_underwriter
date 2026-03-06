# Underwriting Decision Support Suite

A portfolio-grade system demonstrating a real underwriting workflow with
LLM-driven orchestration.  The core features include:

- Document ingestion & medical/underwriting entity extraction (X1 agent).
- RAG Q&A over indexed documents with citations (X6 agent).
- Schema-aware Text-to-SQL for read queries (X4 agent).
- Safe database write planning & commit with user confirmation (X5 agent).
- ML risk scoring with confidence & similar-case retrieval (X2 agent).
- Restricted web research on allowlisted domains (X3 agent).
- "Supervisor" agent implementing ReAct planning via LangGraph; tool
  selection is entirely decided by the LLM, not by hard-coded routing.

## Architecture

- **Backend**: FastAPI app (`underwriting_suite`) deployed to Azure Container
  Apps, fronted by APIM.
- **Agents**: Python modules under `underwriting_suite.agent` expose tools
  with strict JSON schemas, registered in `registry.py`.
- **Supervisor**: `SupervisorReActAgent` uses LangGraph loop (plan, execute,
  reflect, synthesize).  LLM outputs `next_tool` and `tool_input` steps.
- **Database**: SQLite (dev) with SQLAlchemy models and safe write planning.
- **Vector store**: Azure AI Search vector index for RAG.  Embeddings from
  Azure OpenAI.
- **UI**: Next.js front-end (in `ui/`) implements dashboard, applicant
  workspace, chat assistant, and debug/session views.
- **Infra**: Dockerfile, docker-compose for local dev; Bicep + APIM config
  planned.

## Getting Started

1. Copy `.env.example` to `.env` and populate with real Azure/DB values.
2. Run `docker-compose up --build` to start backend (8000) and UI (3000).
3. Access http://localhost:3000 for the Next.js app.
4. Use API at `http://localhost:8000/v1/*` to interact directly.

## Demo Scripts

### "Why routing is LLM-driven"

```text
User: I uploaded a tele-interview note. Extract meds, update the DB, then score the applicant and explain why.
```
1. Supervisor chooses `x1_extract` on the new document.
2. It then opts for `x5_write_plan` and returns a write plan with token.
3. After user confirms, `x5_write_commit` executes the writes.
4. Supervisor then calls `x2_score` for risk scoring.
5. Finally `synthesize` produces a comprehensive summary.

Logs store each `next_tool` decision with timestamps proving the LLM
controlled the flow.

## Development

- Python backend in `src/underwriting_suite`.
- Run backend locally: `uvicorn underwriting_suite.main:app --reload`.
- UI in `ui/`, standard Next.js commands.
- Database migrations: currently using `init_db()`; add Alembic later.

## Testing

Tests live in `tests/` (to be implemented).

## Azure Deployment

- LLMs and embeddings provided by Azure OpenAI/AI Foundry deployments.
- Vector search via Azure AI Search.
- Backend container deployed to Azure Container Apps.
- APIM gateway publishes `/v1/*` with auth & rate limiting.
- Key Vault holds secrets; managed identity used by containers.
- Application Insights for telemetry.

## Evidence & Traceability

Each tool output includes structured JSON with `evidence` or `sources`.
Supervisor stores a trace log (`agent_session_traces` table) with every plan
and tool execution step.  `/v1/sessions/{id}/trace` returns the trace.

## Packaing & Naming

- Python package: `underwriting_suite`.
- Supervisor agent class: `SupervisorReActAgent`.
- Tool agents: `AgentX1Extraction`, `AgentX2RiskModel`, etc.

## TODO

- Implement front-end pages and components.
- Add infrastructure (Bicep/APIM) and deployment docs.
- Add comprehensive tests and CI.
- Optional UI session timeline view.

---

> Decision support only; not for automated underwriting decisions.
