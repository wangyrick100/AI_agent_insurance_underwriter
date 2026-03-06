# AI Insurance Underwriting Agent

A portfolio system demonstrating an **LLM-driven underwriting pipeline** with multi-agent orchestration, document ingestion, RAG Q&A, schema-aware SQL, and ML risk scoring.

---

## Architecture

```
UnderwritingOrchestrator
├── IngestionAgent      — load & chunk documents → TF-IDF vector store
├── RAGAgent            — retrieve relevant chunks → LLM synthesis
├── SQLAgent            — NL → schema-aware SQL → execute against SQLite
└── RiskScoringAgent    — scikit-learn GradientBoosting → ML risk score + LLM explanation
```

### LLM Layer

| Mode | Trigger | Implementation |
|------|---------|----------------|
| **Mock** (default) | No `OPENAI_API_KEY` set | Keyword-routing `MockLLM` — no API calls, runs offline |
| **OpenAI** | `OPENAI_API_KEY` set | `OpenAILLM` wrapping the Chat Completions API |

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) configure OpenAI
cp .env.example .env
# edit .env and set OPENAI_API_KEY

# 3. Run the end-to-end demo
python main.py
```

### Run tests

```bash
pytest
```

All 39 tests pass without an API key.

---

## Components

### `llm/`
- `base.py` — abstract `BaseLLM` interface (`complete(prompt, system) → str`)
- `mock_llm.py` — deterministic mock with keyword-routing for offline dev/test
- `openai_llm.py` — thin wrapper around `openai.OpenAI`
- `factory.py` — returns the right implementation based on `config.USE_MOCK_LLM`

### `utils/`
- `document_loader.py` — load `.txt` / `.pdf` / `.md` files into `Document` objects
- `vector_store.py` — TF-IDF + cosine similarity vector store with disk persistence (no external service needed)
- `database.py` — SQLAlchemy schema (applicants, policies, claims, risk_scores), seed data, `execute_query` helper

### `models/`
- `risk_model.py` — `GradientBoostingClassifier` trained on 2 000 synthetic applicants; features: age, credit score, income, claims history, coverage amount, deductible

### `agents/`
| Agent | Responsibility |
|-------|---------------|
| `IngestionAgent` | Load docs, chunk, add to vector store (idempotent) |
| `RAGAgent` | Embed query, retrieve top-k chunks, LLM synthesis |
| `SQLAgent` | Inject schema DDL into LLM prompt, sanitise SQL, execute |
| `RiskScoringAgent` | Extract features, ML score, DB claims lookup, LLM explanation, persist score |
| `UnderwritingOrchestrator` | Coordinate all four agents; `process_application()` runs the full pipeline |

### `data/sample_policies/`
Three realistic insurance policy documents (commercial CGL, homeowners HO-3, personal auto) used as the RAG corpus.

---

## Pipeline flow — `process_application()`

```
1. RAG → retrieve relevant policy terms for the requested coverage type
2. SQL → pull applicant claims history from the structured database
3. ML  → GradientBoosting risk score + feature importances
4. LLM → synthesise a final underwriting decision (APPROVED / APPROVED_WITH_CONDITIONS / DECLINED)
```

---

## Configuration (`config.py` / `.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(empty)* | OpenAI key; absent → mock mode |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model to use |
| `USE_MOCK_LLM` | auto | Override to `true` to force mock |
| `DATABASE_URL` | `sqlite:///./data/underwriting.db` | SQLAlchemy DB URL |
| `VECTOR_STORE_PATH` | `./data/vector_store` | TF-IDF index directory |

---

## Project structure

```
.
├── agents/
│   ├── ingestion_agent.py
│   ├── orchestrator.py
│   ├── rag_agent.py
│   ├── risk_scoring_agent.py
│   └── sql_agent.py
├── data/
│   └── sample_policies/
│       ├── auto_policy_003.txt
│       ├── commercial_policy_001.txt
│       └── homeowners_policy_002.txt
├── llm/
│   ├── base.py
│   ├── factory.py
│   ├── mock_llm.py
│   └── openai_llm.py
├── models/
│   └── risk_model.py
├── tests/
│   ├── test_ingestion_agent.py
│   ├── test_orchestrator.py
│   ├── test_rag_agent.py
│   ├── test_risk_scoring_agent.py
│   └── test_sql_agent.py
├── utils/
│   ├── database.py
│   ├── document_loader.py
│   └── vector_store.py
├── config.py
├── main.py
├── requirements.txt
└── .env.example
```