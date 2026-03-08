"""Application configuration driven by environment variables.

Provides centralised, type-safe settings with validation for every
infrastructure dependency and operational policy used by the suite.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central settings – all values come from env / .env file.

    Grouped into logical sections:
      1. Azure OpenAI (LLM + embeddings)
      2. Azure AI Search (vector store)
      3. Azure Key Vault / Storage
      4. Database
      5. Application lifecycle
      6. Supervisor & agent policies
      7. Token / cost budgets
      8. Retry & circuit-breaker policies
      9. Web-research allowlist & Bing
     10. APIM gateway
     11. Feature flags
     12. Observability / telemetry
    """

    # ═══════════════════════════════════════════
    #  1.  Azure OpenAI
    # ═══════════════════════════════════════════
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-06-01"
    azure_openai_chat_deployment: str = "gpt-4o"
    azure_openai_embedding_deployment: str = "text-embedding-3-large"
    azure_openai_embedding_dimensions: int = 3072

    # ═══════════════════════════════════════════
    #  2.  Azure AI Search
    # ═══════════════════════════════════════════
    azure_search_endpoint: str = ""
    azure_search_api_key: str = ""
    azure_search_index_name: str = "underwriting-docs"
    azure_search_semantic_config: str = "underwriting-semantic"
    azure_search_top_k: int = Field(default=8, ge=1, le=50)

    # ═══════════════════════════════════════════
    #  3.  Azure Key Vault / Storage
    # ═══════════════════════════════════════════
    azure_keyvault_url: str = ""
    azure_storage_connection_string: str = ""
    azure_storage_container: str = "underwriting-docs"

    # ═══════════════════════════════════════════
    #  4.  Database
    # ═══════════════════════════════════════════
    database_url: str = "sqlite+aiosqlite:///./underwriting.db"
    db_pool_size: int = Field(default=5, ge=1, le=50)
    db_max_overflow: int = Field(default=10, ge=0, le=100)
    db_pool_recycle: int = Field(default=3600, description="Seconds before recycling a connection")

    # ═══════════════════════════════════════════
    #  5.  Application lifecycle
    # ═══════════════════════════════════════════
    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:3000"

    # ═══════════════════════════════════════════
    #  6.  Supervisor & agent policies
    # ═══════════════════════════════════════════
    supervisor_max_iterations: int = Field(
        default=15, ge=3, le=50,
        description="Hard ceiling on ReAct loop iterations",
    )
    supervisor_max_consecutive_errors: int = Field(default=3, ge=1, le=10)
    write_plan_ttl_seconds: int = Field(
        default=900, description="Confirmation-token expiry window (seconds)",
    )
    extraction_max_chunks: int = Field(
        default=20, ge=1, le=100,
        description="Max document chunks sent to skill_extraction per call",
    )
    extraction_chunk_char_limit: int = Field(
        default=6000, ge=500, le=16000,
        description="Per-chunk character cap for extraction context",
    )
    sql_max_result_rows: int = Field(
        default=500, ge=10, le=5000,
        description="Hard cap on rows returned by skill_sql_read",
    )
    rag_rerank_top_n: int = Field(
        default=5, ge=1, le=20,
        description="Number of chunks retained after re-ranking",
    )

    # ═══════════════════════════════════════════
    #  7.  Token & cost budgets
    # ═══════════════════════════════════════════
    token_budget_per_session: int = Field(
        default=120_000, ge=10_000, le=1_000_000,
        description="Maximum prompt+completion tokens consumed per agent session",
    )
    max_prompt_tokens: int = Field(
        default=16_000, ge=1_000,
        description="Maximum context-window tokens for a single LLM call",
    )
    max_completion_tokens: int = Field(
        default=4_096, ge=256,
        description="Default max_tokens for chat completions",
    )
    cost_per_1k_prompt_tokens: float = Field(
        default=0.005, description="USD cost per 1 000 prompt tokens (for budget tracking)"
    )
    cost_per_1k_completion_tokens: float = Field(
        default=0.015, description="USD cost per 1 000 completion tokens"
    )

    # ═══════════════════════════════════════════
    #  8.  Retry & circuit-breaker policies
    # ═══════════════════════════════════════════
    llm_retry_attempts: int = Field(default=3, ge=0, le=10)
    llm_retry_backoff_base: float = Field(
        default=1.5, description="Base seconds for exponential back-off"
    )
    llm_retry_backoff_max: float = Field(default=30.0, description="Max back-off seconds")
    circuit_breaker_failure_threshold: int = Field(
        default=5, ge=1,
        description="Consecutive LLM failures before tripping the breaker",
    )
    circuit_breaker_recovery_timeout: int = Field(
        default=60, ge=10,
        description="Seconds to wait before half-open probe after trip",
    )

    # ═══════════════════════════════════════════
    #  9.  Web research allowlist & Bing
    # ═══════════════════════════════════════════
    web_allowlist_domains: str = (
        "nih.gov,cdc.gov,who.int,mayoclinic.org,uptodate.com,cms.gov,naic.org,"
        "pubmed.ncbi.nlm.nih.gov,clinicaltrials.gov,fda.gov,ama-assn.org,"
        "heart.org,cancer.org,diabetes.org,lung.org"
    )
    bing_search_api_key: str = ""
    bing_search_endpoint: str = "https://api.bing.microsoft.com/v7.0/search"
    web_research_max_sources: int = Field(default=8, ge=1, le=20)

    # ═══════════════════════════════════════════
    # 10.  APIM
    # ═══════════════════════════════════════════
    apim_gateway_url: str = ""
    apim_subscription_key: str = ""

    # ═══════════════════════════════════════════
    # 11.  Feature flags
    # ═══════════════════════════════════════════
    enable_entity_normalisation: bool = Field(
        default=True,
        description="Map extracted entities to ICD-10 / RxNorm / SNOMED codes via LLM",
    )
    enable_extraction_verification: bool = Field(
        default=True,
        description="Run a second-pass LLM verification on X1 extraction output",
    )
    enable_rag_reranking: bool = Field(
        default=True, description="Apply cross-encoder style re-ranking in X6 RAG",
    )
    enable_sql_query_review: bool = Field(
        default=True,
        description="LLM self-review of generated SQL before execution in X4",
    )
    enable_parallel_extraction: bool = Field(
        default=True,
        description="Process document chunks in parallel batches for X1",
    )
    enable_bing_live_search: bool = Field(
        default=False,
        description="Use Bing Search API for live web results in X3 (requires key)",
    )
    enable_cost_tracking: bool = Field(
        default=True, description="Track token usage and estimated cost per session",
    )

    # ═══════════════════════════════════════════
    # 12.  Observability / telemetry
    # ═══════════════════════════════════════════
    otel_service_name: str = "underwriting-suite"
    appinsights_connection_string: str = ""
    enable_trace_persistence: bool = Field(
        default=True, description="Persist ReAct trace steps to the database",
    )

    # ───────────────────────────────────────────
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # ── Derived properties ────────────────────

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowlisted_domains(self) -> list[str]:
        return [d.strip() for d in self.web_allowlist_domains.split(",") if d.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in ("production", "prod")

    @property
    def is_development(self) -> bool:
        return self.app_env.lower() in ("development", "dev", "local")

    # ── Validators ────────────────────────────

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return upper


settings = Settings()
