"""Application configuration driven by environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central settings – all values come from env / .env file."""

    # ── Azure OpenAI ─────────────────────────
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-06-01"
    azure_openai_chat_deployment: str = "gpt-4o"
    azure_openai_embedding_deployment: str = "text-embedding-3-large"

    # ── Azure AI Search ──────────────────────
    azure_search_endpoint: str = ""
    azure_search_api_key: str = ""
    azure_search_index_name: str = "underwriting-docs"

    # ── Azure Key Vault ──────────────────────
    azure_keyvault_url: str = ""

    # ── Azure Storage ────────────────────────
    azure_storage_connection_string: str = ""
    azure_storage_container: str = "underwriting-docs"

    # ── Database ─────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./underwriting.db"

    # ── App ──────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:3000"

    # ── Web Research Allowlist ───────────────
    web_allowlist_domains: str = (
        "nih.gov,cdc.gov,who.int,mayoclinic.org,uptodate.com,cms.gov,naic.org"
    )

    # ── APIM ─────────────────────────────────
    apim_gateway_url: str = ""
    apim_subscription_key: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowlisted_domains(self) -> list[str]:
        return [d.strip() for d in self.web_allowlist_domains.split(",") if d.strip()]


settings = Settings()
