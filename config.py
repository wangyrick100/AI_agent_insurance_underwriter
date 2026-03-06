"""Central configuration module.

Reads values from environment variables (loaded from .env if present).
Defaults ensure the system runs in mock/demo mode without any API keys.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# OpenAI / LLM
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Run in mock-LLM mode when no API key is present or USE_MOCK_LLM is set
_force_mock = os.getenv("USE_MOCK_LLM", "").lower() in ("1", "true", "yes")
USE_MOCK_LLM: bool = _force_mock or not OPENAI_API_KEY

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
DATABASE_URL: str = os.getenv(
    "DATABASE_URL", "sqlite:///./data/underwriting.db"
)

VECTOR_STORE_PATH: str = os.getenv(
    "VECTOR_STORE_PATH", "./data/vector_store"
)

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_POLICIES_DIR = DATA_DIR / "sample_policies"
