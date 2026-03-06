"""FastAPI application entry point for the Underwriting Decision Support Suite."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from underwriting_suite import __app_name__, __version__
from underwriting_suite.api.router import api_router
from underwriting_suite.config import settings
from underwriting_suite.db.database import init_db

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown events."""
    logger.info("Starting %s v%s", __app_name__, __version__)
    await init_db()
    logger.info("Database tables initialised")
    yield
    logger.info("Shutting down %s", __app_name__)


app = FastAPI(
    title=__app_name__,
    version=__version__,
    description=(
        "LLM-driven Supervisor (ReAct + LangGraph) orchestrating underwriting "
        "agents for extraction, scoring, RAG Q&A, SQL read/write, and web research."
    ),
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all API routes
app.include_router(api_router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": __version__}
