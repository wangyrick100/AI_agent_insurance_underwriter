"""Central API router – aggregates all sub-routers."""

from __future__ import annotations

from fastapi import APIRouter

from underwriting_suite.api.agent_routes import router as agent_router
from underwriting_suite.api.applicant_routes import router as applicant_router
from underwriting_suite.api.docs_routes import router as docs_router
from underwriting_suite.api.extraction_routes import router as extraction_router
from underwriting_suite.api.ml_routes import router as ml_router
from underwriting_suite.api.session_routes import router as session_router
from underwriting_suite.api.sql_routes import router as sql_router

api_router = APIRouter()

api_router.include_router(agent_router)
api_router.include_router(applicant_router)
api_router.include_router(docs_router)
api_router.include_router(extraction_router)
api_router.include_router(ml_router)
api_router.include_router(session_router)
api_router.include_router(sql_router)
