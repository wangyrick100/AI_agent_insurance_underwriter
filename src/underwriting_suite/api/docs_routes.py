"""Document upload / ingestion / snippet endpoints."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from underwriting_suite.agent.schemas import IngestRequest
from underwriting_suite.services.document_service import ingest_document, get_snippet

router = APIRouter(prefix="/v1/docs", tags=["documents"])


@router.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    applicant_id: str = Form(...),
    doc_type: str = Form("unknown"),
):
    """Upload and ingest a document – triggers chunking and Azure AI Search indexing."""
    content = (await file.read()).decode("utf-8", errors="replace")
    result = await ingest_document(
        applicant_id=applicant_id,
        filename=file.filename or "untitled",
        content=content,
        doc_type=doc_type,
    )
    return result


@router.get("/{doc_id}/snippets")
async def get_doc_snippet(doc_id: str, chunk_id: str):
    """Return a specific chunk/snippet for the citation viewer."""
    snippet = await get_snippet(doc_id, chunk_id)
    if not snippet:
        raise HTTPException(status_code=404, detail="Snippet not found")
    return snippet
