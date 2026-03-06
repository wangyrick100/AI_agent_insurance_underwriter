"""Document service – handles ingestion, chunking, and storage."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from underwriting_suite.db.database import async_session
from underwriting_suite.services.azure_openai import get_embeddings
from underwriting_suite.services.azure_search import index_document_chunks

logger = logging.getLogger(__name__)

# Chunk size for splitting documents
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


def _split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


async def ingest_document(
    applicant_id: str,
    filename: str,
    content: str,
    doc_type: str = "unknown",
) -> dict[str, Any]:
    """Ingest a document: chunk, embed, index in Azure AI Search, store metadata.

    Args:
        applicant_id: Applicant the document belongs to.
        filename: Original filename.
        content: Full text content of the document.
        doc_type: Type (aps, meds, labs, vitals, tele_interview, paramedics, application_form).

    Returns:
        Dict with document_id, chunk_count, status.
    """
    doc_id = str(uuid.uuid4())
    logger.info("Ingesting document %s for applicant %s", filename, applicant_id)

    # Split into chunks
    text_chunks = _split_text(content)

    # Generate embeddings and prepare for indexing
    chunks_for_index: list[dict[str, Any]] = []
    for i, chunk_text in enumerate(text_chunks):
        chunk_id = str(uuid.uuid4())
        embedding = await get_embeddings(chunk_text)
        chunks_for_index.append({
            "id": chunk_id,
            "content": chunk_text,
            "chunk_index": i,
            "page_number": i + 1,  # approximate
            "content_vector": embedding,
        })

    # Index in Azure AI Search
    indexed_count = await index_document_chunks(
        chunks=chunks_for_index,
        applicant_id=applicant_id,
        document_id=doc_id,
    )

    # Store document metadata in DB
    try:
        from underwriting_suite.models.document import Document, DocumentChunk
        from sqlalchemy import text

        async with async_session() as session:
            doc = Document(
                id=doc_id,
                applicant_id=applicant_id,
                filename=filename,
                doc_type=doc_type,
                chunk_count=len(text_chunks),
                status="indexed" if indexed_count > 0 else "failed",
            )
            session.add(doc)

            for chunk_data in chunks_for_index:
                chunk = DocumentChunk(
                    id=chunk_data["id"],
                    document_id=doc_id,
                    chunk_index=chunk_data["chunk_index"],
                    page_number=chunk_data.get("page_number"),
                    content=chunk_data["content"],
                    embedding_indexed=indexed_count > 0,
                )
                session.add(chunk)

            await session.commit()
    except Exception as e:
        logger.error("Failed to store document metadata: %s", str(e))

    logger.info("Document %s ingested: %d chunks indexed", doc_id, indexed_count)

    return {
        "document_id": doc_id,
        "filename": filename,
        "doc_type": doc_type,
        "chunk_count": len(text_chunks),
        "indexed_count": indexed_count,
        "status": "indexed" if indexed_count > 0 else "failed",
    }


async def get_document_chunks(doc_id: str) -> list[dict[str, Any]]:
    """Retrieve all chunks for a document.

    Args:
        doc_id: Document identifier.

    Returns:
        List of chunk dicts with id, content, chunk_index, page_number.
    """
    try:
        from underwriting_suite.models.document import DocumentChunk
        from sqlalchemy import select

        async with async_session() as session:
            result = await session.execute(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == doc_id)
                .order_by(DocumentChunk.chunk_index)
            )
            chunks = result.scalars().all()
            return [
                {
                    "id": c.id,
                    "content": c.content,
                    "chunk_index": c.chunk_index,
                    "page_number": c.page_number,
                }
                for c in chunks
            ]
    except Exception as e:
        logger.error("Failed to retrieve chunks for doc %s: %s", doc_id, str(e))
        return []


async def get_snippet(doc_id: str, chunk_id: str) -> dict[str, Any] | None:
    """Get a specific chunk/snippet from a document.

    Args:
        doc_id: Document identifier.
        chunk_id: Chunk identifier.

    Returns:
        Chunk dict or None.
    """
    try:
        from underwriting_suite.models.document import DocumentChunk
        from sqlalchemy import select

        async with async_session() as session:
            result = await session.execute(
                select(DocumentChunk).where(
                    DocumentChunk.id == chunk_id,
                    DocumentChunk.document_id == doc_id,
                )
            )
            chunk = result.scalar_one_or_none()
            if chunk:
                return {
                    "id": chunk.id,
                    "document_id": doc_id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                }
    except Exception as e:
        logger.error("Failed to get snippet %s/%s: %s", doc_id, chunk_id, str(e))
    return None
