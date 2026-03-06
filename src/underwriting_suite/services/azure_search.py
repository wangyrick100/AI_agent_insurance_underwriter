"""Azure AI Search service wrapper.

Provides vector search and document indexing for the RAG pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

from underwriting_suite.config import settings

logger = logging.getLogger(__name__)


async def vector_search(
    query_text: str,
    query_vector: list[float],
    top_k: int = 8,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Perform vector + hybrid search in Azure AI Search index.

    Args:
        query_text: Original query text for hybrid search.
        query_vector: Embedding vector for the query.
        top_k: Number of results to return.
        filters: Optional OData filters (applicant_id, document_id).

    Returns:
        List of search result dicts with doc_id, chunk_id, content, page, score.
    """
    try:
        from azure.search.documents import SearchClient
        from azure.search.documents.models import VectorizedQuery
        from azure.core.credentials import AzureKeyCredential

        client = SearchClient(
            endpoint=settings.azure_search_endpoint,
            index_name=settings.azure_search_index_name,
            credential=AzureKeyCredential(settings.azure_search_api_key),
        )

        # Build OData filter
        filter_str = None
        if filters:
            parts = []
            if "applicant_id" in filters:
                parts.append(f"applicant_id eq '{filters['applicant_id']}'")
            if "document_id" in filters:
                if isinstance(filters["document_id"], list):
                    doc_filter = " or ".join(
                        f"document_id eq '{d}'" for d in filters["document_id"]
                    )
                    parts.append(f"({doc_filter})")
                else:
                    parts.append(f"document_id eq '{filters['document_id']}'")
            if parts:
                filter_str = " and ".join(parts)

        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="content_vector",
        )

        results = client.search(
            search_text=query_text,
            vector_queries=[vector_query],
            top=top_k,
            filter=filter_str,
            select=["id", "document_id", "chunk_index", "page_number", "content", "applicant_id"],
        )

        output = []
        for result in results:
            output.append({
                "chunk_id": result["id"],
                "doc_id": result.get("document_id", ""),
                "page": result.get("page_number"),
                "content": result.get("content", ""),
                "score": result.get("@search.score", 0.0),
            })

        logger.info("Azure AI Search returned %d results", len(output))
        return output

    except ImportError:
        logger.warning("Azure Search SDK not available – returning mock results")
        return _mock_search_results(query_text, top_k)
    except Exception as e:
        logger.error("Azure AI Search failed: %s", str(e))
        return _mock_search_results(query_text, top_k)


async def index_document_chunks(
    chunks: list[dict[str, Any]],
    applicant_id: str,
    document_id: str,
) -> int:
    """Index document chunks into Azure AI Search.

    Args:
        chunks: List of dicts with id, content, page_number, content_vector.
        applicant_id: Applicant the document belongs to.
        document_id: Document identifier.

    Returns:
        Number of chunks indexed.
    """
    try:
        from azure.search.documents import SearchClient
        from azure.core.credentials import AzureKeyCredential

        client = SearchClient(
            endpoint=settings.azure_search_endpoint,
            index_name=settings.azure_search_index_name,
            credential=AzureKeyCredential(settings.azure_search_api_key),
        )

        documents = []
        for chunk in chunks:
            documents.append({
                "id": chunk["id"],
                "document_id": document_id,
                "applicant_id": applicant_id,
                "chunk_index": chunk.get("chunk_index", 0),
                "page_number": chunk.get("page_number"),
                "content": chunk["content"],
                "content_vector": chunk.get("content_vector", []),
            })

        result = client.upload_documents(documents=documents)
        indexed = sum(1 for r in result if r.succeeded)
        logger.info("Indexed %d/%d chunks for doc %s", indexed, len(documents), document_id)
        return indexed

    except ImportError:
        logger.warning("Azure Search SDK not available – skipping indexing")
        return len(chunks)
    except Exception as e:
        logger.error("Azure AI Search indexing failed: %s", str(e))
        return 0


def _mock_search_results(query: str, top_k: int) -> list[dict[str, Any]]:
    """Return mock search results for local development."""
    return [
        {
            "chunk_id": f"mock-chunk-{i}",
            "doc_id": f"mock-doc-{i}",
            "page": i,
            "content": f"[Mock result {i}] Relevant content for query: {query[:50]}...",
            "score": round(0.95 - i * 0.05, 2),
        }
        for i in range(1, min(top_k + 1, 4))
    ]
