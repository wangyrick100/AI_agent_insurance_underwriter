"""Ingestion Agent: load documents, chunk, and index them in the vector store."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from llm.base import BaseLLM
from utils.document_loader import Document, load_directory, load_document
from utils.vector_store import VectorStore


class IngestionAgent:
    """Processes raw documents and indexes them for downstream retrieval.

    Parameters
    ----------
    vector_store:
        Shared :class:`VectorStore` instance.
    llm:
        Optional LLM used to generate summaries of ingested documents.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        llm: Optional[BaseLLM] = None,
    ):
        self.vector_store = vector_store
        self.llm = llm
        self._ingested_sources: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_file(self, path: str) -> Document:
        """Load *path*, index it, and return the :class:`Document`.

        Calling this with the same path twice is a no-op (idempotent).
        """
        resolved = str(Path(path).resolve())
        if resolved in self._ingested_sources:
            # Already indexed — return a quick re-load for the caller
            return load_document(path)

        doc = load_document(path)
        self._index_document(doc)
        self._ingested_sources.append(resolved)
        return doc

    def ingest_directory(self, directory: str, extensions: Optional[List[str]] = None) -> List[Document]:
        """Recursively ingest all documents under *directory*."""
        docs = load_directory(directory, extensions=extensions)
        for doc in docs:
            if doc.source not in self._ingested_sources:
                self._index_document(doc)
                self._ingested_sources.append(doc.source)
        return docs

    def summarise(self, doc: Document) -> str:
        """Use the LLM to produce a short summary of *doc* content."""
        if self.llm is None:
            return doc.content[:300] + ("…" if len(doc.content) > 300 else "")
        prompt = (
            "Extract the key information from this insurance document. "
            "Return a concise summary covering: policy type, coverage limits, "
            "deductible, effective dates, and notable exclusions.\n\n"
            f"Document:\n{doc.content[:4000]}"
        )
        return self.llm.complete(prompt)

    @property
    def ingested_count(self) -> int:
        return len(self._ingested_sources)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index_document(self, doc: Document) -> None:
        self.vector_store.add_documents(
            texts=[doc.content],
            metadatas=[{**doc.metadata, "source": doc.source}],
        )
