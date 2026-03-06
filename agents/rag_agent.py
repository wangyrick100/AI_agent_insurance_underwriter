"""RAG Agent: retrieval-augmented generation for Q&A over ingested documents."""

from __future__ import annotations

from typing import List, Optional

from llm.base import BaseLLM
from utils.vector_store import VectorStore


class RAGAgent:
    """Answers questions by first retrieving relevant document chunks.

    Parameters
    ----------
    vector_store:
        Shared :class:`VectorStore` instance populated by the
        :class:`~agents.ingestion_agent.IngestionAgent`.
    llm:
        Language model used to synthesise a final answer from retrieved
        context.
    top_k:
        Number of chunks to retrieve per query.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        llm: BaseLLM,
        top_k: int = 4,
    ):
        self.vector_store = vector_store
        self.llm = llm
        self.top_k = top_k

    def query(self, question: str) -> dict:
        """Answer *question* using retrieved context.

        Returns
        -------
        dict
            ``answer`` (str), ``sources`` (list of source paths),
            ``retrieved_chunks`` (list of chunk dicts with score).
        """
        chunks = self.vector_store.query(question, k=self.top_k)

        if not chunks:
            return {
                "answer": "No relevant documents have been ingested yet.",
                "sources": [],
                "retrieved_chunks": [],
            }

        context = self._build_context(chunks)
        prompt = self._build_prompt(question, context)
        answer = self.llm.complete(prompt)

        sources = sorted({c["metadata"].get("source", "unknown") for c in chunks})
        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": chunks,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_context(self, chunks: List[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            src = chunk["metadata"].get("filename", "unknown")
            parts.append(f"[{i}] Source: {src}\n{chunk['text']}")
        return "\n\n".join(parts)

    def _build_prompt(self, question: str, context: str) -> str:
        return (
            "You are an expert insurance underwriter assistant. "
            "Answer the question below using ONLY the provided context. "
            "If the answer is not in the context, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )
