"""In-process vector store backed by TF-IDF + cosine similarity.

No external services or large model downloads are required.  Documents are
chunked, vectorised with scikit-learn's :class:`TfidfVectorizer`, and
persisted to disk so the index survives process restarts.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    """TF-IDF-based vector store with optional disk persistence.

    Parameters
    ----------
    store_path:
        Directory where the index is saved/loaded.  Pass ``None`` to keep
        the store in memory only.
    chunk_size:
        Approximate character count per text chunk.
    chunk_overlap:
        Number of characters from the end of one chunk that are repeated at
        the start of the next.
    """

    def __init__(
        self,
        store_path: Optional[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        self.store_path = Path(store_path) if store_path else None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._texts: List[str] = []
        self._metadata: List[Dict[str, Any]] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None  # sparse CSR matrix

        if self.store_path and self.store_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def document_count(self) -> int:
        return len(self._texts)

    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Chunk *texts*, add to the store, and rebuild the TF-IDF index."""
        if metadatas is None:
            metadatas = [{} for _ in texts]

        for text, meta in zip(texts, metadatas):
            chunks = self._chunk_text(text)
            for i, chunk in enumerate(chunks):
                self._texts.append(chunk)
                self._metadata.append({**meta, "chunk_index": i, "total_chunks": len(chunks)})

        self._rebuild_index()

        if self.store_path:
            self._save()

    def query(
        self,
        query: str,
        k: int = 4,
    ) -> List[Dict[str, Any]]:
        """Retrieve the *k* most relevant chunks for *query*.

        Returns a list of dicts with keys ``text``, ``score``, and
        ``metadata``.
        """
        if not self._texts or self._vectorizer is None:
            return []

        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        top_k = int(min(k, len(self._texts)))
        indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "text": self._texts[i],
                "score": float(scores[i]),
                "metadata": self._metadata[i],
            }
            for i in indices
        ]

    def clear(self) -> None:
        """Remove all documents from the store."""
        self._texts = []
        self._metadata = []
        self._vectorizer = None
        self._matrix = None
        if self.store_path:
            self._save()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str) -> List[str]:
        """Split *text* into overlapping chunks."""
        if not text.strip():
            return []
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end].strip())
            if end == len(text):
                break
            start = end - self.chunk_overlap
        return [c for c in chunks if c]

    def _rebuild_index(self) -> None:
        if not self._texts:
            return
        self._vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(self._texts)

    def _save(self) -> None:
        self.store_path.mkdir(parents=True, exist_ok=True)
        with open(self.store_path / "texts.json", "w", encoding="utf-8") as fh:
            json.dump({"texts": self._texts, "metadata": self._metadata}, fh)
        if self._vectorizer is not None:
            with open(self.store_path / "vectorizer.pkl", "wb") as fh:
                pickle.dump(self._vectorizer, fh)
            # Sparse matrix as dense numpy array for simplicity
            np.save(
                str(self.store_path / "matrix.npy"),
                self._matrix.toarray(),  # type: ignore[union-attr]
            )

    def _load(self) -> None:
        texts_path = self.store_path / "texts.json"
        vec_path = self.store_path / "vectorizer.pkl"
        mat_path = self.store_path / "matrix.npy"

        if not (texts_path.exists() and vec_path.exists() and mat_path.exists()):
            return

        with open(texts_path, encoding="utf-8") as fh:
            data = json.load(fh)
        self._texts = data["texts"]
        self._metadata = data["metadata"]

        with open(vec_path, "rb") as fh:
            self._vectorizer = pickle.load(fh)  # noqa: S301 — trusted local data

        dense = np.load(str(mat_path))
        from scipy.sparse import csr_matrix  # type: ignore

        self._matrix = csr_matrix(dense)
