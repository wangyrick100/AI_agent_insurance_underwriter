"""Document loader: reads plain-text and PDF files into a common format."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Document:
    """Represents a loaded document ready for ingestion."""

    source: str
    content: str
    metadata: dict = field(default_factory=dict)


def load_text_file(path: str) -> Document:
    """Load a plain-text file and return a :class:`Document`."""
    p = Path(path)
    with open(p, encoding="utf-8") as fh:
        content = fh.read()
    return Document(
        source=str(p.resolve()),
        content=content,
        metadata={"filename": p.name, "extension": p.suffix.lower()},
    )


def load_pdf_file(path: str) -> Document:
    """Load a PDF file and return a :class:`Document`.

    Requires the ``pypdf`` package.  Falls back to an empty content string
    with a warning if the package is not installed.
    """
    p = Path(path)
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(p))
        pages = [page.extract_text() or "" for page in reader.pages]
        content = "\n".join(pages)
    except ImportError:
        import warnings

        warnings.warn(
            "pypdf not installed; PDF content will be empty. "
            "Install with: pip install pypdf",
            stacklevel=2,
        )
        content = ""
    return Document(
        source=str(p.resolve()),
        content=content,
        metadata={
            "filename": p.name,
            "extension": ".pdf",
        },
    )


def load_document(path: str) -> Document:
    """Load a document from *path*, auto-detecting format."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return load_pdf_file(path)
    return load_text_file(path)


def load_directory(directory: str, extensions: List[str] = None) -> List[Document]:
    """Recursively load all documents from *directory*.

    Parameters
    ----------
    directory:
        Root directory to scan.
    extensions:
        Whitelist of file extensions (e.g. ``[".txt", ".pdf"]``).
        Defaults to ``[".txt", ".pdf", ".md"]``.
    """
    if extensions is None:
        extensions = [".txt", ".pdf", ".md"]

    documents: List[Document] = []
    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            if Path(fname).suffix.lower() in extensions:
                full_path = os.path.join(root, fname)
                try:
                    documents.append(load_document(full_path))
                except Exception as exc:  # noqa: BLE001
                    import warnings

                    warnings.warn(f"Could not load {full_path}: {exc}", stacklevel=2)
    return documents
