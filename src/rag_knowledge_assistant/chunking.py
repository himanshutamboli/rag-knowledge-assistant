"""Chunking — the single biggest lever on retrieval quality.

Strategy: paragraph-aware packing. Greedily pack whole paragraphs up to a target
size so chunks stay semantically coherent; carry an ``overlap`` tail into the next
chunk so context isn't severed at boundaries. Paragraphs longer than the target are
hard-split with the same overlap.

Sizes are in characters (~4 chars/token, so size=800 ≈ 200 tokens).
"""

import re
from typing import NamedTuple

from rag_knowledge_assistant.ingestion import Document

DEFAULT_SIZE = 800
DEFAULT_OVERLAP = 150


class Chunk(NamedTuple):
    chunk_id: str
    doc_id: str
    title: str
    source_url: str
    chunk_index: int
    text: str
    n_chars: int


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    step = size - overlap
    return [text[i : i + size] for i in range(0, len(text), step)]


def chunk_text(text: str, size: int = DEFAULT_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[str]:
    """Split text into overlapping, paragraph-aware chunks of ~``size`` chars."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    chunks: list[str] = []
    current = ""
    for para in _paragraphs(text):
        if len(para) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_split(para, size, overlap))
            continue
        candidate = f"{current}\n{para}".strip() if current else para
        if len(candidate) <= size:
            current = candidate
        else:
            chunks.append(current)
            tail = current[-overlap:]
            current = f"{tail}\n{para}".strip()
    if current:
        chunks.append(current)
    return chunks


def chunk_documents(
    docs: list[Document], size: int = DEFAULT_SIZE, overlap: int = DEFAULT_OVERLAP
) -> list[Chunk]:
    """Chunk every document, producing globally-identified Chunk records."""
    chunks = []
    for doc in docs:
        for i, piece in enumerate(chunk_text(doc.text, size=size, overlap=overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{i}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    source_url=doc.source_url,
                    chunk_index=i,
                    text=piece,
                    n_chars=len(piece),
                )
            )
    return chunks
