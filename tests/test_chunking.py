import pytest

from rag_knowledge_assistant.chunking import chunk_documents, chunk_text
from rag_knowledge_assistant.ingestion import Document


def test_chunks_respect_size():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(20))
    chunks = chunk_text(text, size=800, overlap=150)
    assert len(chunks) > 1
    assert all(len(c) <= 800 for c in chunks)
    assert all(c.strip() for c in chunks)


def test_hard_split_has_exact_overlap():
    # a single oversized paragraph is hard-split with exact character overlap
    text = "x" * 2500
    chunks = chunk_text(text, size=800, overlap=150)
    assert len(chunks) >= 3
    for a, b in zip(chunks, chunks[1:], strict=False):
        if len(a) == 800:  # full-size chunk -> next begins with its overlap tail
            assert a[-150:] == b[:150]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("abc", size=100, overlap=100)


def test_chunk_documents_ids_and_indices():
    docs = [
        Document("doc_a", "Doc A", "http://x", "\n\n".join("p " + "w " * 80 for _ in range(6))),
        Document("doc_b", "Doc B", "http://y", "short body of text"),
    ]
    chunks = chunk_documents(docs, size=500, overlap=100)
    a_chunks = [c for c in chunks if c.doc_id == "doc_a"]
    # sequential per-document indices and unique global ids
    assert [c.chunk_index for c in a_chunks] == list(range(len(a_chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.title and c.n_chars == len(c.text) for c in chunks)


def test_deterministic():
    text = "\n\n".join(f"Para {i} " + "token " * 50 for i in range(15))
    assert chunk_text(text) == chunk_text(text)
