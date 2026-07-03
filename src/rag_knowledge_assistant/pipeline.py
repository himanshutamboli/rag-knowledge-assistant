"""Ingestion + chunking pipeline: corpus -> chunks.jsonl.

Run with:  uv run python -m rag_knowledge_assistant.pipeline
"""

import json
from pathlib import Path

from rag_knowledge_assistant.chunking import DEFAULT_OVERLAP, DEFAULT_SIZE, Chunk, chunk_documents
from rag_knowledge_assistant.ingestion import load_corpus
from rag_knowledge_assistant.logging_config import get_logger

logger = get_logger(__name__)

CHUNKS_PATH = Path("data/chunks.jsonl")


def build_chunks(size: int = DEFAULT_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[Chunk]:
    docs = load_corpus()
    chunks = chunk_documents(docs, size=size, overlap=overlap)
    logger.info(
        "Ingested %d docs -> %d chunks (avg %.0f chars/chunk)",
        len(docs),
        len(chunks),
        sum(c.n_chars for c in chunks) / max(len(chunks), 1),
    )
    return chunks


def main() -> None:
    chunks = build_chunks()
    with CHUNKS_PATH.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c._asdict()) + "\n")
    logger.info("Wrote %d chunks -> %s", len(chunks), CHUNKS_PATH)


if __name__ == "__main__":
    main()
