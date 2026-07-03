"""Retrieval layer: embed chunks into a brute-force cosine index and query it.

Brute-force is the right call at this scale (~1.3k chunks): exact, simple, no index
to maintain. Approximate indexes (FAISS/HNSW) only pay off at much larger corpora.

Run with:  uv run python -m rag_knowledge_assistant.retrieval "what is overfitting?"
"""

import sys
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from sklearn.metrics.pairwise import linear_kernel

from rag_knowledge_assistant.chunking import Chunk
from rag_knowledge_assistant.embeddings import Embedder, TfidfEmbedder
from rag_knowledge_assistant.logging_config import get_logger
from rag_knowledge_assistant.pipeline import build_chunks

logger = get_logger(__name__)


class RetrievedChunk(NamedTuple):
    chunk: Chunk
    score: float


@dataclass
class Retriever:
    embedder: Embedder
    chunks: list[Chunk]
    matrix: object  # embedded chunk matrix (sparse), rows L2-normalized

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder | None = None) -> "Retriever":
        embedder = embedder or TfidfEmbedder()
        texts = [c.text for c in chunks]
        embedder.fit(texts)
        return cls(embedder=embedder, chunks=chunks, matrix=embedder.transform(texts))

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        query_vec = self.embedder.transform([query])
        scores = linear_kernel(query_vec, self.matrix).ravel()
        top = np.argsort(-scores)[:k]
        return [RetrievedChunk(self.chunks[i], float(scores[i])) for i in top]


def load_retriever() -> Retriever:
    """Build chunks from the corpus and return a ready-to-query retriever."""
    return Retriever.build(build_chunks())


def main() -> None:
    query = " ".join(sys.argv[1:]) or "what is overfitting?"
    retriever = load_retriever()
    logger.info("Query: %s", query)
    for r in retriever.retrieve(query, k=5):
        logger.info("  %.3f  [%s]  %s…", r.score, r.chunk.title, r.chunk.text[:80])


if __name__ == "__main__":
    main()
