"""Grounded answer generation with inline citations, and refusal when unsupported.

A `Generator` turns a query + retrieved chunks into an answer. Two implementations:

* `ExtractiveGenerator` (default) — deterministic, offline, no API key. Stitches the
  most relevant sentences together with inline [n] citations. Used in tests/CI.
* `ClaudeGenerator` — calls the Anthropic API (claude-opus-4-8) for fluent grounded
  answers. Drop-in via the same protocol; used when an API key is available.

Refusal: if the top retrieval score is below a threshold, we refuse rather than
answer from thin air — the behavior that separates a grounded assistant from a
confident hallucinator.

Run with:  uv run python -m rag_knowledge_assistant.generation "what is overfitting?"
"""

import re
import sys
from typing import NamedTuple, Protocol

from rag_knowledge_assistant.chunking import Chunk
from rag_knowledge_assistant.logging_config import get_logger
from rag_knowledge_assistant.retrieval import Retriever, load_retriever

logger = get_logger(__name__)

MIN_SCORE = 0.05  # below this, retrieval is too weak to ground an answer
GROUNDED_SYSTEM = (
    "You are a knowledge assistant. Answer the question using ONLY the numbered "
    "sources provided. Cite each claim with its source number like [1]. If the "
    "sources do not contain the answer, say you cannot answer from the sources."
)


class Citation(NamedTuple):
    marker: int
    chunk_id: str
    title: str
    source_url: str
    score: float


class Answer(NamedTuple):
    query: str
    text: str
    refused: bool
    citations: list[Citation]


class Generator(Protocol):
    def generate(self, query: str, contexts: list[Chunk]) -> str: ...


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", text.lower()))


def _select_sentence(text: str, query: str, max_chars: int = 280) -> str:
    """Pick the sentence with the most query-term overlap (avoids mid-chunk fragments)."""
    query_terms = _terms(query)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 30]
    if not sentences:
        return text.strip()[:max_chars]
    best = max(sentences, key=lambda s: len(query_terms & _terms(s)))
    return best[:max_chars].strip()


class ExtractiveGenerator:
    """Deterministic, offline generator: the most query-relevant sentence per source,
    with inline [n] citations. A no-LLM baseline used in tests/CI."""

    def generate(self, query: str, contexts: list[Chunk]) -> str:
        parts = [f"{_select_sentence(c.text, query)} [{i}]" for i, c in enumerate(contexts, 1)]
        return " ".join(parts)


class ClaudeGenerator:
    """Grounded generation via the Anthropic API (claude-opus-4-8)."""

    def __init__(self, model: str = "claude-opus-4-8", max_tokens: int = 1024) -> None:
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, query: str, contexts: list[Chunk]) -> str:
        import anthropic  # lazy: only needed when actually calling the API

        sources = "\n\n".join(f"[{i}] {c.title}: {c.text}" for i, c in enumerate(contexts, 1))
        user = f"Sources:\n{sources}\n\nQuestion: {query}"
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=GROUNDED_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


def generate_answer(
    retriever: Retriever,
    query: str,
    k: int = 4,
    min_score: float = MIN_SCORE,
    generator: Generator | None = None,
) -> Answer:
    generator = generator or ExtractiveGenerator()
    hits = retriever.retrieve(query, k=k)
    if not hits or hits[0].score < min_score:
        return Answer(
            query=query,
            text="I don't have enough information in the knowledge base to answer that.",
            refused=True,
            citations=[],
        )
    contexts = [h.chunk for h in hits]
    text = generator.generate(query, contexts)
    citations = [
        Citation(i, h.chunk.chunk_id, h.chunk.title, h.chunk.source_url, h.score)
        for i, h in enumerate(hits, 1)
    ]
    return Answer(query=query, text=text, refused=False, citations=citations)


def main() -> None:
    query = " ".join(sys.argv[1:]) or "what is overfitting?"
    answer = generate_answer(load_retriever(), query)
    logger.info("Q: %s", answer.query)
    logger.info("A: %s", answer.text)
    if answer.refused:
        return
    for c in answer.citations:
        logger.info("  [%d] %.3f %s (%s)", c.marker, c.score, c.title, c.source_url)


if __name__ == "__main__":
    main()
