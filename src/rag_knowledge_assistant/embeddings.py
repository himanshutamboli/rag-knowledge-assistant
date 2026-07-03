"""Embedders behind a small protocol, so retrievers can be swapped and compared.

Default: TF-IDF (lexical). It's deterministic, offline, and needs no model
downloads — ideal for reproducible CI and as the baseline the eval harness (Day 18)
measures against. A dense embedder (e.g. sentence-transformers) can implement the
same protocol and be dropped in; the eval harness is what justifies that upgrade.
"""

from typing import Protocol, runtime_checkable

from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer


@runtime_checkable
class Embedder(Protocol):
    def fit(self, texts: list[str]) -> "Embedder": ...
    def transform(self, texts: list[str]): ...


class TfidfEmbedder:
    """Lexical embedder. Output rows are L2-normalized, so a dot product == cosine."""

    def __init__(self, ngram_range: tuple[int, int] = (1, 2), min_df: int = 2) -> None:
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=ngram_range, min_df=min_df
        )

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: list[str]) -> spmatrix:
        return self.vectorizer.transform(texts)
