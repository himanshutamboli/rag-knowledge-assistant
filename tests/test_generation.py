import pytest

from rag_knowledge_assistant.chunking import chunk_documents
from rag_knowledge_assistant.embeddings import TfidfEmbedder
from rag_knowledge_assistant.generation import generate_answer
from rag_knowledge_assistant.ingestion import Document
from rag_knowledge_assistant.retrieval import Retriever

DOCS = [
    Document(
        "overfit",
        "Overfitting",
        "http://x",
        "Overfitting happens when a model learns noise in the training data and "
        "fails to generalize. Regularization and cross-validation help prevent overfitting.",
    ),
    Document(
        "cosine",
        "Cosine similarity",
        "http://y",
        "Cosine similarity measures the angle between two vectors and is widely used "
        "to compare text embeddings in information retrieval.",
    ),
    Document(
        "kmeans",
        "K-means",
        "http://z",
        "K-means clustering partitions points into k groups by minimizing within-cluster "
        "variance around centroids.",
    ),
]


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    return Retriever.build(chunk_documents(DOCS, size=300, overlap=50), TfidfEmbedder(min_df=1))


def test_grounded_answer_cites_sources(retriever):
    answer = generate_answer(retriever, "what is overfitting?", k=2)
    assert not answer.refused
    assert answer.citations  # at least one source
    assert "[1]" in answer.text  # inline citation marker
    # citation markers are sequential starting at 1
    assert [c.marker for c in answer.citations] == list(range(1, len(answer.citations) + 1))
    # top citation is the overfitting doc
    assert answer.citations[0].chunk_id.startswith("overfit")


def test_refuses_when_unsupported(retriever):
    answer = generate_answer(retriever, "banana bread recipe with chocolate", k=2)
    assert answer.refused
    assert answer.citations == []
    assert "enough information" in answer.text


def test_citations_carry_source_urls(retriever):
    answer = generate_answer(retriever, "cosine similarity vectors", k=2)
    assert not answer.refused
    assert all(c.source_url.startswith("http") for c in answer.citations)
