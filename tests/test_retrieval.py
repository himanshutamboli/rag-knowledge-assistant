import pytest

from rag_knowledge_assistant.chunking import chunk_documents
from rag_knowledge_assistant.embeddings import TfidfEmbedder
from rag_knowledge_assistant.ingestion import Document
from rag_knowledge_assistant.retrieval import Retriever

DOCS = [
    Document(
        "overfit",
        "Overfitting",
        "u1",
        "Overfitting happens when a model learns noise in the training data and "
        "fails to generalize to unseen data. Regularization and cross-validation help.",
    ),
    Document(
        "cosine",
        "Cosine similarity",
        "u2",
        "Cosine similarity measures the angle between two vectors and is widely used "
        "to compare text embeddings in information retrieval.",
    ),
    Document(
        "kmeans",
        "K-means",
        "u3",
        "K-means clustering partitions points into k groups by minimizing within-cluster "
        "variance around centroids.",
    ),
]


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    # min_df=1 for this tiny corpus (default min_df=2 suits the full 1.3k-chunk corpus)
    return Retriever.build(chunk_documents(DOCS, size=300, overlap=50), TfidfEmbedder(min_df=1))


def test_retrieves_relevant_document(retriever):
    top = retriever.retrieve("what is overfitting in machine learning?", k=1)[0]
    assert top.chunk.doc_id == "overfit"


def test_topk_and_scores_descending(retriever):
    results = retriever.retrieve("cosine similarity vectors", k=3)
    assert len(results) == 3
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].chunk.doc_id == "cosine"


def test_deterministic(retriever):
    a = retriever.retrieve("clustering centroids", k=2)
    b = retriever.retrieve("clustering centroids", k=2)
    assert [r.chunk.chunk_id for r in a] == [r.chunk.chunk_id for r in b]
