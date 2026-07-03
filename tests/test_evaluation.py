import pytest

from rag_knowledge_assistant.chunking import chunk_documents
from rag_knowledge_assistant.embeddings import TfidfEmbedder
from rag_knowledge_assistant.evaluation import (
    GoldItem,
    evaluate,
    recall_at_k,
    reciprocal_rank,
)
from rag_knowledge_assistant.ingestion import Document
from rag_knowledge_assistant.retrieval import Retriever


def test_recall_at_k():
    assert recall_at_k({"a", "b"}, ["a", "c", "b"], 1) == 0.5
    assert recall_at_k({"a", "b"}, ["a", "c", "b"], 3) == 1.0
    assert recall_at_k(set(), ["a"], 3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank({"b"}, ["a", "b", "c"]) == 0.5
    assert reciprocal_rank({"a"}, ["a", "b"]) == 1.0
    assert reciprocal_rank({"z"}, ["a", "b"]) == 0.0


DOCS = [
    Document(
        "overfit",
        "Overfitting",
        "u1",
        "Overfitting is when a model memorizes training noise and fails to generalize.",
    ),
    Document(
        "cosine",
        "Cosine similarity",
        "u2",
        "Cosine similarity measures the angle between two embedding vectors.",
    ),
    Document(
        "kmeans",
        "K-means",
        "u3",
        "K-means clustering partitions points into k groups around centroids.",
    ),
]


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    return Retriever.build(chunk_documents(DOCS, size=300, overlap=50), TfidfEmbedder(min_df=1))


def test_evaluate_on_separable_gold(retriever):
    gold = [
        GoldItem("what is overfitting", {"overfit"}),
        GoldItem("cosine similarity vectors", {"cosine"}),
        GoldItem("k-means clustering centroids", {"kmeans"}),
    ]
    metrics = evaluate(gold, retriever)
    # each question's own doc should rank first on this well-separated set
    assert metrics["recall@1"] == 1.0
    assert metrics["mrr"] == 1.0
