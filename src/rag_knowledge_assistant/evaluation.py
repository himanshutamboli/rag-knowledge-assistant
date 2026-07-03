"""Retrieval evaluation harness — recall@k and MRR over a gold question->doc set.

This is what separates a real RAG system from a demo: we *measure* whether
retrieval surfaces the right source, rather than assuming it does. Relevance is
judged at the document level (a retrieved chunk counts for its parent doc).

Metrics:
* recall@k  — fraction of a question's relevant docs found in the top-k retrieved docs.
* MRR       — mean reciprocal rank of the first relevant doc across questions.

Run with:  uv run python -m rag_knowledge_assistant.evaluation
"""

import json
from pathlib import Path
from typing import NamedTuple

from rag_knowledge_assistant.logging_config import get_logger
from rag_knowledge_assistant.retrieval import Retriever, load_retriever

logger = get_logger(__name__)

GOLD_PATH = Path("eval/gold_retrieval.jsonl")
REPORT_PATH = Path("reports/retrieval_eval.md")
KS = (1, 3, 5, 10)
MAX_K = 10


class GoldItem(NamedTuple):
    question: str
    relevant_doc_ids: set[str]


def load_gold(path: Path = GOLD_PATH) -> list[GoldItem]:
    items = []
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            items.append(GoldItem(row["question"], set(row["relevant_doc_ids"])))
    return items


def ranked_doc_ids(retriever: Retriever, query: str, k: int = MAX_K) -> list[str]:
    """Retrieved doc IDs in rank order, de-duplicated (a doc's best chunk sets its rank)."""
    seen: list[str] = []
    for hit in retriever.retrieve(query, k=k):
        if hit.chunk.doc_id not in seen:
            seen.append(hit.chunk.doc_id)
    return seen


def recall_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def reciprocal_rank(relevant: set[str], ranked: list[str]) -> float:
    for i, doc_id in enumerate(ranked, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def evaluate(gold: list[GoldItem], retriever: Retriever) -> dict[str, float]:
    recalls = {k: 0.0 for k in KS}
    rr_total = 0.0
    for item in gold:
        ranked = ranked_doc_ids(retriever, item.question, k=MAX_K)
        for k in KS:
            recalls[k] += recall_at_k(item.relevant_doc_ids, ranked, k)
        rr_total += reciprocal_rank(item.relevant_doc_ids, ranked)
    n = len(gold)
    metrics = {f"recall@{k}": recalls[k] / n for k in KS}
    metrics["mrr"] = rr_total / n
    return metrics


def _markdown(metrics: dict[str, float], n: int) -> str:
    header = "| metric | value |\n| --- | --- |\n"
    rows = "".join(f"| {name} | {value:.3f} |\n" for name, value in metrics.items())
    return (
        "# Retrieval evaluation\n\n"
        f"Gold set: **{n}** questions, document-level relevance. Retriever: TF-IDF + cosine.\n\n"
        + header
        + rows
        + "\n`recall@k` = share of relevant docs in the top-k; `MRR` = mean reciprocal "
        "rank of the first relevant doc.\n"
    )


def main() -> None:
    gold = load_gold()
    metrics = evaluate(gold, load_retriever())
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(_markdown(metrics, len(gold)))
    logger.info("Wrote %s", REPORT_PATH)
    logger.info(" | ".join(f"{name}={value:.3f}" for name, value in metrics.items()))


if __name__ == "__main__":
    main()
