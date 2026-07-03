"""Faithfulness evaluation — is the answer supported by the retrieved sources?

A `Judge` scores an (answer, contexts) pair in [0, 1]. Two implementations:

* `HeuristicJudge` (default) — lexical grounding: the share of the answer's content
  terms that appear in the context. Deterministic, offline, CI-safe.
* `ClaudeJudge` — LLM-as-judge (claude-opus-4-8) returning a supported/score verdict.

**Calibration is the point.** LLM (and heuristic) judges are noisy, so we measure the
judge against a small hand-labeled case set and report agreement. See the failure-mode
note in the generated report.

Run with:  uv run python -m rag_knowledge_assistant.faithfulness
"""

import json
import re
from pathlib import Path
from typing import NamedTuple, Protocol

from rag_knowledge_assistant.evaluation import load_gold
from rag_knowledge_assistant.generation import ExtractiveGenerator
from rag_knowledge_assistant.logging_config import get_logger
from rag_knowledge_assistant.retrieval import load_retriever

logger = get_logger(__name__)

CASES_PATH = Path("eval/faithfulness_cases.jsonl")
REPORT_PATH = Path("reports/faithfulness_eval.md")
THRESHOLD = 0.6  # grounding score at/above which an answer is judged faithful

# Small stop set so overlap reflects content words, not glue words.
STOP = {
    "the",
    "and",
    "are",
    "was",
    "were",
    "for",
    "with",
    "that",
    "this",
    "its",
    "use",
    "used",
    "means",
    "measures",
    "into",
    "two",
    "any",
    "every",
    "always",
}


class Case(NamedTuple):
    context: str
    answer: str
    label: str  # "faithful" | "unfaithful"


class Judge(Protocol):
    def score(self, answer: str, contexts: list[str]) -> float: ...


def _terms(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]{3,}", text.lower())}


class HeuristicJudge:
    """Lexical grounding: fraction of the answer's content terms present in context."""

    def score(self, answer: str, contexts: list[str]) -> float:
        context_terms: set[str] = set()
        for c in contexts:
            context_terms |= _terms(c)
        answer_terms = _terms(answer) - STOP
        if not answer_terms:
            return 1.0
        return len(answer_terms & context_terms) / len(answer_terms)


class ClaudeJudge:
    """LLM-as-judge faithfulness via the Anthropic API (claude-opus-4-8)."""

    def __init__(self, model: str = "claude-opus-4-8") -> None:
        self.model = model

    def score(self, answer: str, contexts: list[str]) -> float:
        import anthropic

        sources = "\n\n".join(contexts)
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=self.model,
            max_tokens=512,
            system=(
                "You are a strict faithfulness judge. Given SOURCES and an ANSWER, decide "
                "what fraction of the answer's claims are directly supported by the sources. "
                "Reply with only a number between 0 and 1."
            ),
            messages=[{"role": "user", "content": f"SOURCES:\n{sources}\n\nANSWER:\n{answer}"}],
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        match = re.search(r"[01](?:\.\d+)?", text)
        return float(match.group()) if match else 0.0


def load_cases(path: Path = CASES_PATH) -> list[Case]:
    return [
        Case(row["context"], row["answer"], row["label"])
        for line in path.read_text().splitlines()
        if line.strip()
        for row in [json.loads(line)]
    ]


def calibrate(cases: list[Case], judge: Judge, threshold: float = THRESHOLD) -> dict[str, float]:
    """Agreement between the judge (thresholded) and human labels."""
    correct = 0
    for case in cases:
        predicted = (
            "faithful" if judge.score(case.answer, [case.context]) >= threshold else "unfaithful"
        )
        correct += predicted == case.label
    return {"n": len(cases), "accuracy": correct / len(cases), "threshold": threshold}


def mean_faithfulness(
    judge: Judge, k: int = 4, generator: ExtractiveGenerator | None = None
) -> float:
    """End-to-end: faithfulness of generated answers over the gold questions."""
    generator = generator or ExtractiveGenerator()
    retriever = load_retriever()
    scores = []
    for item in load_gold():
        contexts = [h.chunk for h in retriever.retrieve(item.question, k=k)]
        answer = generator.generate(item.question, contexts)
        scores.append(judge.score(answer, [c.text for c in contexts]))
    return sum(scores) / len(scores) if scores else 0.0


def main() -> None:
    judge = HeuristicJudge()
    cal = calibrate(load_cases(), judge)
    faithfulness = mean_faithfulness(judge)

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(
        "# Faithfulness evaluation\n\n"
        "## Judge calibration (vs human labels)\n\n"
        f"On **{cal['n']}** hand-labeled cases, the heuristic grounding judge "
        f"(threshold {cal['threshold']}) agrees with human labels "
        f"**{cal['accuracy'] * 100:.0f}%** of the time.\n\n"
        "## End-to-end faithfulness\n\n"
        f"Mean faithfulness of the extractive generator's answers over the gold "
        f"questions: **{faithfulness:.3f}** (extractive answers quote the sources, so "
        "grounding is near-total by construction).\n\n"
        "## Failure modes (why calibrate)\n\n"
        "- The lexical judge measures word overlap, not meaning: an answer that reuses "
        "source vocabulary but reverses a claim can score as faithful (false positive).\n"
        "- It penalizes faithful paraphrases that use synonyms not in the source "
        "(false negative).\n"
        "- For semantic faithfulness, swap in `ClaudeJudge` — but validate *it* the same "
        "way, against human labels, before trusting its scores.\n"
    )
    logger.info("Wrote %s", REPORT_PATH)
    logger.info(
        "judge accuracy=%.2f (n=%d) | mean faithfulness=%.3f",
        cal["accuracy"],
        cal["n"],
        faithfulness,
    )


if __name__ == "__main__":
    main()
