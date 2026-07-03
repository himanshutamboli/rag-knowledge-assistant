# rag-knowledge-assistant

[![CI](https://github.com/himanshutamboli/rag-knowledge-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/himanshutamboli/rag-knowledge-assistant/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Ruff](https://img.shields.io/badge/lint-ruff-orange.svg)](https://github.com/astral-sh/ruff)

> A retrieval-augmented QA assistant with **grounded, cited answers** — and, the part that separates it from every toy chatbot, a real **retrieval + faithfulness evaluation harness**. Built over a corpus of 21 ML/AI Wikipedia articles.

## Why this exists

Anyone can wire an LLM to a vector store. The hard part is *knowing whether retrieval actually works* and whether answers are *faithful* to the sources. This repo treats those as first-class, measured concerns (Days 18–19), not an afterthought.

## Ingestion & chunking (Day 15)

Chunking is the biggest lever on retrieval quality, so it's explicit and tested:

- **Corpus:** 21 Wikipedia articles (ML/AI/data topics), fetched as plain-text extracts with source attribution in [`data/corpus_sources.json`](data/corpus_sources.json). ~1M characters.
- **Strategy:** paragraph-aware packing — whole paragraphs are greedily packed to a ~800-char target (≈200 tokens) to stay semantically coherent, with a 150-char **overlap** carried across boundaries so context isn't severed. Oversized paragraphs are hard-split with the same overlap.
- **Output:** `21 docs → 1,296 chunks` (avg 624 chars), each with a stable `chunk_id`, `doc_id`, title, and source URL for citations.

```bash
uv sync --dev
uv run python -m rag_knowledge_assistant.pipeline   # corpus -> data/chunks.jsonl
uv run pytest                                         # chunking + retrieval tests
```

## Retrieval (Day 16)

- **Pluggable `Embedder` protocol.** Default is a **TF-IDF (lexical)** embedder — deterministic, offline, no model downloads, so CI stays fast and torch-free. A dense embedder (sentence-transformers) implements the same protocol and drops in.
- **Why default to lexical?** The eval harness (Day 18) is what should *justify* a heavier embedder — measure first, upgrade second. TF-IDF is a strong, honest baseline.
- **Vector store:** brute-force cosine over the ~1,296 chunks (exact, zero index maintenance). Approximate indexes (FAISS/HNSW) only earn their keep at much larger scale.

```bash
uv run python -m rag_knowledge_assistant.retrieval "how do transformers use attention?"
# -> top chunks from the Attention and Transformer articles, with scores
```

## Grounded generation + citations (Day 17)

Answers are generated **only** from retrieved sources, with inline `[n]` citations back to the source article + URL. If retrieval is too weak, the assistant **refuses** rather than hallucinate.

- **Pluggable `Generator`.** Default `ExtractiveGenerator` is deterministic and offline (no API key) — it selects the most query-relevant sentence per source and cites it; used in tests/CI. `ClaudeGenerator` (`claude-opus-4-8`) is a drop-in for fluent grounded answers when an API key is present.
- **Refusal.** A zero-overlap query (`"banana bread recipe"`) refuses cleanly. Refusal here is a single lexical-score threshold — a deliberately simple mechanism; **calibrating it properly is exactly what the retrieval + faithfulness eval harness (Days 18–19) is for.**

```bash
uv run python -m rag_knowledge_assistant.generation "what is overfitting and how do you prevent it?"
# -> grounded answer with [1]..[4] citations to Overfitting, Regularization, ... + URLs
```

## Interface (Day 20)

A **FastAPI** app with a streaming chat UI. `/stream` emits the answer over **Server-Sent Events** (word-by-word) followed by a `citations` event; the minimal single-page UI at `/` renders the streaming answer and clickable, scored source links. `/ask` returns the same as JSON.

```bash
uv run uvicorn rag_knowledge_assistant.api:app --reload   # open http://localhost:8000
```

## Retrieval evaluation (Day 18) — the differentiator

A hand-built **gold set of 20 question→document pairs** ([`eval/gold_retrieval.jsonl`](eval/gold_retrieval.jsonl)) drives `recall@k` and `MRR`. This is the part most RAG repos skip — without it you can't know whether retrieval works or whether an embedder change helped.

| metric | value |
| --- | --- |
| recall@1 | 0.875 |
| recall@3 | 1.000 |
| recall@5 | 1.000 |
| recall@10 | 1.000 |
| **MRR** | **0.942** |

**Read:** the TF-IDF baseline is strong on this corpus (recall@3 = 100%, MRR = 0.94) because the questions are keyword-aligned with the Wikipedia titles. That's the honest finding — and it's now the *measured* bar a dense embedder would have to beat to justify the extra dependency. Relevance is judged at the document level; regenerate with `uv run python -m rag_knowledge_assistant.evaluation`.

## Faithfulness evaluation (Day 19)

Is the answer actually supported by the retrieved sources? A pluggable **`Judge`** scores (answer, context) grounding. Default `HeuristicJudge` (lexical overlap, offline) is the CI baseline; `ClaudeJudge` (`claude-opus-4-8`) is a drop-in LLM-as-judge.

The senior move is **calibrating the judge against human labels** rather than trusting it blind:

- On **10 hand-labeled cases** ([`eval/faithfulness_cases.jsonl`](eval/faithfulness_cases.jsonl)), the heuristic judge agrees with human labels **100%** at threshold 0.6 (faithful cases score ~1.0; fabricated answers 0.08–0.22).
- End-to-end, the extractive generator's answers score **0.998** mean faithfulness (they quote the sources, so grounding is near-total by construction).
- **Documented failure modes:** the lexical judge measures word overlap, not meaning — it can pass a claim that reuses source vocabulary but reverses it, and penalize synonym paraphrases. For semantic faithfulness use `ClaudeJudge`, but *validate it against human labels the same way*. Full note: [`reports/faithfulness_eval.md`](reports/faithfulness_eval.md).

## Project structure

```
rag-knowledge-assistant/
├── data/corpus/                  # 21 committed Wikipedia articles (.txt)
├── data/corpus_sources.json      # title + URL + license attribution
├── src/rag_knowledge_assistant/
│   ├── ingestion.py              # load corpus documents
│   ├── chunking.py               # paragraph-aware chunking + overlap
│   ├── pipeline.py               # corpus -> chunks.jsonl
│   ├── embeddings.py             # Embedder protocol + TF-IDF embedder
│   ├── retrieval.py              # brute-force cosine retriever
│   ├── generation.py             # grounded answers + citations + refusal
│   ├── evaluation.py             # recall@k + MRR over the gold set
│   ├── faithfulness.py           # LLM-as-judge / heuristic faithfulness + calibration
│   └── api.py                    # FastAPI: SSE streaming endpoint + chat UI
├── eval/
│   ├── gold_retrieval.jsonl      # 20 gold question -> document pairs
│   └── faithfulness_cases.jsonl  # 10 hand-labeled faithful/unfaithful cases
└── tests/                        # chunking, retrieval, generation, retrieval eval, faithfulness
```

## Roadmap

| Day | Deliverable |
|---|---|
| 15 ✅ | Ingestion + chunking pipeline |
| 16 ✅ | Embeddings (pluggable) + cosine retrieval |
| 17 ✅ | Grounded generation + inline citations + refusal |
| 18 ✅ | **Retrieval eval** — recall@k, MRR, 20-question gold set |
| 19 ✅ | **Faithfulness eval** — judge + calibration vs human labels |
| 20 ✅ | Streaming (SSE) API + chat UI with citations |
| 21 | Ship v1.0 |

## Data source

Wikipedia article text, licensed **CC BY-SA 4.0**; per-article titles and URLs in `data/corpus_sources.json`.

## License

MIT (code). Corpus content: CC BY-SA 4.0 (Wikipedia).
