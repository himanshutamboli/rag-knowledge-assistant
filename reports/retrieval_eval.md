# Retrieval evaluation

Gold set: **20** questions, document-level relevance. Retriever: TF-IDF + cosine.

| metric | value |
| --- | --- |
| recall@1 | 0.875 |
| recall@3 | 1.000 |
| recall@5 | 1.000 |
| recall@10 | 1.000 |
| mrr | 0.942 |

`recall@k` = share of relevant docs in the top-k; `MRR` = mean reciprocal rank of the first relevant doc.
