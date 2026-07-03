# Faithfulness evaluation

## Judge calibration (vs human labels)

On **10** hand-labeled cases, the heuristic grounding judge (threshold 0.6) agrees with human labels **100%** of the time.

## End-to-end faithfulness

Mean faithfulness of the extractive generator's answers over the gold questions: **0.998** (extractive answers quote the sources, so grounding is near-total by construction).

## Failure modes (why calibrate)

- The lexical judge measures word overlap, not meaning: an answer that reuses source vocabulary but reverses a claim can score as faithful (false positive).
- It penalizes faithful paraphrases that use synonyms not in the source (false negative).
- For semantic faithfulness, swap in `ClaudeJudge` — but validate *it* the same way, against human labels, before trusting its scores.
