# Eval Metrics — Interpretation

## What the metrics mean

**citation_correctness** — the fraction of cited URLs in a synthesized report whose text
contains at least one substring from the golden question's `expected_url_substrings` list
(e.g., `"python.org"`, `"redis.io"`).  A score of 1.0 means every cited source comes from
a trusted domain for that question.  A score of 0.0 means no cited URL is from an expected
domain.

**keyphrase_coverage** — the fraction of hand-specified `expected_keyphrases` that appear
(case-insensitively) anywhere in the synthesized report markdown.  A score of 1.0 means
all expected concepts were mentioned.

## Why the scores look like they do right now

Both metrics are at their **mock-pipeline floor**.  MockAgent generates citations with the
URL pattern `https://example.com/{agent}/{hash}/{i}` — these never match real domain
substrings, so `citation_correctness = 0.000` for every question.

`keyphrase_coverage` is non-zero (roughly 0.78 mean on the 20-question set) because MockAgent
echoes the question text back in its summary, and several expected keyphrases are substrings
of the questions themselves.  Phrases that would only appear in a real synthesized answer
(e.g., `"jitter"`, `"429"`, `"idempotency"`, `"real-time price"`) score 0.0 at the mock
floor.

This is intentional and honest: the harness measures real pipeline quality when real search
adapters and a real synthesizer are plugged in.

### Out-of-scope and refusal questions

The golden set includes two **out-of-scope** (`oos-*`) questions and one **refusal**
(`refusal-001`) question.  These are deliberately failure-likely for the mock pipeline:

- `oos-001` (live Bitcoin price) and `oos-002` (full RFC text) require real-time or bulk
  retrieval that MockAgent cannot provide.  Both are expected to score 0.0 citation
  correctness and low keyphrase coverage, establishing a useful lower bound.
- `refusal-001` ("best programming language") has no factual single-correct answer.  A well-
  behaved real pipeline should surface a refusal or "it depends" framing; keyphrase matches
  for `"depends"`, `"use case"`, and `"trade-offs"` reward that behaviour.

These categories make the keyphrase metric range non-trivially interpretable across the
full 20-question set.

## What an LLM judge would add

A real LLM judge would score each synthesized answer on semantic correctness — not just
keyphrase presence — enabling calibrated quality estimation for open-ended questions.  The
integration point is the `JudgeProtocol` placeholder in `evals/harness.py`: implement
`score(question: str, answer: str) -> float`, pass it to `run_harness`, and add the judge
score column to the report.

Cohen's kappa calibration requires two annotators (human + judge) to independently score a
shared set of examples, then compute agreement beyond chance.  A kappa > 0.6 is the
conventional threshold for "substantial agreement" before trusting the judge as a proxy.

## Known limitations

- **MockAgent is not a retrieval system.**  It returns deterministic fake content; the eval
  harness exercises the pipeline plumbing (fan-out, synthesis, scoring), not retrieval quality.
- **Citation domain matching is coarse.**  A real answer citing `https://python.org/path`
  scores the same as one citing any other page on the same domain.
- **Keyphrase matching ignores negation and context.**  "This is not jitter" would count
  as covering `"jitter"`.
