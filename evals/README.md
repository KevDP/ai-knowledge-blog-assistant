# Evals

Regression tests for EVA's agent layer (Phase 2). They verify the agent's
**decisions** — routing, off-topic gating, retrieval recall, memory — not the
prose quality of the generated answer (that would need an LLM-as-judge, which is
non-deterministic and costly; deferred to a later phase).

## Privacy by design: synthetic persona

The golden set (`golden_set.json`) targets a **synthetic persona** ("Alex
Mercado") defined in [`../knowledge_demo/`](../knowledge_demo), not real data.

Why:

- The real knowledge base lives in a private repo and is **gitignored** here
  (`knowledge/`). Its content and, importantly, its chunking structure are kept
  out of the public repo.
- Running evals against a synthetic persona makes this repository **fully
  reproducible** by anyone — clone, ingest, test — without exposing real
  personal data.
- `ingest.py` resolves the knowledge dir automatically: `knowledge/` if present
  (real, local), otherwise `knowledge_demo/` (synthetic, committed). A public
  clone falls back to the demo with zero configuration.

## Running

```bash
# Build the vector index from the synthetic KB, then run the suite.
EVA_KNOWLEDGE_DIR=knowledge_demo python -m src.ingest
python -m evals.run_evals            # add --verbose to print answers
```

Note: `run_evals` invokes the LLM (answer + utility calls). With ~13 cases it
costs a few cents. Utility-call cost is reported at the end; `answer()` cost is
logged separately to stderr.

## What each expectation checks

| Field | Meaning |
|---|---|
| `route` | Expected router decision (`normal` / `meta`) |
| `off_topic` | Whether the relevance gate returned the canned response |
| `sources_include` | The correct source doc appears in the top-k (context window) |
| `min_score` | Top retrieval score is at least this value |

Cases with a `conversation` array replay the turns in order to build real
history, then assert on the **last** turn — that's how conversation memory is
exercised.

## Known limitations vs regressions

A case may carry a `known_limitation` note. If it fails, the runner reports it as
`⚠ known limitation`, not `✗ failure`, and the suite still exits 0. This mirrors
`xfail` in mature test frameworks: it distinguishes *"this regressed"* from
*"this is a documented gap with a planned fix."*

The current known limitation is the `projects` case: "projects" and "work
experience" overlap semantically (both are phrased as *"built X"*), so pure
embedding retrieval can't reliably separate them. The fix is a **reranking
layer**, tracked for Part 3 — not a routing/memory regression.
