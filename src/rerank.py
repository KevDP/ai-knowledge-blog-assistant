"""
rerank.py - Retrieval (Phase 2)

The bi-encoder (BGE/Titan) is good for recall: retrieves plausible candidates
cheaply and at scale. But on KBs with overlapping vocabulary (where same ideas
appears in both projects and work experience) it loses precision.

The cross-encoder is the standard answer. Instead of encoding query and chunk
separately and comparing vectors (bi-encoder), it encodes the pair
(query, chunk) jointly with cross-attention. That captures fine-grained
semantic interactions a dot product over independent embeddings cannot.

Two-stage strategy:
    1. Bi-encoder retrieves top-N (N=10)  - cheap recall over the whole KB
    2. Cross-encoder reranks to top-K (K=3)  - bounded precision over N candidates

Cost: ~150ms extra CPU over 10 candidates with MiniLM-L-6-v2

Critical note on scales: cross-encoder scores are unnormalized logits(typically -10 to +10).
That is why OFF_TOPIC_THRESHOLD and WEAK_BAND_CEIL in agent.py continue to gate the bi-encoder score.
This gate is basically the signal for "did we find anything remotely relevant?" (recall). 
The rerank_score is used only to reorder the already-retrieved candidates(precision).
"""
from __future__ import annotations

import os

from sentence_transformers import CrossEncoder

# For more precision (and ~2x latency) upgrade to L-12.
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Toggle for A/B in the eval suite (measure delta with vs. without rerank).
# EVA_RERANK=0 - rerank pass-through.
RERANK_ENABLED = os.environ.get("EVA_RERANK", "1") == "1"

_model: CrossEncoder | None = None


def _load_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    """
    Reorder `candidates` by real relevance to `query` using a cross-encoder.
    Returns the top_k best according to the new score.

    Each returned chunk carries:
      - "rerank_score": the cross-encoder score (precision signal)
      - "score":        preserved as the bi-encoder cosine (recall signal /
                        display value that downstream consumers already
                        assume is in BGE scale).

    If EVA_RERANK=0 (toggle), returns the first top_k without reordering,
    useful for A/B benchmarks: run the eval suite twice and compare.
    """
    if not candidates:
        return []

    if not RERANK_ENABLED:
        # if pass-through: only truncate to top_k, respecting bi-encoder order.
        return candidates[:top_k]

    model = _load_model()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    reranked = []
    for c, s in zip(candidates, scores):
        item = dict(c)
        item["rerank_score"] = float(s)
        # the score is preserved as bi-encoder cosine, NOT overwritten
        # the LIST ORDER reflects rerank_score (precision)
        # the per-chunk `score` remains its cosine similarity to the query (recall)
        reranked.append(item)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]
