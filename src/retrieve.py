"""
retrieve.py

- top-k chunks most similar to question.

Design:
- Index (embeddings.json) and model gets loaded once per process.
- Similar to coseno = dot product on normalized vectors. 
- Regarding to Ingest.py we assume `normalize_embeddings=True`,
then, we concluded with `query @ matrix.T`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "embeddings.json"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

class RetrievedChunk(TypedDict):
    text: str
    source: str
    topic: str
    score: float

_model: SentenceTransformer | None = None
_index_texts: list[dict] | None = None
_index_matrix: np.ndarray | None = None


def _load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _load_index() -> tuple[list[dict], np.ndarray]:
    """Loads embeddings.json into memory once per process."""
    global _index_texts, _index_matrix
    if _index_texts is None or _index_matrix is None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"Index not found at {INDEX_PATH}. "
                "Run first: python -m src.ingest"
            )
        raw = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        _index_texts = [{"text": c["text"], "source": c["source"], "topic": c["topic"]} for c in raw]
        _index_matrix = np.array([c["embedding"] for c in raw], dtype=np.float32)
    return _index_texts, _index_matrix


def retrieve(query: str, k: int = 3) -> list[RetrievedChunk]:
    """
    Returns the k chunks with the highest cosine similarity to `query`.
    Ordered from highest to lowest score.
    """
    if not query.strip():
        return []

    model = _load_model()
    texts, matrix = _load_index()

    # normalizing, dot product = cos.
    q_vec = model.encode(
        [BGE_QUERY_INSTRUCTION + query],
        normalize_embeddings=True,
    )[0]
    scores = matrix @ q_vec  # shape: (n_chunks,)

    # Top-k per descendant score. 
    # argpartition is O(n) vs argsort O(n log n);
    # just as a best practice
    k = min(k, len(scores))
    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    return [
        {
            "text": texts[i]["text"],
            "source": texts[i]["source"],
            "topic": texts[i]["topic"],
            "score": float(scores[i]),
        }
        for i in top_idx
    ]


if __name__ == "__main__":
    # test
    import sys
    q = " ".join(sys.argv[1:]) or "what is kevin's experience?"
    print(f"Query: {q}\n")
    for hit in retrieve(q, k=3):
        print(f"[{hit['score']:.3f}] {hit['source']} ({hit['topic']})")
        print(f"  {hit['text'][:200]}...\n")
