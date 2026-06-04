"""
retrieve.py — Recupera los top-k chunks más similares a una pregunta.

Diseño:
- Funciones puras, sin side effects salvo carga lazy del índice y del modelo.
- El índice (embeddings.json) y el modelo se cargan una sola vez por proceso,
  cacheados en variables module-level. Importante para el CLI interactivo: el
  primer query paga el costo (~2s carga modelo + JSON), los siguientes son ms.
- Similitud coseno = dot product entre vectores normalizados. ingest.py ya los
  normaliza, así que aquí asumimos `normalize_embeddings=True` y solo hacemos
  `query @ matrix.T`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "embeddings.json"

# DEBE coincidir con ingest.py. Si los modelos no son el mismo,
# los embeddings del índice y los del query están en espacios distintos
# y la similitud no significa nada. Mantener sincronizado.
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE recomienda prepender una instrucción al QUERY (no al document) para
# tareas de retrieval. Mejora ~3-5% el recall. Sin esto, BGE sigue
# funcionando pero pierdes parte de la ventaja sobre MiniLM.
# Ver: https://huggingface.co/BAAI/bge-small-en-v1.5#usage-for-retrieval
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class RetrievedChunk(TypedDict):
    text: str
    source: str
    topic: str
    score: float


# ─────────────────────────────────────────────────────────────────────────────
# Lazy singletons. No los expongas públicamente; usa retrieve() / get_model().
# ─────────────────────────────────────────────────────────────────────────────
_model: SentenceTransformer | None = None
_index_texts: list[dict] | None = None
_index_matrix: np.ndarray | None = None


def _load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _load_index() -> tuple[list[dict], np.ndarray]:
    """Carga embeddings.json en memoria una sola vez."""
    global _index_texts, _index_matrix
    if _index_texts is None or _index_matrix is None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"Índice no encontrado en {INDEX_PATH}. "
                "Corre primero: python -m src.ingest"
            )
        raw = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        _index_texts = [{"text": c["text"], "source": c["source"], "topic": c["topic"]} for c in raw]
        _index_matrix = np.array([c["embedding"] for c in raw], dtype=np.float32)
    return _index_texts, _index_matrix


def retrieve(query: str, k: int = 3) -> list[RetrievedChunk]:
    """
    Devuelve los k chunks con mayor similitud coseno respecto a `query`.
    Resultado ordenado de mayor a menor score.
    """
    if not query.strip():
        return []

    model = _load_model()
    texts, matrix = _load_index()

    # Embed la pregunta CON el prefijo de instrucción BGE. Normalizamos
    # para que el dot product sea coseno.
    q_vec = model.encode(
        [BGE_QUERY_INSTRUCTION + query],
        normalize_embeddings=True,
    )[0]
    scores = matrix @ q_vec  # shape: (n_chunks,)

    # Top-k por score descendente. argpartition es O(n) vs argsort O(n log n);
    # a esta escala (cientos de chunks) la diferencia es irrelevante, pero
    # acostumbra al hábito correcto.
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
    # Smoke test rápido si lo corres directo.
    import sys
    q = " ".join(sys.argv[1:]) or "what is kevin's experience?"
    print(f"Query: {q}\n")
    for hit in retrieve(q, k=3):
        print(f"[{hit['score']:.3f}] {hit['source']} ({hit['topic']})")
        print(f"  {hit['text'][:200]}...\n")
