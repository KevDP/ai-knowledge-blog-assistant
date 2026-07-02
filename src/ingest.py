"""
ingest.py

Read all the markdown files ./knowledge/, start chunking process,
calculate embeddings with sentence-transformers (locally) show
the result via ./embeddings.json.

Diseño:
- Single responsibility: This file just generates the index.
- It is executed each time that knowledge/ gets changed.
- Frontmatter YAML it is discarged from embeddings but `topic` it is still saved
  `language` as debugging metadata and future filters.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TypedDict

from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "embeddings.json"


def _resolve_knowledge_dir() -> Path:
    """
    Directorio de la knowledge base. Prioridad:
      1. $EVA_KNOWLEDGE_DIR       (override explícito)
      2. ./knowledge             (data REAL de Kevin — gitignored, no se versiona)
      3. ./knowledge_demo        (persona SINTÉTICA — sí versionada)

    Efecto: local con data real usa knowledge/; un clon público (sin knowledge/)
    cae a knowledge_demo/ automáticamente, sin configuración. Así el repo es
    reproducible por cualquiera sin exponer datos reales.
    """
    env = os.environ.get("EVA_KNOWLEDGE_DIR")
    if env:
        return ROOT / env
    if (ROOT / "knowledge").is_dir():
        return ROOT / "knowledge"
    return ROOT / "knowledge_demo"


KNOWLEDGE_DIR = _resolve_knowledge_dir()

# Model: BAAI/bge-small-en-v1.5 — 384-dim, ~130MB, CPU-friendly.
# For querying BGE recommends instruction prefix;
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Frontmatter YAML
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class Chunk(TypedDict):
    text: str
    source: str
    topic: str
    embedding: list[float]


def _parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    """Devuelve (metadata, body). Sin frontmatter → ({}, raw)."""
    match = FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    body = raw[match.end():]
    return meta, body


def _chunk_paragraphs(body: str, min_chars: int = 80) -> list[str]:
    """
    Chunking ingenuo por párrafo (separador `\\n\\n`). Filtra párrafos muy
    cortos (líneas sueltas de cabecera) para no contaminar el índice con ruido.

    Nota honesta: esto es lo más simple que funciona para un knowledge base
    pequeño. Si más adelante el índice crece (>1k chunks) o las respuestas
    pierden contexto entre párrafos, se reemplaza por un chunker con overlap.
    """
    paragraphs = [p.strip() for p in body.split("\n\n")]
    return [p for p in paragraphs if len(p) >= min_chars]


def load_documents() -> list[tuple[str, dict[str, str], list[str]]]:
    """Lee knowledge/*.md y devuelve [(source, meta, chunks), ...]."""
    docs = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        chunks = _chunk_paragraphs(body)
        docs.append((path.name, meta, chunks))
    return docs


def _build_embedding_text(chunk: str, source: str, topic: str) -> str:
    """
    Contextual retrieval (Anthropic pattern): el texto que se EMBEBE incluye
    metadata explícita (source, topic) para que el embedding capture el
    concepto del documento, no solo el contenido literal del fragmento.

    El texto que se PASA a Claude sigue siendo solo `chunk` — no se inflan
    los tokens del prompt en runtime. Solo cambia la representación vectorial.

    Por qué importa: "## Languages\\n- Python — 4 years" como texto crudo no
    matchea bien con un query como "what are kevin's main skills?". Pero
    "Document: skills.md | Topic: skills | Content: ## Languages\\n- Python..."
    sí matchea — el embedding aprende la asociación skills↔contenido técnico.
    """
    return f"Document: {source} | Topic: {topic} | Content: {chunk}"


def build_index() -> list[Chunk]:
    print(f"[ingest] Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    docs = load_documents()
    if not docs:
        raise RuntimeError(f"No markdown found under {KNOWLEDGE_DIR}")

    flat_texts: list[str] = []          # Original content (Claude)
    flat_embed_texts: list[str] = []    # Embedded content
    flat_meta: list[tuple[str, str]] = []  # (source, topic)
    for source, meta, chunks in docs:
        topic = meta.get("topic", "")
        for chunk in chunks:
            flat_texts.append(chunk)
            flat_embed_texts.append(_build_embedding_text(chunk, source, topic))
            flat_meta.append((source, topic))

    print(f"[ingest] Embedding {len(flat_texts)} chunks from {len(docs)} files (contextual)...")
    embeddings = model.encode(
        flat_embed_texts,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    index: list[Chunk] = []
    for text, (source, topic), vec in zip(flat_texts, flat_meta, embeddings, strict=True):
        index.append({
            "text": text,           # text sent to Claude
            "source": source,
            "topic": topic,
            "embedding": vec.tolist(),
        })
    return index


def main() -> None:
    index = build_index()
    INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    size_kb = INDEX_PATH.stat().st_size / 1024
    print(f"[ingest] Wrote {len(index)} chunks → {INDEX_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
