"""
ingest.py

Read all the markdown files ./knowledge/, start chunking process,
calculate embeddings with sentence-transformers (locally) show
the result via ./embeddings.json.

Design:
- Single responsibility: this file just generates the index.
- Runs each time knowledge/ changes.
- Frontmatter YAML is stripped from embeddings but `topic` is kept as
  metadata for debugging and future filters.
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
    Knowledge-base directory. Resolution order:
      1. $EVA_KNOWLEDGE_DIR   (explicit override)
      2. ./knowledge          (REAL data - gitignored, not versioned)
      3. ./knowledge_demo     (SYNTHETIC persona - versioned in the public repo)

    Effect: local with real data uses knowledge/; a public clone (no knowledge/)
    falls back to knowledge_demo/ automatically, no configuration required.
    That way the repo is reproducible by anyone without exposing real data.
    """
    env = os.environ.get("EVA_KNOWLEDGE_DIR")
    if env:
        return ROOT / env
    if (ROOT / "knowledge").is_dir():
        return ROOT / "knowledge"
    return ROOT / "knowledge_demo"


KNOWLEDGE_DIR = _resolve_knowledge_dir()

# Model: BAAI/bge-small-en-v1.5 (384-dim, ~130MB, CPU-friendly).
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
    """Returns (metadata, body). Without frontmatter → ({}, raw)."""
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
    Naive per-paragraph chunking (separator `\\n\\n`). Filters out very short
    paragraphs (stray heading lines) to keep noise out of the index.

    Honest note: this is the simplest thing that works for a small knowledge
    base. If the index grows (>1k chunks) or answers lose context across
    paragraphs, swap this for an overlap-aware chunker.
    """
    paragraphs = [p.strip() for p in body.split("\n\n")]
    return [p for p in paragraphs if len(p) >= min_chars]


def load_documents() -> list[tuple[str, dict[str, str], list[str]]]:
    """Reads knowledge/*.md and returns [(source, meta, chunks), ...]."""
    docs = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        chunks = _chunk_paragraphs(body)
        docs.append((path.name, meta, chunks))
    return docs


def _build_embedding_text(chunk: str, source: str, topic: str) -> str:
    """
    Contextual retrieval (Anthropic pattern): the text that is EMBEDDED includes
    explicit metadata (source, topic) so the embedding captures the document's
    concept, not just the raw fragment content.

    The text that is PASSED to Claude is still just `chunk`, no prompt-token
    bloat at runtime. Only the vector representation changes.

    Why it matters: "## Languages\\n- Python - 4 years" as raw text does not
    match well against a query like "what are kevin's main skills?". But
    "Document: skills.md | Topic: skills | Content: ## Languages\\n- Python..."
    does - the embedding learns the association skills - technical content.
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
