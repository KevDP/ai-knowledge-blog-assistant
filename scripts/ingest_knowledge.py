#!/usr/bin/env python3
"""
Ingest knowledge/*.md into DynamoDB embeddings table (Phase 1c).

Run as a step in GitHub Actions after terraform apply. Uses:
- Bedrock Titan Embeddings v2 to vectorize chunks (contextual prefix)
- DynamoDB PutItem to populate the table

Idempotent: same chunk_id replaces previous item, so re-running on unchanged
knowledge is harmless. Cost per run: ~$0.001 (Titan tokens).

Vector storage: float32 vectors packed as base64 strings (smaller than DDB
Number lists, faster to decode in Lambda).

Locally: needs AWS credentials with bedrock:InvokeModel + dynamodb:PutItem.
In CI: inherits OIDC creds from workflow's role.
"""
import base64
import json
import math
import os
import re
import struct
import sys
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "knowledge"

REGION = os.environ.get("AWS_REGION", "us-east-1")
TITAN_MODEL_ID = os.environ.get("TITAN_MODEL_ID", "amazon.titan-embed-text-v2:0")
KNOWLEDGE_TABLE = os.environ.get("KNOWLEDGE_TABLE", "ai-kb-knowledge")

bedrock = boto3.client("bedrock-runtime", region_name=REGION)
ddb = boto3.client("dynamodb", region_name=REGION)

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Return (metadata_dict, body_without_frontmatter)."""
    match = FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    body = raw[match.end():]
    return meta, body


def chunk_paragraphs(body: str, min_chars: int = 80) -> list[str]:
    """Split body by blank-line paragraphs; drop ones too short to be signal."""
    paragraphs = [p.strip() for p in body.split("\n\n")]
    return [p for p in paragraphs if len(p) >= min_chars]


def embed(text: str) -> list[float]:
    """Get Titan v2 embedding, normalized for cosine via dot product."""
    response = bedrock.invoke_model(
        modelId=TITAN_MODEL_ID,
        body=json.dumps({"inputText": text}),
        contentType="application/json",
    )
    payload = json.loads(response["body"].read())
    vec = payload["embedding"]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


def encode_vector(vec: list[float]) -> str:
    """Pack float32 vector into base64 string. ~5KB for 1024-dim Titan output."""
    return base64.b64encode(struct.pack(f"{len(vec)}f", *vec)).decode("ascii")


def build_embed_text(chunk: str, source: str, topic: str) -> str:
    """Contextual chunking (Anthropic pattern, same as Phase 0).
    Enriches the embedded representation with metadata; Lambda stores the
    raw chunk separately to send to Claude unchanged."""
    return f"Document: {source} | Topic: {topic} | Content: {chunk}"


def main() -> None:
    md_files = sorted(KNOWLEDGE_DIR.glob("*.md"))
    if not md_files:
        print(f"No markdown found under {KNOWLEDGE_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"[ingest] table: {KNOWLEDGE_TABLE} | model: {TITAN_MODEL_ID} | region: {REGION}")
    print(f"[ingest] files: {len(md_files)}\n")

    total_chunks = 0
    for path in md_files:
        raw = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(raw)
        chunks = chunk_paragraphs(body)
        topic = meta.get("topic", "")
        source = path.name

        print(f"  {source} ({len(chunks)} chunks)")

        for i, chunk in enumerate(chunks):
            chunk_id = f"{source}#{i}"
            embed_text = build_embed_text(chunk, source, topic)
            vec = embed(embed_text)
            encoded = encode_vector(vec)

            ddb.put_item(
                TableName=KNOWLEDGE_TABLE,
                Item={
                    "chunk_id": {"S": chunk_id},
                    "source": {"S": source},
                    "topic": {"S": topic},
                    "text": {"S": chunk},
                    "embedding": {"S": encoded},
                },
            )
            total_chunks += 1

    print(f"\n[ingest] Wrote {total_chunks} chunks to {KNOWLEDGE_TABLE}")


if __name__ == "__main__":
    main()
