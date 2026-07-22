#!/usr/bin/env python3
"""
Ingest knowledge files into DynamoDB embeddings table (Phase 1.5).

Differences vs Phase 1.3:
- Source: S3 private bucket (NOT the repo). KB never enters git history.
- Chunking: section-based (split by ## h2 headings) with h1+h2 prepend for
  hierarchical context. Replaces paragraph chunking that fragmented short
  sections in files like education.md.
- min_chars: 40 (was 80). Captures short-but-meaningful sections.

Run as a step in GitHub Actions after terraform apply. Uses:
- S3 GetObject to fetch markdown files from KB bucket
- Bedrock Titan Embeddings v2 to vectorize chunks (contextual prefix)
- DynamoDB PutItem to populate the knowledge table

Idempotent: same chunk_id replaces previous item. Re-running is harmless.
Cost per run: ~$0.001 (Titan tokens, scaling linearly with KB size).

In CI: inherits OIDC creds from workflow's role.
Locally: needs aws-vault or short-lived session creds with the same perms.
"""
import base64
import json
import math
import os
import re
import struct
import sys
import tempfile
from pathlib import Path

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
TITAN_MODEL_ID = os.environ.get("TITAN_MODEL_ID", "amazon.titan-embed-text-v2:0")
KNOWLEDGE_TABLE = os.environ.get("KNOWLEDGE_TABLE", "ai-kb-knowledge")
CACHE_TABLE = os.environ.get("CACHE_TABLE", "ai-kb-cache")
KNOWLEDGE_BUCKET = os.environ["KNOWLEDGE_BUCKET"]  # required, no default
KNOWLEDGE_PREFIX = os.environ.get("KNOWLEDGE_PREFIX", "knowledge/")
MIN_CHARS = int(os.environ.get("MIN_CHARS", "40"))

bedrock = boto3.client("bedrock-runtime", region_name=REGION)
ddb = boto3.client("dynamodb", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
H1_RE = re.compile(r"^# (.+?)$", re.MULTILINE)


def download_knowledge_from_s3() -> Path:
    """Pull all .md files from s3://{BUCKET}/{PREFIX} to a temp dir.
    Returns the local Path holding the downloaded files."""
    local_dir = Path(tempfile.mkdtemp(prefix="eva-kb-"))
    print(f"[ingest] downloading from s3://{KNOWLEDGE_BUCKET}/{KNOWLEDGE_PREFIX} -> {local_dir}")

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=KNOWLEDGE_BUCKET, Prefix=KNOWLEDGE_PREFIX)

    count = 0
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".md"):
                continue
            filename = key.split("/")[-1]
            local_path = local_dir / filename
            s3.download_file(KNOWLEDGE_BUCKET, key, str(local_path))
            count += 1

    if count == 0:
        print(f"No .md files found under s3://{KNOWLEDGE_BUCKET}/{KNOWLEDGE_PREFIX}", file=sys.stderr)
        sys.exit(1)

    print(f"[ingest] downloaded {count} files")
    return local_dir


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


def chunk_by_sections(body: str, min_chars: int = MIN_CHARS) -> list[str]:
    """Split markdown body by ## h2 headings. Each chunk preserves:
    - File's # h1 heading (prepended for hierarchical context)
    - The ## h2 heading (kept in the section)
    - All content until the next ## (or EOF)

    Filter sections shorter than min_chars to avoid noise from empty headers.
    For files without ## headings, returns one chunk per file (the whole body).

    Why prepend h1: a section "## Languages" in skills.md and a hypothetical
    "## Languages" in some other file would otherwise embed identically. The
    h1 prefix ("# Skills & tech stack" vs "# Other file") disambiguates.
    """
    h1_match = H1_RE.search(body)
    h1_prefix = f"# {h1_match.group(1).strip()}\n\n" if h1_match else ""

    # Split on lines starting with "## "
    sections = re.split(r"(?m)^(?=## )", body)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue

        if section.startswith("## "):
            chunk = f"{h1_prefix}{section}"
        else:
            if len(section) < min_chars * 2:
                continue
            # Don't re-prepend h1 if section already starts with the h1 line.
            if h1_prefix and section.startswith(h1_prefix.strip()):
                chunk = section
            else:
                chunk = f"{h1_prefix}{section}"

        if len(chunk) >= min_chars:
            chunks.append(chunk)

    return chunks


def embed(text: str) -> list[float]:
    """Titan v2 embedding, normalized for cosine via dot product."""
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
    """Pack float32 vector into base64 string (~5KB for 1024-dim Titan)."""
    return base64.b64encode(struct.pack(f"{len(vec)}f", *vec)).decode("ascii")


def build_embed_text(chunk: str, source: str, topic: str) -> str:
    """Contextual chunking (Anthropic pattern). The text EMBEDDED is enriched
    with metadata; Lambda stores the raw chunk separately to send to Claude."""
    return f"Document: {source} | Topic: {topic} | Content: {chunk}"


def purge_cache() -> int:
    """Delete every item in the response cache. Called after a successful
    re-ingest so users don't get stale cached answers built from old chunks.

    Without this, cache TTL (24h) keeps serving outdated info for up to a day
    after a KB update. Purging is the cheapest way to guarantee consistency
    between knowledge updates and what users see.

    Idempotent and safe to run with empty cache. BatchWriteItem caps at 25
    deletes per call, so we batch in chunks of 25.
    """
    print(f"\n[purge] scanning {CACHE_TABLE}...")
    items = []
    paginator = ddb.get_paginator("scan")
    for page in paginator.paginate(
        TableName=CACHE_TABLE,
        ProjectionExpression="question_hash",
    ):
        items.extend(page.get("Items", []))

    if not items:
        print("[purge] cache already empty")
        return 0

    deleted = 0
    for i in range(0, len(items), 25):
        batch = items[i:i + 25]
        ddb.batch_write_item(
            RequestItems={
                CACHE_TABLE: [
                    {"DeleteRequest": {"Key": {"question_hash": it["question_hash"]}}}
                    for it in batch
                ]
            }
        )
        deleted += len(batch)
    print(f"[purge] deleted {deleted} cached items")
    return deleted


def main() -> None:
    knowledge_dir = download_knowledge_from_s3()
    md_files = sorted(knowledge_dir.glob("*.md"))

    print(f"[ingest] table: {KNOWLEDGE_TABLE} | model: {TITAN_MODEL_ID} | region: {REGION}")
    print(f"[ingest] chunking: by-section, min_chars={MIN_CHARS}, prepend h1\n")

    total_chunks = 0
    for path in md_files:
        raw = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(raw)
        chunks = chunk_by_sections(body)
        topic = meta.get("topic", "")
        source = path.name

        print(f"  {source} -> {len(chunks)} chunks")

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

    # Atomic-ish KB refresh: purge cache AFTER chunks are in place. If purge
    # fails the new chunks are still indexed (user gets fresh answers on
    # cache miss). If purge succeeds but chunks failed, we don't get here.
    purge_cache()


if __name__ == "__main__":
    main()
