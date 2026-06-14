# ─────────────────────────────────────────────────────────────────────────────
# DynamoDB - knowledge embeddings table (Phase 1c)
#
# Stores chunks + Titan embeddings. Lambda scans this table on cold start
# to load all embeddings into memory; cosine similarity runs in-process.
#
# Scale: ~30-50 items, ~5 KB each (1024-dim float32 vectors as base64).
# Well within DDB free tier (25 GB + 25 RCU/WCU forever free).
# No vector index needed - cosine in Lambda is sub-10ms at this scale.
# ─────────────────────────────────────────────────────────────────────────────

resource "aws_dynamodb_table" "knowledge" {
  name         = "${local.name_prefix}-knowledge"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "chunk_id"

  attribute {
    name = "chunk_id"
    type = "S"
  }

  point_in_time_recovery {
    enabled = false # rebuildable from knowledge/*.md, no recovery value
  }
}
