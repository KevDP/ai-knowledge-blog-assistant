# ───────────────────────────────────────────────────────────────────────────────────────────────────────
# DynamoDB - knowledge embeddings table
#
# Stores chunks + Titan embeddings. Lambda scans this table on cold start to load all embeddings into memory.
#
# Well within DDB free tier. No vector index needed at this scale, cosine in Lambda is ~10ms.
# ───────────────────────────────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# DynamoDB - response cache
#
# Stores question_hash -> answer with TTL 24h. Avoids re-invoking Claude and
# Titan for repeated questions. Cost saving expected for ~40% on real traffic.
#
# variables:
#   question_hash (PK, string)  - sha256 of stripped question
#   question (string)           - original text, for CloudWatch debugging
#   answer (string)             - Claude's response
#   sources_json (string)       - JSON-encoded list of source files
#   ttl (number)                - epoch seconds, DDB auto-purges expired items
# ─────────────────────────────────────────────────────────────────────────────

resource "aws_dynamodb_table" "cache" {
  name         = "${local.name_prefix}-cache"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "question_hash"

  attribute {
    name = "question_hash"
    type = "S"
  }

  # DDB auto-deletes items where the value in this attribute is in the past.
  # Handler double-checks ttl on read to avoid serving stale items.
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  point_in_time_recovery {
    enabled = false # no recovery value
  }
}
