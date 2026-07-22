# ─────────────────────────────────────────────────────────────────────────────
# Lambda function
#
# This file helps to validate API Gateway - Lambda before feed the LLM
# Make a controlled enviornment first for testing
# ─────────────────────────────────────────────────────────────────────────────

# Lambda runtime uses zip file in /lambda 
# Terraform detect changes, if handler.py gets modified, it redeploys

data "archive_file" "eva_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../lambda"
  output_path = "${path.module}/build/eva.zip"
}

resource "aws_lambda_function" "eva" {
  function_name    = "${local.name_prefix}-eva"
  role             = aws_iam_role.lambda_exec.arn
  handler          = "handler.lambda_handler"
  runtime          = "python3.12"
  filename         = data.archive_file.eva_zip.output_path
  source_code_hash = data.archive_file.eva_zip.output_base64sha256
  timeout          = 30 # Bedrock invocation cold start
  memory_size      = 256

  environment {
    variables = {
      PHASE              = "1.4"
      BEDROCK_MODEL_ID   = var.bedrock_model_id
      TITAN_MODEL_ID     = var.titan_model_id
      KNOWLEDGE_TABLE    = aws_dynamodb_table.knowledge.name
      CACHE_TABLE        = aws_dynamodb_table.cache.name
      MAX_QUESTION_CHARS = "500"
      # Asymmetric margin (more buffer on off-topic side) 
      # off-topic margin (~0.10 avg) - on-topic margin ~0.30.
      RELEVANCE_THRESHOLD = "0.25"
      TOP_K               = "5"
      CACHE_TTL_SECONDS   = "86400" # 24h
    }
  }
}

# Log group should be already determined
# decision reasons:
# 1. Retention customizable
# 2. Supported Terraform-managed resource
# 3. "known bug", terraform destroy can unuse log group
# and creation can fail with "ResourceAlreadyExists".

resource "aws_cloudwatch_log_group" "eva_logs" {
  name              = "/aws/lambda/${aws_lambda_function.eva.function_name}"
  retention_in_days = 14
}
