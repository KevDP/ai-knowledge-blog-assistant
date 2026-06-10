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
  timeout          = 10
  memory_size      = 256

  environment {
    variables = {
      # parameters for initial phase BEDROCK_MODEL_ID, RELEVANCE_THRESHOLD, etc.
      PHASE = "1"
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
