# ─────────────────────────────────────────────────────────────────────────────
# Weekly usage digest
#
# Runs in AWS on a schedule. It keeps working whether or not anyone remembers to run it.
#
# Cost: one Lambda invocation per week (free tier), one GetMetricData call
# (free), one SNS email (free tier). Effectively zero.
# ─────────────────────────────────────────────────────────────────────────────

data "archive_file" "digest_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../lambda_digest"
  output_path = "${path.module}/build/digest.zip"
}

# ─── Execution role ──────────────────────────────────────────────────────────
# Read-only on metrics, publish-only on one topic. 
# A compromised digest can read numbers and send you an email.
# It cannot touch EVA, the knowledge base or the cache.

resource "aws_iam_role" "digest_exec" {
  name               = "${local.name_prefix}-digest-exec"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "digest_logs" {
  role       = aws_iam_role.digest_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "digest" {
  statement {
    # GetMetricData takes no resource-level permissions, so "*" is the only
    # valid form here. It is read-only on metric values.
    actions   = ["cloudwatch:GetMetricData"]
    resources = ["*"]
  }

  statement {
    actions   = ["sns:Publish"]
    resources = [aws_sns_topic.alerts.arn]
  }

  # Logs Insights over the CloudFront Function log, to report who is actually hitting the site. 
  # StartQuery takes a resource-level scope, so it is pinned to edge-function log groups in this account and cannot read the EVA logs or anything else. 

  statement {
    actions   = ["logs:StartQuery"]
    resources = ["arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/cloudfront/function/*:*"]
  }

  statement {
    actions   = ["logs:GetQueryResults", "logs:StopQuery"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "digest" {
  name   = "${local.name_prefix}-digest-read-and-publish"
  role   = aws_iam_role.digest_exec.id
  policy = data.aws_iam_policy_document.digest.json
}

# ─── Function ────────────────────────────────────────────────────────────────

resource "aws_lambda_function" "digest" {
  function_name    = "${local.name_prefix}-digest"
  role             = aws_iam_role.digest_exec.arn
  handler          = "handler.lambda_handler"
  runtime          = "python3.12"
  filename         = data.archive_file.digest_zip.output_path
  source_code_hash = data.archive_file.digest_zip.output_base64sha256

  timeout     = 120
  memory_size = 128

  environment {
    variables = {
      TOPIC_ARN                  = aws_sns_topic.alerts.arn
      EVA_FUNCTION_NAME          = aws_lambda_function.eva.function_name
      EVA_API_ID                 = aws_apigatewayv2_api.eva.id
      BEDROCK_MODEL_ID           = var.bedrock_model_id
      TITAN_MODEL_ID             = var.titan_model_id
      CLOUDFRONT_DISTRIBUTION_ID = var.cloudfront_distribution_id
      EDGE_LOG_GROUP             = var.cloudfront_function_log_group
    }
  }
}

resource "aws_cloudwatch_log_group" "digest_logs" {
  name              = "/aws/lambda/${aws_lambda_function.digest.function_name}"
  retention_in_days = 90
}

# ─── Schedule ────────────────────────────────────────────────────────────────
# EventBridge Scheduler over a CloudWatch Events rule.

resource "aws_iam_role" "scheduler" {
  name = "${local.name_prefix}-digest-scheduler"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRole"
      Principal = { Service = "scheduler.amazonaws.com" }
      Condition = {
        StringEquals = { "aws:SourceAccount" = data.aws_caller_identity.current.account_id }
      }
    }]
  })
}

data "aws_caller_identity" "current" {}

resource "aws_iam_role_policy" "scheduler_invoke" {
  name = "${local.name_prefix}-digest-scheduler-invoke"
  role = aws_iam_role.scheduler.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "lambda:InvokeFunction"
      Resource = aws_lambda_function.digest.arn
    }]
  })
}

resource "aws_scheduler_schedule" "digest_weekly" {
  name                         = "${local.name_prefix}-digest-weekly"
  schedule_expression          = "cron(0 9 ? * MON *)"
  schedule_expression_timezone = "America/Mexico_City"

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = aws_lambda_function.digest.arn
    role_arn = aws_iam_role.scheduler.arn
  }
}
