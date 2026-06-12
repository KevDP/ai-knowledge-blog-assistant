# ─────────────────────────────────────────────────────────────────────────────
# Lambda execution IAM role (runtime)
#
# This is different from Github Actions role created (CD)
# Principle of least privilege, only permissions role will be impacted.
# ─────────────────────────────────────────────────────────────────────────────

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda_exec" {
  name               = "${local.name_prefix}-lambda-exec"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

# AWSLambdaBasicExecutionRole will be allowed to write in CloudWatch Logs (CreateLogStream + PutLogEvents)

resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Bedrock InvokeModel permission
# Scoped to foundation-model and inference-profile ARNs
# Both included because newer Claude models on Bedrock may require inference profiles
# If compromised, attacker can only invoke Bedrock models, not modify them or list

data "aws_iam_policy_document" "lambda_bedrock" {
  statement {
    actions = ["bedrock:InvokeModel"]
    resources = [
      "arn:aws:bedrock:*::foundation-model/*",
      "arn:aws:bedrock:*:*:inference-profile/*",
    ]
  }
}

resource "aws_iam_role_policy" "lambda_bedrock" {
  name   = "${local.name_prefix}-lambda-bedrock-invoke"
  role   = aws_iam_role.lambda_exec.id
  policy = data.aws_iam_policy_document.lambda_bedrock.json
}
