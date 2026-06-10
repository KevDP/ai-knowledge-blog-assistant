# ─────────────────────────────────────────────────────────────────────────────
# API Gateway HTTP API (no REST).
#
# HTTP API vs REST:
# - $1/M requests vs $3.50/M (HTTP is 3.5× cheaper)
# - HTTP lower latency
#
# In this case, HTTP API is better on all spots
# REST API is only justificable if usage plans with consumer tracking is needed
# ─────────────────────────────────────────────────────────────────────────────

resource "aws_apigatewayv2_api" "eva" {
  name          = "${local.name_prefix}-eva-api"
  protocol_type = "HTTP"

  # CORS temporary opened, it allows to local test with curl
  # When necessary, allow_origins parameter will lists autorized domains
  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["POST", "OPTIONS"]
    allow_headers = ["content-type"]
    max_age       = 300
  }
}

# API GW + Lambda (v.2)
resource "aws_apigatewayv2_integration" "eva" {
  api_id                 = aws_apigatewayv2_api.eva.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.eva.invoke_arn
  integration_method     = "POST"
  payload_format_version = "2.0"
}

# Route: POST /eva - integration in one endpoint
resource "aws_apigatewayv2_route" "eva" {
  api_id    = aws_apigatewayv2_api.eva.id
  route_key = "POST /eva"
  target    = "integrations/${aws_apigatewayv2_integration.eva.id}"
}

# Stage $default: public API without path prefix
# defense stack to Throttling situations:
# 10 req/s rate as default, burst if is up to 20.
# result: user is a bot, rate limit stop requests

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.eva.id
  name        = "$default"
  auto_deploy = true

  default_route_settings {
    throttling_burst_limit = 20
    throttling_rate_limit  = 10
  }
}

# API GW invokes Lambda
# Without this step, requests get AccessDenied.
# source_arn with /*/* (Principle of least privilege)
resource "aws_lambda_permission" "api_gw_invoke" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.eva.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.eva.execution_arn}/*/*"
}
