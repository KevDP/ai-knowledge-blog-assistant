# Outputs útiles después de terraform apply. Visibles en GitHub Actions logs
# y en `terraform output`. NO incluir secretos aquí — los outputs no son sensitive.

output "api_endpoint" {
  description = "URL pública del endpoint POST /eva. Test: curl -X POST <url> -d '{\"question\":\"hi\"}'"
  value       = "${aws_apigatewayv2_api.eva.api_endpoint}/eva"
}

output "lambda_function_name" {
  description = "Nombre del Lambda. Útil para: aws logs tail /aws/lambda/<nombre> --follow"
  value       = aws_lambda_function.eva.function_name
}

output "cloudwatch_log_group" {
  description = "Log group del Lambda en CloudWatch"
  value       = aws_cloudwatch_log_group.eva_logs.name
}

output "knowledge_table_name" {
  description = "DynamoDB table holding the knowledge embeddings (Phase 1.3)"
  value       = aws_dynamodb_table.knowledge.name
}

output "cache_table_name" {
  description = "DynamoDB table for response cache (Phase 1.4)"
  value       = aws_dynamodb_table.cache.name
}

output "alerts_topic_arn" {
  description = "SNS topic for alarms and the weekly digest. The email subscription starts as pending_confirmation: check your inbox and click the link, or nothing is delivered."
  value       = aws_sns_topic.alerts.arn
}

output "digest_function_name" {
  description = "Weekly digest Lambda. Test it now instead of waiting for Monday: aws lambda invoke --function-name <name> /dev/stdout"
  value       = aws_lambda_function.digest.function_name
}
