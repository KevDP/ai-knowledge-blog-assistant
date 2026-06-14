# Variable defaults

variable "aws_region" {
  description = "AWS region infrastructure"
  type        = string
  default     = "us-east-1"
}

variable "bedrock_model_id" {
  description = "Bedrock model ID. Haiku 4.5 requires the US cross-region inference profile. Direct foundation-model invocation returns ValidationException."
  type        = string
  default     = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
}

variable "titan_model_id" {
  description = "Bedrock Titan Embeddings v2 model ID. Used for query and document vectorization in the RAG pipeline."
  type        = string
  default     = "amazon.titan-embed-text-v2:0"
}

variable "monthly_budget_usd" {
  description = "monthly budget in usd. Alarms set in 50% / 80% / 100% (forecasted)."
  type        = number
  default     = 10
}

variable "alarm_email" {
  description = "Email for alarm notifications. Set already in TF_VAR_alarm_email."
  type        = string
}
