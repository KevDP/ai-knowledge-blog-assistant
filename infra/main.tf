# ─────────────────────────────────────────────────────────────────────────────
# Provider, remote backend
# Backend S3 + DDB table
# ─────────────────────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.9.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.70"
    }
  }

  backend "s3" {
    bucket         = "kev-ai-kb-tfstate-eva"
    key            = "phase-1/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "ai-kb-tflock"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region

  # default_tags to supported resources
  # cost management best practice

  default_tags {
    tags = {
      Project     = "ai-knowledge-assistant"
      Environment = "dev"
      ManagedBy   = "terraform"
      Repo        = "KevDP/ai-knowledge-blog-assistant"
    }
  }
}

# Prefix determined for next project resources to be created
# Role can't touch resources out of this prefix

locals {
  name_prefix = "ai-kb"
}
