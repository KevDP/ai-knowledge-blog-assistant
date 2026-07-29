# EVA: Portfolio RAG Assistant on AWS

Production-grade RAG chatbot answering questions about my professional
background, deployed serverless on AWS. Cost-controlled to <20 mxn/month at
portfolio traffic. Multi-turn conversation memory, contextual retrieval,
and 4-layer FinOps defense.

**Live demo:** [kevdelgado.com](https://kevdelgado.com), click the EVA
icon in the bottom-right and try any question.

## What it does

Any visitor to my portfolio can ask questions like *"What did Kevin study
in Tokyo?"* or *"How much did he reduce ETL latency at Deloitte?"* and
get a grounded answer sourced from a private knowledge base, not
fabricated by the model. The assistant refuses off-topic queries at the
edge (no LLM cost incurred), handles follow-up questions with memory of
prior turns, and stays cheap under sustained traffic.

[![EVA at a glance: RAG + LangGraph agent + cross-encoder reranker](https://kevdelgado.com/diagrams/thumbnails/eva-project-summary-diagram.png)](https://kevdelgado.com/diagrams/eva-project-summary-diagram.html)

## Architecture

[![Phase 1 AWS architecture](https://kevdelgado.com/diagrams/thumbnails/eva-phase1-aws-architecture-diagram.png)](https://kevdelgado.com/diagrams/eva-phase1-aws-architecture-diagram.html)

_Click any diagram in this README to open the interactive version._

## Key technical decisions

**Contextual retrieval:**  embeddings are enriched with document + topic
metadata before vectorization, while the raw chunk is what reaches
Claude.

**Titan embeddings v2 + DynamoDB:** at 20–30 chunks a full DDB scan finishes in ~30.
Alternatives like OpenSearch Serverless start at ~$700/month.
Reevaluation trigger: retrieval latency >500ms.

**Multi-turn conversation memory:** the client sends recent history
with each request. The embedding input for the current turn is enriched
with the last two turns so short follow-ups (*"tell me more about the
second one"*) inherit topic context. The relevance-gate threshold
relaxes by 40% when history is present, since valid follow-ups score
naturally lower. Response caching is skipped when history is non-empty
to avoid cross-conversation collisions.

**Reformulation-aware retrieval:** the gate is adaptive: on single-turn
queries a strict threshold blocks noise; on follow-ups a relaxed
threshold accepts context-dependent phrasing.

**Agent layer (local track, not deployed):** `src/agent.py` rebuilds the
pipeline as a LangGraph state machine: route by question type, resolve
pronouns against history, retrieve, rerank, grade, reformulate on weak
recall, answer. `src/rerank.py` adds a cross-encoder second stage
(`ms-marco-MiniLM-L-6-v2`) over the bi-encoder's top-N, which fixed the case
where two similar KB chunks were consistently ranked wrong. `EVA_RERANK=0`
turns it into a pass-through so the eval suite can A/B it without code
changes. This track runs locally and under `evals/`; the deployed Lambda is
still the Phase 1.4 pipeline described above.

[![Agent graph: routing, memory, reformulation](https://kevdelgado.com/diagrams/thumbnails/ai-workflow-diagram.png)](https://kevdelgado.com/diagrams/ai-workflow-diagram.html)

[![Two-stage retrieval with cross-encoder reranking](https://kevdelgado.com/diagrams/thumbnails/retrieval-pipeline-diagram.png)](https://kevdelgado.com/diagrams/retrieval-pipeline-diagram.html)

_Interactive only (no static export yet):
[memory trace](https://kevdelgado.com/diagrams/eva-memory-trace-diagram.html)
·
[reformulation guard](https://kevdelgado.com/diagrams/eva-reformulation-guard-diagram.html)_

## 4-layer FinOps defense

Every layer is free or near-free. Together they cap runaway cost
regardless of traffic pattern.

| # | Layer | Where | Cost blocked |
|---|---|---|---|
| 1 | Relevance gate | Lambda in-code | Off-topic queries never reach the LLM |
| 2 | Input length cap | Lambda in-code | Prompt-injection with large payloads |
| 3 | Rate limit | API Gateway | Volume abuse: 429s before invoking Lambda |
| 4 | Budget alarms | AWS Budgets | Reactive worst-case cap |

_Interactive only (no static export yet):
[how the relevance threshold was calibrated](https://kevdelgado.com/diagrams/eva-threshold-calibration-diagram.html)_

**Honest limitation:** the relevance gate is probabilistic. Queries that
share vocabulary with the KB (*"AWS certification study guide"*) pass
the gate. The system prompt handles this gracefully by redirecting
without hallucinating; Layer 3 remains the actual defense against volume
abuse.

## Cost reality

Numbers use Bedrock Haiku 4.5 pricing (mid-2026):

| Scenario | No defense | With 4-layer stack | + response cache (40% hit) |
|---|---|---|---|
| 500 visitors/mo × 10 queries | $2.30 | $2.30 | **$1.38** |
| Bot scraper 1k queries/day | ~$135/mo | **$0** (429s + gate) | **$0** |

## Tech stack

| Layer | Choice |
|---|---|
| Compute | AWS Lambda (Python 3.12) |
| API | API Gateway HTTP API |
| LLM | Amazon Bedrock: Haiku 4.5 via inference profile |
| Embeddings | Amazon Bedrock |
| Vector store | DynamoDB |
| Cache | DynamoDB |
| IaC | Terraform 1.9+ with S3 + DDB remote state |
| CI/CD | GitHub Actions with OIDC federation |
| Observability | CloudWatch Logs + Insights |

## Repository layout

```
infra/                     Terraform: Lambda, API Gateway, DynamoDB, IAM (least-privilege), budgets, outputs
    main.tf                Provider + remote backend (S3 + DDB lock)
    api_gateway.tf         HTTP API + CORS + throttling + Lambda perm
    lambda.tf              Function + log group + env vars
    dynamodb.tf            Knowledge table + response cache (TTL)
    iam.tf                 Execution role with narrow-scoped policies
    budget.tf              Cost alarms at 50 / 80 / 100 %
    variables.tf           Configurable model IDs, thresholds, TTLs

lambda/                    DEPLOYED. Terraform zips this directory and nothing else.
    handler.py             Phase 1.4 pipeline: validate - cache - embed - retrieve - gate - invoke - cache - log
    requirements.txt       Intentionally empty: boto3 ships with the Python 3.12 runtime

src/                       Agent track (blog Parts 2 and 3). Runs locally, NOT in the Lambda bundle.
    agent.py               LangGraph state machine: route - resolve - retrieve - rerank - grade - reformulate - answer
    rerank.py              Cross-encoder precision stage, disabled with EVA_RERANK=0
    retrieve.py            Embedding + cosine top-k
    ingest.py              Chunking + embedding of KB files
    llm.py                 Claude client wrapper
    chat.py                Local CLI for manual probing

evals/
    run_evals.py           Scored regression run over the golden set
    golden_set.json        Labelled queries, including known_limitation cases

knowledge_demo/            Six placeholder KB files so the pipeline runs without the private KB
scripts/
    ingest_knowledge.py    Populates DynamoDB from KB files

.github/workflows/
    terraform.yml          OIDC federation - plan on dev, apply on main
```

## Deploy

Prerequisites:
- AWS account with Bedrock model access enabled for Claude Haiku 4.5 and Titan Embeddings v2 (`us-east-1`)
- S3 bucket + DynamoDB table for Terraform remote state
- OIDC identity provider for GitHub Actions (created once per account)
- IAM role trust-scoped to this repository (Terraform + Bedrock invoke)

Full walkthrough, including the OIDC setup, the KB ingestion pipeline that lives in a separate private repo,
and the honest bugs I hit.

## Blog posts

Companion long-form writeups of the design decisions, all at
[kevdelgado.com/blog](https://kevdelgado.com/blog):

- **[Part 1: a RAG assistant over my own portfolio](https://kevdelgado.com/blog/construyendo-eva-parte-1/)**
  (2026-07-01): the full pipeline, contextual retrieval, the 4-layer FinOps
  defense, and why a DynamoDB scan is enough at this scale.
- **[Part 2: from pipeline to agent](https://kevdelgado.com/blog/construyendo-eva-parte-2/)**
  (2026-07-02): LangGraph orchestration, routing by question type,
  conversation memory, query reformulation when retrieval scores weak, and an
  eval suite so regressions stop being guesswork.
- **[Part 3: polished implementations and a testing-focused demo](https://kevdelgado.com/blog/construyendo-eva-parte-3/)**
  (2026-07-03): two-stage retrieval with a cross-encoder reranker, and a guard
  that reverts a reformulation when it degrades more than it improves.

## What's not included

- **The real knowledge base.** My actual KB content is private: visible only through EVA's answers,
  never committed to a public repository.
  `knowledge_demo/` holds placeholder files so the pipeline runs end to end without it.
- **Frontend widget.** The chat UI (Next.js) lives with the portfolio site, not here.
- **Production ingest workflow.** `scripts/ingest_knowledge.py` populates DynamoDB from local files.
  The private workflow that syncs the real KB from S3 and invalidates the cache atomically lives elsewhere,
  and is not required to run the backend defined here.
