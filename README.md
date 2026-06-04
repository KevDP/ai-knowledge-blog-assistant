# AI Knowledge Assistant

RAG-powered chatbot that answers questions about Kevin Delgado — backend for the EVA assistant on [kev-blog](https://github.com/KevDP/kev-blog).

> **Status:** Phase 0 (local RAG, no cloud). Phase 1 (AWS Lambda + Bedrock + DynamoDB) planned.

## Architecture

**Phase 0 (current):** Pure Python. No cloud.

```
knowledge/*.md  →  ingest.py  →  embeddings.json
                                       │
   user question  ──────────────→  retrieve.py  ──→  top-k chunks
                                                          │
                                                          ▼
                                              llm.py  →  Claude API  →  answer
```

**Phase 1 (planned):** API Gateway → Lambda (Python) → Bedrock (Claude Haiku) → DynamoDB. Terraform-managed.

## Quickstart (Phase 0)

```bash
# 1. Setup
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Config
cp .env.example .env
# Edit .env and paste your ANTHROPIC_API_KEY

# 3. Build the index
python -m src.ingest

# 4. Chat
python -m src.chat
```

## Project structure

```
ai-knowledge-assistant/
├── knowledge/          Source-of-truth markdown (about, experience, projects, ...)
├── src/
│   ├── ingest.py       Chunk + embed knowledge → embeddings.json
│   ├── retrieve.py     Cosine similarity, returns top-k chunks
│   ├── llm.py          Claude API client (Anthropic SDK)
│   └── chat.py         CLI entrypoint
├── .env.example        Template — copy to .env
└── requirements.txt
```

## Tech (Phase 0)

- **Embeddings:** `sentence-transformers` with `BAAI/bge-small-en-v1.5` (384-dim, CPU)
  - Contextual chunking: documents embedded with `Document | Topic | Content` metadata prefix; raw chunks sent to LLM
- **Vector store:** plain JSON on disk (sufficient for <1000 chunks)
- **LLM:** Claude Haiku 4.5 via Anthropic API
- **Relevance gate:** cosine similarity threshold 0.55 (probabilistic — see code comments for tuning notes and known limits)
- **No cloud, no Docker, no DB.** Total infra cost: $0.

## License

MIT
