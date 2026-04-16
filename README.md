# RAGBrain

**Open-source, BYOK document intelligence with agentic RAG — a personal NotebookLM you can run yourself.**

Drop your PDFs, paste your RSS feeds. Query your entire reading history. Get daily AI lessons delivered to Telegram.

---

## What it does

RAGBrain is a framework for building a personal knowledge base powered by advanced RAG:

- **Ingest anything** — PDFs (books, papers), web articles, Substack, Medium, RSS feeds
- **Advanced retrieval** — Hybrid dense + BM25 search, RRF fusion, CrossEncoder reranking
- **Agentic RAG** — CRAG (Corrective RAG) + Self-Reflective RAG via LangGraph
- **Daily digests** — Morning article summaries via Telegram; optional evening book lesson (off by default)
- **BYOK** — Bring Your Own (Anthropic) API key; embeddings and reranking run locally for free

---

## Architecture

```
Raw Sources (PDF / Web / RSS)
          │
          ▼
   Extraction Layer
   ├── PDFExtractor (PyMuPDF)
   ├── WebExtractor (trafilatura)
   └── RSSExtractor (feedparser)
          │
          ▼
   Chunking Layer
   ├── SemanticChunker  (cosine-distance sentence splitting)
   ├── CodeChunker      (function/class boundary aware)
   └── ChunkRouter      (auto-detects block type)
          │
          ▼
   Qdrant Vector Store  (namespaced per user)
          │
          ▼
   Hybrid Retrieval
   ├── Dense search (sentence-transformers/all-mpnet-base-v2)
   ├── BM25 sparse search (rank_bm25)
   ├── Reciprocal Rank Fusion
   └── CrossEncoder reranker (ms-marco-MiniLM-L-6-v2)
          │
          ▼
   Agentic RAG Graph (LangGraph)
   ├── retrieve → grade (CRAG) → rewrite → retrieve [loop]
   └── generate → hallucination check → finalize [loop]
          │
          ▼
   Delivery
   ├── Telegram bot (/query, /digest, /ingest)
   ├── CLI (ragbrain query "...")
   └── Scheduler (morning articles; optional evening book lesson)
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for Qdrant)
- An [Anthropic API key](https://console.anthropic.com)

### 1. Clone and install

```bash
git clone https://github.com/bhargobdeka/ragbrain.git
cd ragbrain
make setup
```

This installs all dependencies and starts Qdrant via Docker.

### 2. Configure

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

```env
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

Everything else has sensible defaults.

### 3. Ingest your first document

```bash
# Index a PDF book
ragbrain ingest path/to/your-book.pdf

# Index a web article
ragbrain ingest https://example.substack.com/p/article-slug

# Index and register as a book (for daily lessons)
ragbrain ingest path/to/your-book.pdf
# → answer "yes" when asked to register as a book
```

### 4. Query your knowledge base

```bash
ragbrain query "What is RLHF and how does it compare to RLAIF?"
```

### 5. Get a digest

```bash
ragbrain digest
```

### 6. Start the Telegram bot (optional)

1. Create a bot via [@BotFather](https://t.me/BotFather) and get your token
2. Add to `.env`:
   ```env
   RAGBRAIN_TELEGRAM_BOT_TOKEN=your-bot-token
   RAGBRAIN_TELEGRAM_CHAT_ID=your-chat-id
   ```
3. Start the bot:
   ```bash
   ragbrain serve
   ```

Bot commands: `/query`, `/digest`, `/ingest`, `/books`

---

## Configuration Reference

All settings live in `.env` (see `.env.example` for the full list).

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key |
| `RAGBRAIN_LLM_MODEL` | `claude-3-5-sonnet-20241022` | Main LLM for generation |
| `RAGBRAIN_LLM_FAST_MODEL` | `claude-3-5-haiku-20241022` | Fast LLM for grading/rewriting |
| `RAGBRAIN_QDRANT_URL` | `http://localhost:6333` | Qdrant instance URL |
| `RAGBRAIN_EMBEDDING_MODEL` | `all-mpnet-base-v2` | Local embedding model |
| `RAGBRAIN_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local CrossEncoder |
| `RAGBRAIN_RSS_FEEDS` | *(empty)* | Comma-separated RSS feed URLs |
| `RAGBRAIN_INTERESTS` | `RAG systems, LLM agents, ...` | Topics for relevance scoring |
| `RAGBRAIN_TELEGRAM_BOT_TOKEN` | *(optional)* | Telegram bot token |
| `RAGBRAIN_TELEGRAM_CHAT_ID` | *(optional)* | Your Telegram chat ID |

---

## CLI Reference

```bash
ragbrain ingest <path_or_url>     # Index a PDF or URL
ragbrain query  "<question>"      # Ask a question
ragbrain digest                   # Print today's digest
ragbrain serve                    # Start Telegram bot
ragbrain schedule                 # Start automated scheduler
ragbrain fetch-articles           # Fetch + summarize RSS feeds now
```

### Makefile shortcuts

```bash
make setup                        # Install + start Qdrant
make ingest PATH=./book.pdf       # Ingest a file
make ingest URL=https://...       # Ingest a URL
make query Q="your question"      # Query
make digest                       # Generate digest
make serve                        # Start Telegram bot
make test                         # Run tests
make lint                         # Run linter
```

---

## How the Agentic RAG works

RAGBrain uses two advanced RAG patterns implemented as a LangGraph `StateGraph`:

**CRAG (Corrective RAG)**
- After retrieval, an LLM grades each document for relevance
- If fewer than half are relevant, the query is automatically rewritten and retrieval is retried (up to `RAGBRAIN_MAX_RETRIES` times)

**Self-Reflective RAG**
- After generation, an LLM checks whether every claim in the answer is grounded in the retrieved documents
- If not grounded, the generator is called again (up to `RAGBRAIN_MAX_HALLUCINATION_RETRIES` times)

---

## Adding a Premium Extractor (Optional)

The extractor layer is designed for easy replacement. To add LandingAI ADE or LLMWhisperer:

1. Subclass `BaseExtractor` in `src/ragbrain/ingestion/extractors/`
2. Implement `can_handle(source)` and `extract(source) -> Document`
3. Register it in `IngestionPipeline._extract()` before the default extractors

---

## Project Structure

```
ragbrain/
├── src/ragbrain/
│   ├── config.py            # All settings via pydantic-settings
│   ├── models.py            # Document, Chunk, RetrievalResult, Digest
│   ├── ingestion/           # Extractors + chunkers + pipeline
│   ├── vectorstore/         # Qdrant client
│   ├── retrieval/           # Dense + BM25 + RRF + CrossEncoder
│   ├── agents/              # LangGraph CRAG + self-reflective RAG
│   ├── pipelines/           # Articles pipeline, Books pipeline
│   ├── delivery/            # Telegram bot + formatter
│   ├── scheduler.py         # APScheduler for automated delivery
│   └── cli.py               # Typer CLI entry point
├── tests/                   # Unit tests
├── _planning/               # Design docs (gitignored)
├── docker-compose.yml       # Qdrant
├── Makefile                 # Developer shortcuts
├── pyproject.toml           # Dependencies
└── .env.example             # Configuration template
```

---

## License

MIT — see [LICENSE](LICENSE).
