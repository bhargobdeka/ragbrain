# RAGBrain Architecture — Self-Description

> This file is used by the Architecture Review Agent to understand what RAGBrain
> currently implements.  It is compared against recent AI research to identify
> gaps and suggest upgrades.  Keep it up to date when components change.

## Chunking

| Component | Approach | Limitations |
|---|---|---|
| Text chunking | **Semantic chunking** — cosine distance between sentence embeddings, split at topic boundaries (85th-percentile threshold) | No overlap / sliding window; may create very short chunks |
| Code chunking | **Tree-sitter AST** — extracts functions, classes, methods as complete units with scope_chain, docstring, imports metadata | Only 6 grammars (Python, JS, TS, Go, Rust, Java); fallback to regex for others |

## Embeddings

| Component | Model | Dimensions | Notes |
|---|---|---|---|
| Text embedding | `all-mpnet-base-v2` (sentence-transformers) | 768 | Local, CPU-based |
| Code embedding | `microsoft/unixcoder-base` (HuggingFace transformers) | 768 | AST-aware pre-training; local, ~500 MB |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | — | CrossEncoder, used post-RRF on top-20 candidates |

## Retrieval

| Component | Approach |
|---|---|
| Dense search | Qdrant named vectors — queries both "text" and "code" spaces in parallel |
| Sparse search | BM25 (rank_bm25 library) — built over all stored chunk texts |
| Fusion | Reciprocal Rank Fusion (k=60) — merges dense and sparse ranked lists |
| Reranking | CrossEncoder re-scores top-20 RRF candidates → returns top-5 |

## Agentic RAG

| Pattern | Implementation |
|---|---|
| CRAG (Corrective RAG) | LLM grades each chunk for relevance → if < 50% relevant, rewrites query and re-retrieves (max 2 retries) |
| Self-Reflective RAG | After generation, LLM checks if answer is grounded in source documents → regenerates if hallucinated (max 2 retries) |
| Orchestration | LangGraph StateGraph with conditional edges for the retry loops |

## LLM

| Role | Model |
|---|---|
| Generation | Claude Sonnet 4.6 (Anthropic) |
| Grading / Rewriting / Checking | Claude Haiku 4.5 (Anthropic) |
| Provider | BYOK via ANTHROPIC_API_KEY |

## Vector Store

| Component | Details |
|---|---|
| Engine | Qdrant (local mode by default — no Docker) |
| Schema | Named vectors: {"text": mpnet, "code": unixcoder} per point |
| Multi-tenancy | Collection-per-user pattern |

## Ingestion Sources

| Source | Extractor | Library |
|---|---|---|
| PDF books | PDFExtractor | PyMuPDF |
| Web articles / blogs | WebExtractor | trafilatura |
| RSS feeds | RSSExtractor | feedparser |
| Slack DM (AI news) | SlackExtractor | slack-sdk |

## Pipelines

| Pipeline | Function |
|---|---|
| Articles | RSS fetch → dedup by embedding similarity → LLM relevance scoring → summarize top articles |
| Books | PDF chapter detection → track progress in JSON → daily 3-bullet micro-lesson + reflection question |
| Architecture Review | Read Slack news → load this file → LLM gap analysis → prioritised recommendations |

## Delivery

| Channel | Library |
|---|---|
| CLI | Typer + Rich |
| Telegram | python-telegram-bot |
| Slack (posting back) | slack-sdk |

## Known Limitations

- No multi-modal ingestion yet (images, diagrams extracted as text only)
- No graph-based retrieval (knowledge graphs, dependency graphs)
- No multi-agent coordination (single RAG agent, no task decomposition)
- No streaming generation (responses arrive all at once)
- No ColBERT or late-interaction retrieval models
- No fine-tuned embedding models (uses off-the-shelf only)
- No user feedback loop for retrieval quality improvement
- Code embeddings limited to 512 tokens per chunk (UniXcoder max)
