# RAGBrain Architecture State

This file is auto-maintained by `ragbrain plan-upgrades`.
It tracks upgrade plans across runs so the planner builds on prior work rather than
repeating itself. Each run appends a new section with its date, summary, and recommendations.

**Do not delete this file** — it is the long-term memory for the self-improving loop.
You may add manual notes in the "Manual Notes" section below; the agent will read them.

---

## Manual Notes

Add any context here that you want the planner to always consider:

- Priority focus: retrieval quality and latency over throughput
- We are a single-developer open-source project — prefer low-effort, high-impact upgrades
- Avoid large infrastructure changes (e.g. moving off Qdrant) without strong justification
- Security hardening is deferred until before the first public release

---

## Run: Friday, March 27, 2026
### Summary
Today's Slack briefing (March 27, 2026) surfaced two distinct signals: a live supply-chain attack on LiteLLM (a direct dependency risk for any Python-based LLM project) and the first live agent-to-agent payment transactions by Santander, Mastercard, and Visa — a leading indicator that multi-agent coordination is moving from research to production. The knowledge base (currently only The RLHF Book) had no indexed material on any of these topics, confirming RAGBrain's knowledge corpus is thin on security and multi-agent literature. Cross-referencing against ARCHITECTURE.md and the (empty) prior-run state, three concrete upgrade recommendations emerge: (1) immediate dependency-pinning and supply-chain hardening triggered by the LiteLLM attack, (2) a multi-agent task-decomposition layer to close the known "no multi-agent coordination" gap, now validated as production-relevant by the payments news, and (3) ingesting dedicated RAG/retrieval security and multi-agent literature into the knowledge base to stop returning empty answers on these topics. No prior recommendations exist to carry forward.
### Recommendations
- [HIGH] **Security / Dependency Management**: Pin all LLM-related PyPI dependencies (litellm, langchain, anthropic, qdrant-client, etc.) to exact versions in requirements.txt / pyproject.toml. Add pip-audit or safety to the CI pipeline to alert on newly published CVEs. Rotate any env vars, API keys, or secrets that were present in environments running LiteLLM 1.82.7–1.82.8. Consider using a private PyPI mirror or hash-verified installs (pip install --require-hashes) for the critical dependency set. *(Effort: DAYS)*
- [MEDIUM] **Agentic RAG — Multi-Agent Coordination**: Introduce a lightweight task-decomposition layer on top of the existing LangGraph StateGraph. A Planner agent breaks complex user queries into sub-tasks; existing CRAG/Self-Reflective RAG nodes handle each sub-task independently; a Synthesiser agent merges results. Start with a two-level hierarchy (Planner → Worker) using Claude 3.5 Haiku for planning to keep cost low. This directly closes the 'no multi-agent coordination / no task decomposition' known limitation. *(Effort: WEEKS)*
- [MEDIUM] **Knowledge Base / Ingestion**: Ingest a curated set of RAG, retrieval, and AI-security papers into the knowledge base so future architecture reviews return substantive answers rather than 'no information found'. Suggested seed list: (a) ColBERT / PLAID papers (late-interaction retrieval), (b) BEIR benchmark paper (retrieval evaluation), (c) GraphRAG / KG-RAG survey, (d) MLSecOps / AI supply-chain security literature (e.g. MITRE ATLAS). Use the existing RSS + WebExtractor pipeline to subscribe to arXiv cs.IR and cs.CR feeds. *(Effort: DAYS)*

### Already Covered
- ✓ Reciprocal Rank Fusion (RRF) for hybrid dense+sparse retrieval — already implemented
- ✓ CRAG (Corrective RAG) with query rewriting — already implemented
- ✓ Self-Reflective RAG with hallucination checking — already implemented
- ✓ CrossEncoder reranking post-RRF — already implemented
- ✓ Semantic chunking with cosine-distance topic boundaries — already implemented
- ✓ AST-based code chunking via Tree-sitter — already implemented
- ✓ LangGraph StateGraph orchestration — already implemented

---
