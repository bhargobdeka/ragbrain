"""Retrieve node: calls the hybrid retriever and populates state."""

from __future__ import annotations

import threading

from ragbrain.agents.state import RAGState
from ragbrain.retrieval.hybrid import HybridRetriever
from ragbrain.retrieval.intent import detect_source_intent

# Lazily initialised on first call so model loading and the Qdrant lock are
# only acquired when a real query runs, not at import time.
# Lock prevents a race condition where Deep Agents calls search_knowledge_base
# concurrently from multiple threads, both seeing _retriever is None and both
# trying to open a second QdrantClient (which fails with "already accessed").
_retriever: HybridRetriever | None = None
_retriever_lock = threading.Lock()


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:   # double-checked locking
                _retriever = HybridRetriever()
    return _retriever


def retrieve(state: RAGState) -> dict:
    """Retrieve relevant chunks using hybrid search.

    Uses the rewritten query if one exists (post-CRAG rewrite), otherwise the
    original query.  On the first attempt, the query is inspected for source
    intent (e.g. "Slack briefing", "book chapter") and a metadata filter is
    applied to narrow Qdrant to the right source type.  After a CRAG rewrite,
    the filter is relaxed so the rewrite can search more broadly.
    """
    query = state.get("rewritten_query") or state["query"]
    user_id = state.get("user_id")
    is_rewrite = bool(state.get("rewritten_query"))

    # Only apply source-type filter on the first attempt.
    # After a CRAG rewrite we broaden the search to avoid over-constraining.
    filters: dict | None = None
    if not is_rewrite:
        intent = detect_source_intent(state["query"])
        if intent is not None:
            filters = {"source_type": intent.value}

    documents = _get_retriever().retrieve(query=query, user_id=user_id, filters=filters)

    return {
        "documents": documents,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        "rewritten_query": None,  # clear after use
    }
