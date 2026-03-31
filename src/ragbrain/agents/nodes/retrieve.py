"""Retrieve node: calls the hybrid retriever and populates state."""

from __future__ import annotations

from ragbrain.agents.state import RAGState
from ragbrain.retrieval.hybrid import HybridRetriever

# Lazily initialised on first call so model loading and the Qdrant lock are
# only acquired when a real query runs, not at import time.
_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def retrieve(state: RAGState) -> dict:
    """Retrieve relevant chunks using hybrid search.

    Uses the rewritten query if one exists (post-CRAG rewrite),
    otherwise uses the original query.
    """
    query = state.get("rewritten_query") or state["query"]
    user_id = state.get("user_id")

    documents = _get_retriever().retrieve(query=query, user_id=user_id)

    return {
        "documents": documents,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        "rewritten_query": None,  # clear after use
    }
