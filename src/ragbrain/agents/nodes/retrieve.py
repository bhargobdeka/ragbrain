"""Retrieve node: calls the hybrid retriever and populates state."""

from __future__ import annotations

from ragbrain.agents.state import RAGState
from ragbrain.retrieval.hybrid import HybridRetriever

_retriever = HybridRetriever()


def retrieve(state: RAGState) -> dict:
    """Retrieve relevant chunks using hybrid search.

    Uses the rewritten query if one exists (post-CRAG rewrite),
    otherwise uses the original query.
    """
    query = state.get("rewritten_query") or state["query"]
    user_id = state.get("user_id")

    documents = _retriever.retrieve(query=query, user_id=user_id)

    return {
        "documents": documents,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        "rewritten_query": None,  # clear after use
    }
