"""Rewrite node: reformulate the query when retrieved documents are irrelevant."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from ragbrain.agents.state import RAGState
from ragbrain.config import settings

_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a query optimizer. Rewrite the user's question to improve document retrieval. "
        "Make it more specific, use alternative terminology, and focus on the core information need. "
        "Return only the rewritten query, nothing else.",
    ),
    (
        "human",
        "Original question: {question}\n\nRewrite this question to improve search results:",
    ),
])


def rewrite_query(state: RAGState) -> dict:
    """Rewrite the query to improve retrieval results."""
    llm = settings.get_fast_llm()
    chain = _REWRITE_PROMPT | llm

    original_query = state["query"]
    response = chain.invoke({"question": original_query})
    rewritten = response.content.strip()

    return {"rewritten_query": rewritten}
