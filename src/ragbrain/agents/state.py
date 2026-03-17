"""LangGraph state for the agentic RAG pipeline."""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict

from ragbrain.models import RetrievalResult


def _keep_last(existing: Any, new: Any) -> Any:
    """Reducer: always keep the latest value."""
    return new


class RAGState(TypedDict):
    """Shared state flowing through the RAG graph nodes."""

    # Input
    query: str                                  # original user question
    user_id: str | None                         # for namespace isolation

    # Retrieval
    documents: list[RetrievalResult]            # retrieved + reranked chunks
    retrieval_attempts: int                     # how many times we've retrieved

    # Grading (CRAG)
    grade_result: str                           # "relevant" | "irrelevant"
    rewritten_query: str | None                 # query after rewriting

    # Generation
    generation: str                             # answer from LLM
    sources: list[dict[str, str]]               # [{title, url, snippet}]

    # Hallucination check
    hallucination_check: str                    # "grounded" | "not_grounded"
    hallucination_retries: int

    # Final output
    answer: str
    error: str | None
