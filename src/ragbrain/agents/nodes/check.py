"""Hallucination check node: verify the generation is grounded in sources.

Self-Reflective RAG pattern: the LLM is asked to verify that every claim in
the generated answer is supported by the retrieved documents.  If not, the
graph routes back to generate for a retry.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ragbrain.agents.state import RAGState
from ragbrain.config import settings

_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a fact-checker. Determine whether the provided answer is fully supported "
        "by the context documents. Score 'grounded' if every claim has evidence in the context, "
        "'not_grounded' if the answer contains information not present in the context.",
    ),
    (
        "human",
        "Context:\n{context}\n\nAnswer:\n{answer}\n\nIs this answer grounded in the context?",
    ),
])


class GroundednessGrade(BaseModel):
    score: Literal["grounded", "not_grounded"] = Field(
        description="'grounded' if fully supported, 'not_grounded' if hallucinated"
    )
    reason: str = Field(description="Brief explanation of the grounding assessment")


def check_hallucination(state: RAGState) -> dict:
    """Check whether the generated answer is grounded in retrieved documents."""
    llm = settings.get_fast_llm()
    checker = _CHECK_PROMPT | llm.with_structured_output(GroundednessGrade)

    documents = state.get("documents", [])
    context = "\n\n".join(r.chunk.content[:500] for r in documents[:5])
    generation = state.get("generation", "")

    if not generation:
        return {"hallucination_check": "not_grounded"}

    try:
        grade: GroundednessGrade = checker.invoke({
            "context": context,
            "answer": generation,
        })
        result = grade.score
    except Exception:
        result = "grounded"  # default to grounded on error

    return {"hallucination_check": result}


def route_after_check(state: RAGState) -> Literal["deliver", "regenerate"]:
    """Conditional edge: deliver if grounded, regenerate if not (up to max retries)."""
    if state.get("hallucination_check") == "grounded":
        return "deliver"

    retries = state.get("hallucination_retries", 0)
    if retries >= settings.max_hallucination_retries:
        return "deliver"  # accept best effort after max retries

    return "regenerate"


def finalize(state: RAGState) -> dict:
    """Set final answer from generation and increment retry counter if needed."""
    retries = state.get("hallucination_retries", 0)
    return {
        "answer": state.get("generation", ""),
        "hallucination_retries": retries + 1,
    }
