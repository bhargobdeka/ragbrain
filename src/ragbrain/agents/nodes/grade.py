"""Grade node: LLM-as-judge that assesses document relevance (CRAG pattern).

If fewer than half of the retrieved documents are relevant to the query,
the graph routes to the rewrite node for query reformulation.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ragbrain.agents.state import RAGState
from ragbrain.config import settings

_GRADE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a relevance grader. Assess whether the retrieved document is "
        "relevant to the user question. Give a binary score: 'yes' if relevant, 'no' if not.",
    ),
    (
        "human",
        "Question: {question}\n\nDocument:\n{document}\n\nIs this document relevant? (yes/no)",
    ),
])


class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the document is relevant, 'no' otherwise"
    )


def grade_documents(state: RAGState) -> dict:
    """Grade each retrieved document; set grade_result to relevant/irrelevant."""
    llm = settings.get_fast_llm()
    grader = _GRADE_PROMPT | llm.with_structured_output(GradeDocuments)

    query = state.get("rewritten_query") or state["query"]
    documents = state.get("documents", [])

    if not documents:
        return {"grade_result": "irrelevant"}

    relevant_count = 0
    for result in documents:
        try:
            grade: GradeDocuments = grader.invoke({
                "question": query,
                "document": result.chunk.content[:1000],
            })
            if grade.binary_score == "yes":
                relevant_count += 1
        except Exception:
            relevant_count += 1  # default to relevant on error

    # If fewer than half are relevant, trigger query rewrite
    threshold = max(1, len(documents) // 2)
    grade_result = "relevant" if relevant_count >= threshold else "irrelevant"

    return {"grade_result": grade_result}


def route_after_grade(state: RAGState) -> Literal["generate", "rewrite"]:
    """Conditional edge: route based on grade result and retry count."""
    if state.get("grade_result") == "relevant":
        return "generate"

    attempts = state.get("retrieval_attempts", 0)
    if attempts >= settings.max_retries:
        # Exhausted retries; generate with what we have
        return "generate"

    return "rewrite"
