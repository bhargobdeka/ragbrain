"""LLM-as-judge scoring for RAG evaluation.

Uses the same Claude Haiku fast LLM already wired into the pipeline
(settings.get_fast_llm()) so no new model setup is required.

Judges:
    llm_faithfulness       — is every claim in the answer supported by context?
    llm_relevance          — does the answer address the question?
    llm_context_relevance  — are the retrieved chunks relevant to the query?

Known limitation: using the same model family (Claude) as the pipeline means
the judge may share the same biases. For stricter evaluation swap the judge
to a different provider in settings (e.g. openai:gpt-4o).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ragbrain.config import settings

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    judge_type: str
    score: float        # 0.0–1.0
    passed: bool
    reason: str


# ---------------------------------------------------------------------------
# Faithfulness judge
# ---------------------------------------------------------------------------

_FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a faithfulness evaluator for RAG systems. "
        "Score how well the answer is supported by the provided context documents. "
        "1.0 = every claim is directly supported by the context. "
        "0.0 = answer contains claims that cannot be found in the context. "
        "Be strict: any claim without context support lowers the score.",
    ),
    (
        "human",
        "Context documents:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer to evaluate:\n{answer}\n\n"
        "Faithfulness score (0.0–1.0) and brief reason:",
    ),
])


class _FaithfulnessGrade(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="Faithfulness score")
    reason: str = Field(description="Brief explanation")


def judge_faithfulness(
    question: str,
    answer: str,
    sources: list[dict],
    threshold: float = 0.7,
) -> JudgeResult:
    """Score how well the answer is grounded in the retrieved sources."""
    context = "\n\n---\n\n".join(
        s.get("content", "") or s.get("snippet", "") or s.get("title", "")
        for s in sources[:5]
    )
    if not context.strip():
        return JudgeResult(
            "llm_faithfulness", 0.0, False,
            "No source context available to evaluate faithfulness.",
        )

    llm = settings.get_fast_llm()
    try:
        graded: _FaithfulnessGrade = (
            _FAITHFULNESS_PROMPT | llm.with_structured_output(_FaithfulnessGrade)
        ).invoke({"context": context, "question": question, "answer": answer})
        score, reason = graded.score, graded.reason
    except Exception as e:
        logger.warning("Faithfulness judge error: %s", e)
        score, reason = 0.5, f"Judge call failed: {e}"

    return JudgeResult("llm_faithfulness", score, score >= threshold, reason)


# ---------------------------------------------------------------------------
# Answer relevance judge
# ---------------------------------------------------------------------------

_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an answer relevance evaluator. "
        "Score how directly and completely the answer addresses the question. "
        "1.0 = fully answers the question with no off-topic content. "
        "0.0 = answer does not address the question at all.",
    ),
    (
        "human",
        "Question: {question}\n\n"
        "Answer to evaluate:\n{answer}\n\n"
        "Relevance score (0.0–1.0) and brief reason:",
    ),
])


class _RelevanceGrade(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reason: str


def judge_relevance(
    question: str,
    answer: str,
    threshold: float = 0.7,
) -> JudgeResult:
    """Score how well the answer addresses the original question."""
    if not answer.strip():
        return JudgeResult("llm_relevance", 0.0, False, "Answer is empty.")

    llm = settings.get_fast_llm()
    try:
        graded: _RelevanceGrade = (
            _RELEVANCE_PROMPT | llm.with_structured_output(_RelevanceGrade)
        ).invoke({"question": question, "answer": answer})
        score, reason = graded.score, graded.reason
    except Exception as e:
        logger.warning("Relevance judge error: %s", e)
        score, reason = 0.5, f"Judge call failed: {e}"

    return JudgeResult("llm_relevance", score, score >= threshold, reason)


# ---------------------------------------------------------------------------
# Context relevance judge
# ---------------------------------------------------------------------------

_CONTEXT_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a retrieval quality evaluator. "
        "Score how relevant the retrieved context chunks are to the question. "
        "1.0 = all chunks are highly relevant. "
        "0.0 = retrieved chunks are completely unrelated to the question.",
    ),
    (
        "human",
        "Question: {question}\n\n"
        "Retrieved context:\n{context}\n\n"
        "Context relevance score (0.0–1.0) and brief reason:",
    ),
])


class _ContextRelevanceGrade(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reason: str


def judge_context_relevance(
    question: str,
    sources: list[dict],
    threshold: float = 0.7,
) -> JudgeResult:
    """Score how relevant the retrieved chunks are to the query."""
    context = "\n\n---\n\n".join(
        s.get("content", "") or s.get("snippet", "") or s.get("title", "")
        for s in sources[:5]
    )
    if not context.strip():
        return JudgeResult(
            "llm_context_relevance", 0.0, False,
            "No sources were retrieved — cannot evaluate context relevance.",
        )

    llm = settings.get_fast_llm()
    try:
        graded: _ContextRelevanceGrade = (
            _CONTEXT_RELEVANCE_PROMPT | llm.with_structured_output(_ContextRelevanceGrade)
        ).invoke({"question": question, "context": context})
        score, reason = graded.score, graded.reason
    except Exception as e:
        logger.warning("Context relevance judge error: %s", e)
        score, reason = 0.5, f"Judge call failed: {e}"

    return JudgeResult("llm_context_relevance", score, score >= threshold, reason)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run_judge(
    judge_type: str,
    question: str,
    answer: str,
    sources: list[dict],
    threshold: float = 0.7,
) -> JudgeResult | None:
    """Convenience dispatcher — returns None for unknown judge types."""
    if judge_type == "llm_faithfulness":
        return judge_faithfulness(question, answer, sources, threshold)
    if judge_type == "llm_relevance":
        return judge_relevance(question, answer, threshold)
    if judge_type == "llm_context_relevance":
        return judge_context_relevance(question, sources, threshold)
    return None
