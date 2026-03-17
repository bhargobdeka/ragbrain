"""Assemble the agentic RAG StateGraph.

Flow:
    START
      └─► retrieve
            └─► grade_documents
                  ├─► (relevant / exhausted retries) ─► generate_answer
                  │                                         └─► check_hallucination
                  │                                               ├─► (grounded / max retries) ─► finalize ─► END
                  │                                               └─► (not_grounded) ─► generate_answer  (loop)
                  └─► (irrelevant + retries left) ─► rewrite_query ─► retrieve  (loop)
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ragbrain.agents.nodes.check import check_hallucination, finalize, route_after_check
from ragbrain.agents.nodes.generate import generate_answer
from ragbrain.agents.nodes.grade import grade_documents, route_after_grade
from ragbrain.agents.nodes.retrieve import retrieve
from ragbrain.agents.nodes.rewrite import rewrite_query
from ragbrain.agents.state import RAGState


def build_rag_graph():
    """Build and compile the agentic RAG graph.

    Returns:
        A compiled LangGraph StateGraph ready for invocation.
    """
    builder = StateGraph(RAGState)

    # Register nodes
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("check_hallucination", check_hallucination)
    builder.add_node("finalize", finalize)

    # Entry point
    builder.add_edge(START, "retrieve")

    # After retrieval, always grade
    builder.add_edge("retrieve", "grade_documents")

    # After grading: relevant → generate, irrelevant → rewrite
    builder.add_conditional_edges(
        "grade_documents",
        route_after_grade,
        {"generate": "generate_answer", "rewrite": "rewrite_query"},
    )

    # After rewrite, retrieve again
    builder.add_edge("rewrite_query", "retrieve")

    # After generation, check for hallucinations
    builder.add_edge("generate_answer", "check_hallucination")

    # After check: grounded → finalize, not grounded → regenerate
    builder.add_conditional_edges(
        "check_hallucination",
        route_after_check,
        {"deliver": "finalize", "regenerate": "generate_answer"},
    )

    # Finalize → end
    builder.add_edge("finalize", END)

    return builder.compile()


def query(question: str, user_id: str | None = None) -> dict:
    """Convenience function to run a single query through the RAG graph.

    Args:
        question: The user's question.
        user_id: Optional namespace for multi-tenant isolation.

    Returns:
        Final state dict with 'answer' and 'sources' keys.
    """
    graph = build_rag_graph()
    initial_state: RAGState = {
        "query": question,
        "user_id": user_id,
        "documents": [],
        "retrieval_attempts": 0,
        "grade_result": "",
        "rewritten_query": None,
        "generation": "",
        "sources": [],
        "hallucination_check": "",
        "hallucination_retries": 0,
        "answer": "",
        "error": None,
    }
    return graph.invoke(initial_state)
