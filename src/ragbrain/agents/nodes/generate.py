"""Generate node: produce a grounded answer from retrieved documents."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from ragbrain.agents.state import RAGState
from ragbrain.config import settings

_GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a knowledgeable assistant. Answer the user's question using ONLY the provided "
        "context documents. If the context does not contain enough information to answer, say so "
        "clearly. Always cite your sources by referencing the document title or URL.\n\n"
        "Context:\n{context}",
    ),
    ("human", "{question}"),
])


def _format_context(state: RAGState) -> tuple[str, list[dict[str, str]]]:
    """Format retrieved documents into a context string and source list."""
    documents = state.get("documents", [])
    context_parts: list[str] = []
    sources: list[dict[str, str]] = []

    for i, result in enumerate(documents, start=1):
        chunk = result.chunk
        source_label = chunk.title or chunk.source_url or f"Document {i}"
        context_parts.append(f"[{i}] {source_label}\n{chunk.content}")

        if not any(s["url"] == chunk.source_url for s in sources):
            sources.append({
                "title": chunk.title or source_label,
                "url": chunk.source_url,
                "snippet": chunk.content[:200],
            })

    return "\n\n---\n\n".join(context_parts), sources


def generate_answer(state: RAGState) -> dict:
    """Generate an answer from the retrieved and graded documents."""
    llm = settings.get_llm()
    chain = _GENERATE_PROMPT | llm

    context, sources = _format_context(state)
    question = state.get("rewritten_query") or state["query"]

    response = chain.invoke({"context": context, "question": question})

    return {
        "generation": response.content,
        "sources": sources,
        "hallucination_retries": state.get("hallucination_retries", 0),
    }
