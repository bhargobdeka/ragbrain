"""Generate node: produce a grounded answer from retrieved documents."""

from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from ragbrain.agents.state import RAGState
from ragbrain.config import settings

_GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a knowledgeable assistant. Answer the user's question using ONLY the provided "
        "context documents. If the context does not contain enough information to answer, say so "
        "clearly. Always cite your sources by referencing the document title or URL.\n\n"
        "Code chunks include structural metadata (Scope, Language, Docstring, Imports) to help "
        "you understand their purpose and location. Use this metadata when explaining code.\n\n"
        "Context:\n{context}",
    ),
    ("human", "{question}"),
])


def _format_chunk(i: int, result: Any) -> str:
    """Render a single retrieved chunk as a context block for the LLM.

    Code chunks get extra structural context — scope, docstring, imports —
    so the model understands what the code unit does and where it lives.
    """
    from ragbrain.models import BlockType
    chunk = result.chunk
    source_label = chunk.title or chunk.source_url or f"Document {i}"

    if chunk.block_type == BlockType.CODE:
        header_parts = [f"[{i}] {source_label}"]
        if chunk.scope_chain:
            header_parts.append(f"Scope: {chunk.scope_chain}")
        if chunk.language:
            header_parts.append(f"Language: {chunk.language}")
        if chunk.docstring:
            header_parts.append(f'Docstring: "{chunk.docstring}"')
        imports: list[str] = chunk.metadata.get("imports", [])
        if imports:
            header_parts.append("Imports:\n  " + "\n  ".join(imports[:10]))
        header = "\n".join(header_parts)
        return f"{header}\n\n```{chunk.language or ''}\n{chunk.content}\n```"

    return f"[{i}] {source_label}\n{chunk.content}"


def _format_context(state: RAGState) -> tuple[str, list[dict[str, str]]]:
    """Format retrieved documents into a context string and source list."""
    documents = state.get("documents", [])
    context_parts: list[str] = []
    sources: list[dict[str, str]] = []

    for i, result in enumerate(documents, start=1):
        chunk = result.chunk
        source_label = chunk.title or chunk.source_url or f"Document {i}"
        context_parts.append(_format_chunk(i, result))

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
