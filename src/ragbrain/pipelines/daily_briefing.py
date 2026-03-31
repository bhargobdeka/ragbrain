"""Daily briefing pipeline — phone-friendly learning digests.

Two public functions:

    generate_daily_briefing()
        Reads today's Slack news from Qdrant, explains what Tuk covered and
        how each item relates to RAGBrain's current architecture.  Formatted
        for mobile Telegram reading (short paragraphs, clear headings).

    architecture_snapshot()
        Reads ARCHITECTURE.md and generates a plain-English explanation of
        what RAGBrain currently does — no jargon, readable on a phone.  Good
        for keeping up with the codebase while away from the desk.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from ragbrain.config import settings

logger = logging.getLogger(__name__)

_ARCHITECTURE_PATH = Path(__file__).resolve().parents[3] / "ARCHITECTURE.md"

# ---- Prompts -------------------------------------------------------------

_BRIEFING_SYSTEM = """You are RAGBrain's daily learning assistant.
Your job: take today's AI news (from Tuk's Slack briefing) and write a
concise, educational summary for the developer who built RAGBrain.

Format your response for mobile Telegram reading:
- Short paragraphs (max 3 sentences each)
- Use plain language — no unnecessary jargon
- Use HTML formatting only: <b>bold</b> and <i>italic</i>
- Max total length: 800 words

Structure:
1. <b>Today's AI News (from Tuk)</b> — 3-5 bullet points of what was covered
2. <b>Relevance to RAGBrain</b> — for each major news item, say in 1-2 sentences
   whether RAGBrain already handles it, partially handles it, or doesn't yet
3. <b>Key Takeaway</b> — one sentence: the most important thing to know today
"""

_BRIEFING_HUMAN = """Today's Slack news:

{news_content}

---

RAGBrain's current architecture summary:

{architecture_summary}

Write the briefing now."""


_SNAPSHOT_SYSTEM = """You are explaining a software project to its developer
who is on vacation and reading on their phone.

Write a plain-English snapshot of what the RAGBrain system currently does.
Rules:
- No code, no file paths, no jargon
- Use HTML: <b>bold</b> for component names, <i>italic</i> for key terms
- Max 600 words
- Structure: what it does overall, then each major component in 2-3 sentences
- End with "What's coming next" based on architecture-state.md if provided
"""

_SNAPSHOT_HUMAN = """Here is the full technical architecture document:

{architecture_md}

{state_section}

Write the plain-English snapshot now."""


# ---- Helpers -------------------------------------------------------------

def _load_recent_slack_content(lookback_hours: int = 24) -> str:
    """Pull the most recent Slack chunks from Qdrant and return as text."""
    store = None
    try:
        from ragbrain.vectorstore.qdrant import QdrantStore

        store = QdrantStore()
        coll = store.collection_name()

        # Scroll all points, filter by source_type=slack
        offset = None
        slack_chunks: list[str] = []
        while True:
            results, offset = store._client.scroll(
                collection_name=coll,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for pt in results:
                p = pt.payload or {}
                if p.get("source_type") == "slack":
                    slack_chunks.append(p.get("content", ""))
            if offset is None:
                break

        if not slack_chunks:
            return "(No Slack news ingested yet — run `ragbrain ingest-slack` first.)"

        return "\n\n---\n\n".join(slack_chunks[:20])  # cap at 20 chunks
    except Exception:
        logger.exception("Failed to load Slack content from Qdrant")
        return "(Could not load Slack news.)"
    finally:
        if store is not None:
            store.close()


def _load_architecture() -> str:
    if _ARCHITECTURE_PATH.exists():
        return _ARCHITECTURE_PATH.read_text(encoding="utf-8")
    return "(ARCHITECTURE.md not found.)"


def _load_architecture_state() -> str:
    state_path = Path("architecture-state.md")
    if state_path.exists():
        content = state_path.read_text(encoding="utf-8")
        # Return only the last 1500 chars (most recent entries)
        return content[-1500:]
    return ""


# ---- Public API ----------------------------------------------------------

def generate_daily_briefing(lookback_hours: int = 24) -> str:
    """Generate a mobile-friendly Telegram briefing of today's Slack news.

    Returns an HTML-formatted string ready to send via Telegram.
    """
    llm = settings.get_fast_llm()

    news_content = _load_recent_slack_content(lookback_hours)
    arch = _load_architecture()
    # Only pass the first 2000 chars of architecture to keep prompt short
    arch_summary = arch[:2000]

    prompt = ChatPromptTemplate.from_messages([
        ("system", _BRIEFING_SYSTEM),
        ("human", _BRIEFING_HUMAN),
    ])
    chain = prompt | llm

    try:
        response = chain.invoke({
            "news_content": news_content,
            "architecture_summary": arch_summary,
        })
        content = response.content if hasattr(response, "content") else str(response)
        date_str = datetime.now(timezone.utc).strftime("%A, %B %d")
        return f"<b>RAGBrain Daily Briefing — {date_str}</b>\n\n{content}"
    except Exception:
        logger.exception("Failed to generate daily briefing")
        return (
            "<b>Daily Briefing unavailable</b>\n\n"
            "Could not generate briefing (LLM error). "
            "Check logs with <code>tail -f ~/.ragbrain/scheduler.log</code>."
        )


def architecture_snapshot() -> str:
    """Generate a plain-English snapshot of what RAGBrain currently does.

    Returns an HTML-formatted string ready to send via Telegram.
    Suitable for on-demand reading while away from the codebase.
    """
    llm = settings.get_fast_llm()

    arch_md = _load_architecture()
    state_content = _load_architecture_state()
    state_section = (
        f"\n\nRecent upgrade plans and notes:\n\n{state_content}"
        if state_content.strip()
        else ""
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SNAPSHOT_SYSTEM),
        ("human", _SNAPSHOT_HUMAN),
    ])
    chain = prompt | llm

    try:
        response = chain.invoke({
            "architecture_md": arch_md,
            "state_section": state_section,
        })
        content = response.content if hasattr(response, "content") else str(response)
        date_str = datetime.now(timezone.utc).strftime("%B %d, %Y")
        return (
            f"<b>RAGBrain Architecture Snapshot — {date_str}</b>\n\n"
            f"{content}\n\n"
            f"<i>Send /architecture any time for an updated snapshot.</i>"
        )
    except Exception:
        logger.exception("Failed to generate architecture snapshot")
        return (
            "<b>Architecture snapshot unavailable</b>\n\n"
            "LLM error — check logs."
        )
