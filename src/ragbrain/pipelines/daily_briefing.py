"""Daily briefing pipeline — phone-friendly learning digests.

Two public functions:

    generate_daily_briefing()
        Reads today's RSS/web-indexed news from Qdrant and explains how each
        item relates to RAGBrain's current architecture.  Formatted for
        mobile Telegram reading (short paragraphs, clear headings).

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
Your job: take today's AI news (from the user's RSS feeds, indexed into the
knowledge base) and write a concise, educational summary for the developer
who built RAGBrain.

Format your response for mobile Telegram reading:
- Short paragraphs (max 3 sentences each)
- Use plain language — no unnecessary jargon
- Use HTML formatting only: <b>bold</b> and <i>italic</i>
- Max total length: 800 words

Structure:
1. <b>Today's AI News</b> — 3-5 bullet points of what was covered
2. <b>Relevance to RAGBrain</b> — for each major news item, say in 1-2 sentences
   whether RAGBrain already handles it, partially handles it, or doesn't yet
3. <b>Key Takeaway</b> — one sentence: the most important thing to know today
"""

_BRIEFING_HUMAN = """Today's AI news (from indexed articles):

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

_LINKEDIN_POST_SYSTEM = """You write high-engagement LinkedIn posts for AI builders.

Write one short post based on today's AI news and RAGBrain architecture updates.

Rules:
- Plain text only (no markdown, no HTML)
- Hook in first 2 lines (contrarian or number-driven)
- 3-5 short paragraphs
- 150-300 words
- Mention the builder angle: "I'm building RAGBrain, an open-source agentic RAG framework"
- Include one concrete implementation insight from today's changes
- No external links
- End with one engagement question
"""

_LINKEDIN_POST_HUMAN = """Today's AI news:
{news_content}

RAGBrain architecture summary:
{architecture_summary}

Write one LinkedIn post draft now."""

_TWITTER_THREAD_SYSTEM = """You write concise X/Twitter threads for technical AI updates.

Write exactly 3 tweets in a thread.

Rules:
- Plain text only
- Format exactly:
  Tweet 1: ...
  Tweet 2: ...
  Tweet 3: ...
- Each tweet <= 280 chars
- Tweet 1 must have a strong hook in first 8 words
- Include at least one specific metric or concrete detail in Tweet 1
- Mention practical relevance to RAGBrain (agentic RAG / retrieval / security / evals)
- End Tweet 3 with a question or CTA
"""

_TWITTER_THREAD_HUMAN = """Today's AI news:
{news_content}

RAGBrain architecture summary:
{architecture_summary}

Write the 3-tweet thread now."""


# ---- Helpers -------------------------------------------------------------

# Chunks from RSS + web article ingestion (see ArticlesPipeline.run(also_ingest=True)).
# Legacy Slack chunks are still shown if present so older indexes remain useful.
_NEWS_SOURCE_TYPES = frozenset({"rss", "web", "slack"})


def _load_recent_news_from_kb(lookback_hours: int = 24) -> str:
    """Pull recent RSS/web (and legacy Slack) chunks from Qdrant and return as text."""
    store = None
    try:
        from ragbrain.vectorstore.qdrant import QdrantStore

        store = QdrantStore()
        coll = store.collection_name()

        offset = None
        news_chunks: list[str] = []
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
                if p.get("source_type") in _NEWS_SOURCE_TYPES:
                    news_chunks.append(p.get("content", ""))
            if offset is None:
                break

        if not news_chunks:
            return (
                "(No AI news in the knowledge base yet — set RAGBRAIN_RSS_FEEDS_STR and run "
                "`ragbrain fetch-articles` or enable daily automation so articles are indexed.)"
            )

        return "\n\n---\n\n".join(news_chunks[:20])  # cap at 20 chunks
    except Exception:
        logger.exception("Failed to load news content from Qdrant")
        return "(Could not load indexed news.)"
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

def get_daily_inputs(lookback_hours: int = 24) -> tuple[str, str]:
    """Load shared daily inputs once (for briefing + social drafts).

    Returns:
        tuple(news_content, architecture_summary)
    """
    news_content = _load_recent_news_from_kb(lookback_hours)
    arch = _load_architecture()
    arch_summary = arch[:2000]
    return news_content, arch_summary

def generate_daily_briefing(
    lookback_hours: int = 24,
    news_content: str | None = None,
    architecture_summary: str | None = None,
) -> str:
    """Generate a mobile-friendly Telegram briefing from today's indexed AI news.

    Returns an HTML-formatted string ready to send via Telegram.
    """
    llm = settings.get_fast_llm()

    if news_content is None or architecture_summary is None:
        news_content, architecture_summary = get_daily_inputs(lookback_hours)

    prompt = ChatPromptTemplate.from_messages([
        ("system", _BRIEFING_SYSTEM),
        ("human", _BRIEFING_HUMAN),
    ])
    chain = prompt | llm

    try:
        response = chain.invoke({
            "news_content": news_content,
            "architecture_summary": architecture_summary,
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


def generate_social_posts(news_content: str, architecture_summary: str) -> dict[str, str]:
    """Generate ready-to-copy LinkedIn + X/Twitter drafts.

    Args:
        news_content: Today's indexed AI news text (RSS/web).
        architecture_summary: Concise architecture summary context.

    Returns:
        {"linkedin": "<post>", "twitter": "<thread>"}
        All values are plain text (no HTML) for easy copy-paste.
    """
    llm = settings.get_fast_llm()

    linkedin_chain = ChatPromptTemplate.from_messages([
        ("system", _LINKEDIN_POST_SYSTEM),
        ("human", _LINKEDIN_POST_HUMAN),
    ]) | llm

    twitter_chain = ChatPromptTemplate.from_messages([
        ("system", _TWITTER_THREAD_SYSTEM),
        ("human", _TWITTER_THREAD_HUMAN),
    ]) | llm

    linkedin_text = ""
    twitter_text = ""

    try:
        li_resp = linkedin_chain.invoke({
            "news_content": news_content,
            "architecture_summary": architecture_summary,
        })
        linkedin_text = li_resp.content if hasattr(li_resp, "content") else str(li_resp)
    except Exception:
        logger.exception("Failed to generate LinkedIn draft")
        linkedin_text = "LinkedIn draft unavailable (LLM error)."

    try:
        tw_resp = twitter_chain.invoke({
            "news_content": news_content,
            "architecture_summary": architecture_summary,
        })
        twitter_text = tw_resp.content if hasattr(tw_resp, "content") else str(tw_resp)
    except Exception:
        logger.exception("Failed to generate X/Twitter draft")
        twitter_text = "Twitter draft unavailable (LLM error)."

    return {
        "linkedin": linkedin_text.strip(),
        "twitter": twitter_text.strip(),
    }
