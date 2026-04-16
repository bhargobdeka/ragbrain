"""Architecture Review Agent — self-improving RAG pipeline.

Loads recent AI news from configured RSS feeds (same as the article digest),
loads RAGBrain's ARCHITECTURE.md self-description, then uses an LLM to perform
gap analysis and generate prioritised upgrade recommendations.

LangGraph pipeline:
    fetch_news → load_architecture → gap_analysis → format_report

The report can be posted to Telegram and/or printed to CLI.
"""

from __future__ import annotations

import logging
import ssl
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ragbrain.config import settings

logger = logging.getLogger(__name__)

_ARCHITECTURE_PATH = Path(__file__).resolve().parents[3] / "ARCHITECTURE.md"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ReviewState(TypedDict):
    news_items: list[str]
    architecture_text: str
    recommendations: list[dict[str, Any]]
    report: str
    error: str | None


# ---------------------------------------------------------------------------
# LLM output schemas
# ---------------------------------------------------------------------------

class Recommendation(BaseModel):
    component: str = Field(description="RAGBrain component affected (e.g. 'Chunking', 'Retrieval')")
    current_state: str = Field(description="What RAGBrain currently does for this component")
    news_signal: str = Field(description="The specific news item or research that triggered this")
    suggestion: str = Field(description="Concrete upgrade suggestion")
    priority: str = Field(description="HIGH / MEDIUM / LOW based on impact and feasibility")
    rationale: str = Field(description="Why this upgrade matters")


class ReviewOutput(BaseModel):
    already_covered: list[str] = Field(
        description="News items that RAGBrain already handles well"
    )
    recommendations: list[Recommendation] = Field(
        description="Prioritised list of architecture improvements"
    )
    summary: str = Field(description="One-paragraph executive summary of the review")


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def fetch_news(state: ReviewState) -> dict:
    """Load recent article summaries from RSS feeds (and index into Qdrant)."""
    from ragbrain.pipelines.articles import ArticlesPipeline

    ap = ArticlesPipeline()
    try:
        summaries = ap.run(also_ingest=True)
    finally:
        ap.close()

    news_items = []
    for s in summaries:
        block = f"{s.title}\n{s.key_takeaway}\n{s.summary}"
        if block.strip():
            news_items.append(block[:3000])

    if not news_items:
        return {"news_items": [], "error": "No recent articles found — check RAGBRAIN_RSS_FEEDS_STR."}

    logger.info("Fetched %d article summaries from RSS.", len(news_items))
    return {"news_items": news_items}


def load_architecture(state: ReviewState) -> dict:
    """Load ARCHITECTURE.md as the self-description."""
    if not _ARCHITECTURE_PATH.exists():
        return {"architecture_text": "", "error": "ARCHITECTURE.md not found."}

    text = _ARCHITECTURE_PATH.read_text(encoding="utf-8")
    return {"architecture_text": text}


def gap_analysis(state: ReviewState) -> dict:
    """LLM compares recent AI news against current architecture."""
    news = state.get("news_items", [])
    arch = state.get("architecture_text", "")

    if not news:
        return {"error": "No news items to analyse."}
    if not arch:
        return {"error": "No architecture description loaded."}

    llm = settings.get_llm()

    news_block = "\n\n---\n\n".join(news)

    prompt = f"""You are a senior AI/ML architect reviewing the RAGBrain framework.

Below is RAGBrain's current architecture, followed by recent AI news and research signals.

Your task:
1. Identify which news items RAGBrain ALREADY handles well (no action needed).
2. Identify GAPS — areas where new research, tools, or patterns could improve RAGBrain.
3. For each gap, provide a concrete, actionable recommendation with a priority level.

Focus on:
- New retrieval techniques (ColBERT, late interaction, graph RAG, etc.)
- New embedding models or training approaches
- New chunking strategies
- New agent patterns (multi-agent, tool use, planning)
- Security concerns relevant to the RAG pipeline
- New frameworks or libraries that could replace or augment current components

Be specific. Reference the exact news item that triggered each recommendation.
Don't recommend things that are already implemented.

---

## RAGBrain Architecture (current state):

{arch}

---

## Recent AI News (last 24h):

{news_block}
"""

    structured_llm = llm.with_structured_output(ReviewOutput)
    try:
        result: ReviewOutput = structured_llm.invoke(prompt)
    except Exception:
        logger.exception("Structured output failed, falling back to plain text")
        raw = llm.invoke(prompt)
        return {
            "recommendations": [],
            "report": raw.content if hasattr(raw, "content") else str(raw),
        }

    recs = [r.model_dump() for r in result.recommendations]
    return {"recommendations": recs, "report": ""}


def format_report(state: ReviewState) -> dict:
    """Format the gap analysis into a human-readable report."""
    recs = state.get("recommendations", [])
    error = state.get("error")

    if error and not recs:
        return {"report": f"Architecture review could not complete: {error}"}

    if state.get("report"):
        return {}

    lines: list[str] = []
    now = datetime.now(timezone.utc).strftime("%A, %B %d, %Y")
    lines.append(f"RAGBrain Architecture Review — {now}")
    lines.append("=" * 50)
    lines.append("")

    # Group by priority
    high = [r for r in recs if r.get("priority", "").upper() == "HIGH"]
    medium = [r for r in recs if r.get("priority", "").upper() == "MEDIUM"]
    low = [r for r in recs if r.get("priority", "").upper() == "LOW"]

    for label, group in [("HIGH PRIORITY", high), ("MEDIUM PRIORITY", medium), ("LOW PRIORITY", low)]:
        if not group:
            continue
        lines.append(f"--- {label} ---")
        lines.append("")
        for r in group:
            lines.append(f"  Component:     {r.get('component', '?')}")
            lines.append(f"  Current:       {r.get('current_state', '?')}")
            lines.append(f"  News signal:   {r.get('news_signal', '?')}")
            lines.append(f"  Suggestion:    {r.get('suggestion', '?')}")
            lines.append(f"  Rationale:     {r.get('rationale', '?')}")
            lines.append("")

    if not recs:
        lines.append("No architecture gaps found — RAGBrain is up to date!")

    return {"report": "\n".join(lines)}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_review_graph():
    """Build and compile the architecture review LangGraph."""
    builder = StateGraph(ReviewState)

    builder.add_node("fetch_news", fetch_news)
    builder.add_node("load_architecture", load_architecture)
    builder.add_node("gap_analysis", gap_analysis)
    builder.add_node("format_report", format_report)

    builder.add_edge(START, "fetch_news")
    builder.add_edge("fetch_news", "load_architecture")
    builder.add_edge("load_architecture", "gap_analysis")
    builder.add_edge("gap_analysis", "format_report")
    builder.add_edge("format_report", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def post_to_slack(report: str) -> bool:
    """Post the review report to the configured Slack channel (legacy).

    Returns True on success, False on failure.
    """
    channel = settings.slack_post_channel_id or settings.slack_channel_id
    if not settings.slack_bot_token or not channel:
        logger.warning("Slack not configured — cannot post review.")
        return False

    try:
        import certifi
        from slack_sdk import WebClient

        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        client = WebClient(token=settings.slack_bot_token, ssl=ssl_ctx)
        client.chat_postMessage(channel=channel, text=report)
        logger.info("Posted architecture review to Slack channel %s", channel)
        return True
    except Exception:
        logger.exception("Failed to post to Slack")
        return False


def post_to_telegram(report: str) -> bool:
    """Post the review report to the configured Telegram chat."""
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        logger.warning("Telegram not configured — cannot post review.")
        return False

    try:
        import html as html_lib

        from ragbrain.delivery.telegram import notify_telegram_html_sync

        safe = html_lib.escape(report)
        notify_telegram_html_sync(f"<pre>{safe}</pre>")
        logger.info("Posted architecture review to Telegram.")
        return True
    except Exception:
        logger.exception("Failed to post architecture review to Telegram")
        return False


def run_review(post_slack: bool = False, post_telegram: bool = False) -> str:
    """Run the full architecture review pipeline.

    Args:
        post_slack: If True, also post results to Slack (legacy).
        post_telegram: If True, also post results to Telegram.

    Returns:
        The formatted report string.
    """
    graph = build_review_graph()
    initial_state: ReviewState = {
        "news_items": [],
        "architecture_text": "",
        "recommendations": [],
        "report": "",
        "error": None,
    }
    final = graph.invoke(initial_state)
    report = final.get("report", "No report generated.")

    if post_telegram:
        post_to_telegram(report)
    if post_slack:
        post_to_slack(report)

    return report
