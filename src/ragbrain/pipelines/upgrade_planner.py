"""Upgrade Planner — Deep Agents-powered self-improving architecture loop.

Architecture:
    A Deep Agents orchestrator (claude-sonnet-4-6) is given four tools:
        read_architecture()         → loads ARCHITECTURE.md
        read_architecture_state()   → loads persistent upgrade history
        fetch_slack_news()          → recent AI news from Slack
        search_knowledge_base()     → queries our own Qdrant RAG pipeline

    The agent autonomously:
      1. Gathers current architecture + history
      2. Pulls recent Slack news
      3. Searches the KB for relevant research already indexed
      4. Plans upgrades (HIGH / MEDIUM / LOW), using its built-in write_todos
         to track sub-tasks within the session
      5. Returns a structured UpgradePlan (Pydantic)

    After each run the plan is appended to architecture-state.md so future
    runs remember what was already discussed.

Usage:
    ragbrain plan-upgrades [--post-slack]
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from langchain.tools import tool
from pydantic import BaseModel, Field

from ragbrain.config import settings

logger = logging.getLogger(__name__)

_ARCH_PATH  = Path(__file__).resolve().parents[3] / "ARCHITECTURE.md"
_STATE_PATH = Path(__file__).resolve().parents[3] / "architecture-state.md"


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class UpgradeRecommendation(BaseModel):
    component: str  = Field(description="RAGBrain component affected (e.g. Retrieval, Chunking, Agents)")
    suggestion: str = Field(description="Concrete, actionable upgrade suggestion")
    priority: str   = Field(description="HIGH / MEDIUM / LOW based on impact and feasibility")
    effort: str     = Field(description="Estimated effort: DAYS / WEEKS / MONTHS")
    news_signal: str = Field(description="The exact news item or paper that triggered this")
    rationale: str  = Field(description="Why this upgrade matters for RAGBrain")


class UpgradePlan(BaseModel):
    """Structured upgrade plan returned by the planner agent."""
    summary: str = Field(description="One-paragraph executive summary of the analysis")
    recommendations: list[UpgradeRecommendation] = Field(
        description="Prioritised list of concrete upgrade recommendations"
    )
    already_covered: list[str] = Field(
        description="News items / techniques RAGBrain already handles well"
    )
    deferred: list[str] = Field(
        default_factory=list,
        description="Items from previous runs that are still pending"
    )


# ---------------------------------------------------------------------------
# Tools for the Deep Agent
# ---------------------------------------------------------------------------

@tool
def read_architecture() -> str:
    """Read RAGBrain's current architecture from ARCHITECTURE.md.

    Returns the full architecture description including chunking strategies,
    embedding models, retrieval methods, agent patterns, and known limitations.
    """
    if not _ARCH_PATH.exists():
        return "ARCHITECTURE.md not found — file may have been moved."
    return _ARCH_PATH.read_text(encoding="utf-8")


@tool
def read_architecture_state() -> str:
    """Read the persistent upgrade history from architecture-state.md.

    Returns all previous upgrade plans so you can track what was recommended
    in prior runs and avoid duplicating already-planned work.
    """
    if not _STATE_PATH.exists():
        return "No previous upgrade history found — this is the first run."
    return _STATE_PATH.read_text(encoding="utf-8")


@tool
def fetch_slack_news(lookback_hours: int = 24) -> str:
    """Fetch recent AI news from the configured Slack channel.

    Args:
        lookback_hours: How many hours back to look (default 24).

    Returns:
        Formatted list of news items with titles and excerpts.
    """
    try:
        from ragbrain.ingestion.extractors.slack import SlackExtractor

        extractor = SlackExtractor(fetch_urls=False)
        docs = extractor.extract_recent(lookback_hours=lookback_hours)
    except Exception as e:
        return f"Could not fetch Slack news: {e}"

    if not docs:
        return f"No messages found in the last {lookback_hours} hours."

    items = []
    for i, doc in enumerate(docs, 1):
        title = doc.title or "Untitled"
        excerpt = (doc.raw_text or "")[:600].replace("\n", " ")
        items.append(f"{i}. **{title}**\n   {excerpt}")

    return f"Found {len(docs)} news items in the last {lookback_hours}h:\n\n" + "\n\n".join(items)


@tool
def search_knowledge_base(query: str) -> str:
    """Search RAGBrain's indexed knowledge base (papers, articles, books).

    Use this to check if a technique mentioned in the news is already
    indexed, understand it better, or find related prior work.

    Args:
        query: Natural language search query.

    Returns:
        Answer synthesised from the knowledge base plus source citations.
    """
    try:
        from ragbrain.agents.graph import query as rag_query

        result = rag_query(query)
    except Exception as e:
        return f"Knowledge base search failed: {e}"

    answer  = result.get("answer", "").strip()
    sources = result.get("sources", [])

    if not answer:
        return "No relevant content found in the knowledge base for this query."

    out = [f"Answer: {answer}"]
    if sources:
        out.append("\nSources:")
        for s in sources:
            title = s.get("title", "Unknown")
            url   = s.get("url", "")
            out.append(f"  • {title}" + (f"  {url}" if url else ""))

    return "\n".join(out)


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def _append_to_state_file(plan: UpgradePlan, date_str: str) -> None:
    """Append the latest plan to architecture-state.md for cross-run memory."""
    section_lines = [
        f"\n## Run: {date_str}\n",
        f"### Summary\n{plan.summary}\n",
        "### Recommendations\n",
    ]
    for r in plan.recommendations:
        section_lines.append(
            f"- [{r.priority}] **{r.component}**: {r.suggestion} *(Effort: {r.effort})*\n"
        )

    if plan.already_covered:
        section_lines.append("\n### Already Covered\n")
        for item in plan.already_covered:
            section_lines.append(f"- ✓ {item}\n")

    section_lines.append("\n---\n")

    if _STATE_PATH.exists():
        existing = _STATE_PATH.read_text(encoding="utf-8")
    else:
        existing = (
            "# RAGBrain Architecture State\n\n"
            "This file is auto-maintained by `ragbrain plan-upgrades`.\n"
            "It tracks upgrade plans across runs so the planner can build on prior work.\n\n"
            "---\n"
        )

    _STATE_PATH.write_text(existing + "".join(section_lines), encoding="utf-8")
    logger.info("Appended upgrade plan to %s", _STATE_PATH)


# ---------------------------------------------------------------------------
# Format report (used when structured output falls back to text)
# ---------------------------------------------------------------------------

def _format_plan(plan: UpgradePlan, date_str: str) -> str:
    lines = [
        f"RAGBrain Upgrade Plan — {date_str}",
        "=" * 55,
        "",
        f"Summary: {plan.summary}",
        "",
    ]

    high   = [r for r in plan.recommendations if r.priority.upper() == "HIGH"]
    medium = [r for r in plan.recommendations if r.priority.upper() == "MEDIUM"]
    low    = [r for r in plan.recommendations if r.priority.upper() == "LOW"]

    for label, group in [("HIGH PRIORITY", high), ("MEDIUM PRIORITY", medium), ("LOW PRIORITY", low)]:
        if not group:
            continue
        lines += [f"--- {label} ---", ""]
        for r in group:
            lines += [
                f"  Component:    {r.component}",
                f"  Suggestion:   {r.suggestion}",
                f"  Effort:       {r.effort}",
                f"  News signal:  {r.news_signal}",
                f"  Rationale:    {r.rationale}",
                "",
            ]

    if plan.already_covered:
        lines += ["--- ALREADY COVERED ---", ""]
        for item in plan.already_covered:
            lines.append(f"  ✓ {item}")
        lines.append("")

    if plan.deferred:
        lines += ["--- DEFERRED FROM PRIOR RUNS ---", ""]
        for item in plan.deferred:
            lines.append(f"  ⏳ {item}")
        lines.append("")

    if not plan.recommendations:
        lines.append("No architecture gaps found — RAGBrain is up to date!")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_upgrade_planner(post_slack: bool = False) -> str:
    """Run the Deep Agents upgrade planner.

    The agent gathers news, searches the KB, and produces a prioritised
    UpgradePlan. Results are persisted to architecture-state.md.

    Args:
        post_slack: If True, also post the report to Slack.

    Returns:
        Formatted report string.
    """
    from deepagents import create_deep_agent
    from langgraph.checkpoint.memory import MemorySaver

    date_str = datetime.now(timezone.utc).strftime("%A, %B %d, %Y")

    system_prompt = f"""\
You are RAGBrain's architecture upgrade planner — a senior AI/ML architect
whose job is to keep RAGBrain at the state of the art.

Your workflow for EVERY run:
1. Call read_architecture() to understand what RAGBrain currently implements.
2. Call read_architecture_state() to see what was planned in prior runs.
3. Call fetch_slack_news() to get today's AI research signals.
4. For each interesting news item, call search_knowledge_base() to check if
   RAGBrain's indexed documents already contain relevant papers or context.
5. Synthesise everything into a prioritised upgrade plan.

Prioritisation rules:
  HIGH   — Proven technique, direct applicability, low migration risk.
  MEDIUM — Promising approach, worth a prototype/spike.
  LOW    — Research-stage, monitor for now.

Do NOT recommend things already implemented in ARCHITECTURE.md.
Do NOT repeat recommendations from prior runs unless they are still pending.
Reference the exact news item or paper for every recommendation.

Today's date: {date_str}
"""

    checkpointer = MemorySaver()

    agent = create_deep_agent(
        model=settings.get_llm(),
        tools=[
            read_architecture,
            read_architecture_state,
            fetch_slack_news,
            # search_knowledge_base intentionally excluded: it opens Qdrant which
            # is already held by the main process, causing "already accessed" hangs.
            # The planner gets sufficient signal from Slack news + architecture docs.
        ],
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        response_format=UpgradePlan,
    )

    thread_id = f"upgrade-plan-{datetime.now(timezone.utc).strftime('%Y%m%d')}"

    result = agent.invoke(
        {
            "messages": [{
                "role": "user",
                "content": (
                    "Please analyse recent AI news and compare it against RAGBrain's "
                    "architecture. Use read_architecture, read_architecture_state, and "
                    "fetch_slack_news to gather information, then produce a prioritised "
                    "upgrade plan."
                ),
            }]
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    plan: UpgradePlan | None = result.get("structured_response")

    if plan:
        report = _format_plan(plan, date_str)
        _append_to_state_file(plan, date_str)
    else:
        # Fallback: extract last assistant message
        messages = result.get("messages", [])
        last_content = ""
        for msg in reversed(messages):
            content = getattr(msg, "content", None) or str(msg)
            if content and content.strip():
                last_content = content
                break
        report = f"RAGBrain Upgrade Plan — {date_str}\n{'=' * 55}\n\n{last_content}"

    if post_slack:
        try:
            from ragbrain.pipelines.architecture_review import post_to_slack
            post_to_slack(report)
        except Exception:
            logger.exception("Failed to post upgrade plan to Slack")

    return report


def get_upgrade_recommendations() -> list[dict]:
    """Run the upgrade planner and return structured recommendations as dicts.

    Each dict has keys matching UpgradeRecommendation fields:
        component, suggestion, priority, effort, news_signal, rationale

    Returns an empty list if planning fails.  Suitable for the scheduler's
    daily_automation_job which needs structured data to create Proposals.
    """
    from deepagents import create_deep_agent
    from langgraph.checkpoint.memory import MemorySaver

    date_str = datetime.now(timezone.utc).strftime("%A, %B %d, %Y")

    system_prompt = f"""\
You are RAGBrain's architecture upgrade planner.
Use read_architecture, read_architecture_state, and fetch_slack_news to gather
information, then return a prioritised UpgradePlan.
Today's date: {date_str}
"""
    checkpointer = MemorySaver()
    agent = create_deep_agent(
        model=settings.get_llm(),
        # search_knowledge_base excluded — it opens Qdrant which is already
        # held by the main process, causing "already accessed" deadlocks.
        tools=[read_architecture, read_architecture_state, fetch_slack_news],
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        response_format=UpgradePlan,
    )
    thread_id = f"upgrade-recs-{datetime.now(timezone.utc).strftime('%Y%m%d')}"

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Analyse and return a prioritised upgrade plan."}]},
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception:
        logger.exception("get_upgrade_recommendations: agent invocation failed")
        return []

    plan: UpgradePlan | None = result.get("structured_response")
    if plan is None:
        return []

    _append_to_state_file(plan, date_str)

    return [rec.model_dump() for rec in plan.recommendations]
