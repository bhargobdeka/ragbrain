"""Lightweight query intent detector for source-type routing.

Analyses the query with keyword rules and returns an optional SourceType hint
that the retriever uses to pre-filter Qdrant results.  This avoids the costly
CRAG retry loop that occurs when the first retrieval pulls the wrong source
(e.g. book chunks when the user is asking about a Slack briefing).

No LLM is used — pattern matching is fast, free, and deterministic.
"""

from __future__ import annotations

import re

from ragbrain.models import SourceType

# ---- Keyword sets per source type ----------------------------------------

# Slack / news briefing signals
_SLACK_EXACT: set[str] = {
    "tuk", "briefing", "slack", "newsletter",
    "announcement", "update", "roundup",
}
_SLACK_PHRASES: list[str] = [
    "today's briefing", "yesterday's briefing", "daily briefing",
    "daily ai", "weekly briefing", "news briefing", "ai news",
    "tuk's", "tuk said", "tuk mentioned",
]

# Book / PDF signals
_BOOK_EXACT: set[str] = {
    "book", "chapter", "page", "pdf", "rlhf",
    "textbook", "section", "meap",
}
_BOOK_PHRASES: list[str] = [
    "according to the book", "in the book", "the rlhf book",
]

# Web article / RSS signals
_WEB_EXACT: set[str] = {
    "article", "blog", "post", "website", "webpage",
    "paper", "arxiv", "research", "publication",
}
_WEB_PHRASES: list[str] = [
    "from the web", "online article", "blog post",
]

# Code signals
_CODE_EXACT: set[str] = {
    "function", "class", "method", "module", "snippet",
    "code", "implementation", "codebase", "script",
}

# ---- Detector ------------------------------------------------------------


def detect_source_intent(query: str) -> SourceType | None:
    """Return a SourceType filter hint inferred from the query, or None.

    None means "search everything" — the caller should not apply any filter.

    Examples::

        detect_source_intent("What did Tuk say in today's briefing?")
        # → SourceType.SLACK

        detect_source_intent("What chapter covers reward models?")
        # → SourceType.PDF

        detect_source_intent("What is RLHF?")
        # → None  (general — search all sources)
    """
    q = query.lower()
    words = set(re.findall(r"\b\w+\b", q))

    # Check Slack signals first (highest priority — very specific)
    if words & _SLACK_EXACT or any(p in q for p in _SLACK_PHRASES):
        return SourceType.SLACK

    # Book/PDF signals
    if words & _BOOK_EXACT or any(p in q for p in _BOOK_PHRASES):
        return SourceType.PDF

    # Web/article signals
    if words & _WEB_EXACT or any(p in q for p in _WEB_PHRASES):
        return SourceType.WEB

    # Code signals
    if words & _CODE_EXACT:
        return SourceType.CODE

    return None
