"""X/Twitter thread draft generator.

Takes a daily AI briefing and produces a concise tweet thread (3-5 tweets,
each ≤280 chars) suitable for copy-pasting into X.  Uses the fast LLM
(Haiku) since this is a straightforward text-transformation task.
"""

from __future__ import annotations

import logging
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage

from ragbrain.config import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a senior AI engineer who shares daily AI news on X/Twitter.
Convert the provided AI briefing into a punchy tweet thread.

Rules:
- Output 3 to 5 tweets, separated by a line containing only "---".
- Each tweet MUST be ≤ 280 characters (hard limit — count carefully).
- Tweet 1 is the hook: summarise the day's biggest story in one sentence,
  add a 🧵 emoji at the end.
- Middle tweets: one key story each, use bullet-style conciseness.
- Final tweet: a one-line personal takeaway or call-to-action.
- Use plain text. Hashtags sparingly (max 2 per tweet). No links.
- Write in first person. Tone: sharp, opinionated, no fluff.
- Output ONLY the tweets separated by "---". No numbering, no preamble."""

_HUMAN_PROMPT = "Here is today's AI briefing:\n\n{briefing}"


def generate_x_thread(briefing_text: str) -> list[str]:
    """Generate a list of tweet strings from a daily briefing."""
    llm = settings.get_fast_llm()
    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_HUMAN_PROMPT.format(briefing=briefing_text)),
    ])
    raw = response.content.strip()
    tweets = [t.strip() for t in raw.split("---") if t.strip()]

    over = [i for i, t in enumerate(tweets, 1) if len(t) > 280]
    if over:
        logger.warning("Tweets exceeding 280 chars: %s — you may want to trim", over)

    return tweets


def format_draft_md(tweets: list[str], date_str: str | None = None) -> str:
    """Format tweets into a human-readable markdown draft file."""
    date_str = date_str or date.today().isoformat()
    lines = [f"# X Thread Draft — {date_str}", ""]
    for i, tweet in enumerate(tweets, 1):
        lines.append(f"## Tweet {i}  ({len(tweet)}/280 chars)")
        lines.append("")
        lines.append(tweet)
        lines.append("")
    lines.append("---")
    lines.append("*Edit freely, then copy-paste each tweet into X.*")
    lines.append("")
    return "\n".join(lines)
