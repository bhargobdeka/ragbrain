"""Slack delivery — posting briefings and proposals, polling for approvals.

Works out of the box with just the existing RAGBRAIN_SLACK_BOT_TOKEN
and RAGBRAIN_SLACK_CHANNEL_ID.

Proposal approval flow (message-based, no extra setup needed):
  1. Post proposal message to the DM channel with instructions.
  2. Background poller scans for reply messages matching:
       "approve <id>", "skip <id>", "explain <id>"
  3. On match: updates ProposalStore, triggers AutoImplementer for approvals,
     posts result back to Slack.

Optional — richer Block Kit buttons (requires RAGBRAIN_SLACK_APP_TOKEN):
  If RAGBRAIN_SLACK_APP_TOKEN is set, proposals are posted as interactive
  Block Kit messages with Approve / Skip / Explain buttons.
  Enable Socket Mode in your Slack App settings and generate an App-level
  token (xapp-...) to use this mode.
"""

from __future__ import annotations

import logging
import re
import ssl
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ragbrain.config import settings

if TYPE_CHECKING:
    from ragbrain.pipelines.proposals import Proposal

logger = logging.getLogger(__name__)

# Matches: "approve abc1", "skip abc1", "explain abc1" (case-insensitive)
_APPROVAL_RE = re.compile(
    r"^\s*(approve|skip|explain)\s+([a-f0-9]{6,8})\s*$", re.IGNORECASE
)


# ---- Slack client factory ------------------------------------------------

def _build_client():
    """Return a Slack WebClient with macOS-safe SSL context."""
    import certifi
    from slack_sdk import WebClient

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    return WebClient(token=settings.slack_bot_token, ssl=ssl_ctx)


def _channel() -> str:
    return settings.slack_post_channel_id or settings.slack_channel_id


# ---- Posting helpers -----------------------------------------------------

def post_message(text: str) -> bool:
    """Post a plain-text or mrkdwn message to the configured Slack channel."""
    if not settings.slack_bot_token or not _channel():
        logger.warning("Slack not configured — skipping post.")
        return False
    try:
        client = _build_client()
        client.chat_postMessage(channel=_channel(), text=text, mrkdwn=True)
        return True
    except Exception:
        logger.exception("Failed to post to Slack")
        return False


def post_proposal(proposal: "Proposal") -> bool:
    """Post a proposal to Slack.

    If RAGBRAIN_SLACK_APP_TOKEN is set, uses interactive Block Kit buttons.
    Otherwise, posts a text message with reply instructions.
    """
    if not settings.slack_bot_token or not _channel():
        logger.warning("Slack not configured — cannot post proposal.")
        return False

    # Check for Socket Mode app token → use Block Kit buttons
    app_token = getattr(settings, "slack_app_token", "")
    if app_token:
        return _post_proposal_block_kit(proposal)
    else:
        return _post_proposal_text(proposal)


def _post_proposal_text(proposal: "Proposal") -> bool:
    """Post proposal as text with reply instructions (no app token needed)."""
    lines = [
        f"*[{proposal.priority}] {proposal.title}*",
        f"_Component:_ {proposal.component}",
        "",
        f"*Why:* {proposal.description}",
    ]
    if proposal.news_signal:
        lines += ["", f"*Triggered by:* {proposal.news_signal[:300]}"]
    lines += [
        "",
        f"*Implementation plan:*",
        proposal.implementation_plan[:500],
        "",
        f"━━━━━━━━━━━━━━━━━━━━━━",
        f"Reply with one of:",
        f"  • `approve {proposal.id}` — implement this change",
        f"  • `skip {proposal.id}` — skip this proposal",
        f"  • `explain {proposal.id}` — show full details",
    ]
    try:
        client = _build_client()
        client.chat_postMessage(
            channel=_channel(),
            text="\n".join(lines),
            mrkdwn=True,
        )
        return True
    except Exception:
        logger.exception("Failed to post proposal to Slack")
        return False


def _post_proposal_block_kit(proposal: "Proposal") -> bool:
    """Post proposal as interactive Block Kit message (requires app token)."""
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*[{proposal.priority}] {proposal.title}*\n"
                    f"_Component:_ {proposal.component}\n\n"
                    f"*Why:* {proposal.description[:300]}"
                ),
            },
        },
        {"type": "divider"},
        {
            "type": "actions",
            "block_id": f"proposal_{proposal.id}",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✅ Approve"},
                    "style": "primary",
                    "value": f"approve:{proposal.id}",
                    "action_id": f"approve_{proposal.id}",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "⏭ Skip"},
                    "value": f"skip:{proposal.id}",
                    "action_id": f"skip_{proposal.id}",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "🔍 Explain"},
                    "value": f"explain:{proposal.id}",
                    "action_id": f"explain_{proposal.id}",
                },
            ],
        },
    ]
    try:
        client = _build_client()
        client.chat_postMessage(
            channel=_channel(),
            text=f"Proposal: {proposal.title}",
            blocks=blocks,
        )
        return True
    except Exception:
        logger.exception("Failed to post Block Kit proposal to Slack")
        return False


def post_briefing(briefing_html: str) -> bool:
    """Post a daily briefing to Slack. Converts HTML tags to mrkdwn."""
    # Convert simple HTML to Slack mrkdwn
    text = briefing_html
    text = re.sub(r"<b>(.*?)</b>", r"*\1*", text, flags=re.DOTALL)
    text = re.sub(r"<i>(.*?)</i>", r"_\1_", text, flags=re.DOTALL)
    text = re.sub(r"<code>(.*?)</code>", r"`\1`", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)   # strip any remaining HTML
    return post_message(text)


# ---- Approval poller -----------------------------------------------------

_SEEN_TS: set[str] = set()   # timestamps already processed this session


def poll_and_process_approvals() -> int:
    """Scan recent DM messages for approve/skip/explain commands.

    Processes each new matching message once and returns the count handled.
    Call this in a loop to continuously watch for approvals.
    """
    if not settings.slack_bot_token or not _channel():
        return 0

    from ragbrain.pipelines.proposals import get_store

    processed = 0
    try:
        client = _build_client()
        resp = client.conversations_history(channel=_channel(), limit=20)
        messages = resp.get("messages", [])
    except Exception:
        logger.exception("Failed to fetch Slack history for approval polling")
        return 0

    for msg in messages:
        ts = msg.get("ts", "")
        text = (msg.get("text") or "").strip()

        # Skip already-processed, bot messages, and empty
        if not text or ts in _SEEN_TS:
            continue
        # Skip messages sent by the bot itself
        if msg.get("bot_id") or msg.get("subtype") == "bot_message":
            _SEEN_TS.add(ts)
            continue

        m = _APPROVAL_RE.match(text)
        if not m:
            continue

        action = m.group(1).lower()
        proposal_id = m.group(2).lower()
        _SEEN_TS.add(ts)

        store = get_store()
        proposal = store.get(proposal_id)

        if proposal is None:
            post_message(f"⚠️ Proposal `{proposal_id}` not found. Check `~/.ragbrain/proposals.json`.")
            continue

        logger.info("Slack approval action '%s' on proposal %s", action, proposal_id)

        if action == "skip":
            store.skip(proposal_id)
            post_message(f"⏭ Skipped: *{proposal.title}*")

        elif action == "explain":
            lines = [
                f"*{proposal.title}*",
                f"_Component:_ {proposal.component}  |  _Priority:_ {proposal.priority}",
                "",
                f"*Description:* {proposal.description}",
                "",
                f"*Implementation plan:*",
                proposal.implementation_plan,
            ]
            if proposal.news_signal:
                lines += ["", f"*News signal:* {proposal.news_signal}"]
            lines += [
                "",
                f"Reply `approve {proposal.id}` to implement, or `skip {proposal.id}` to skip.",
            ]
            post_message("\n".join(lines))

        elif action == "approve":
            store.approve(proposal_id)
            post_message(
                f"✅ Approved: *{proposal.title}*\n"
                f"⏳ Starting auto-implementation on your Mac…"
            )
            _run_auto_implement(proposal)

        processed += 1

    return processed


def _run_auto_implement(proposal: "Proposal") -> None:
    """Run AutoImplementer in a background thread and post result to Slack."""
    import threading

    def _impl():
        try:
            from ragbrain.pipelines.auto_implement import implement_proposal
            from ragbrain.pipelines.proposals import get_store

            result = implement_proposal(proposal)
            store = get_store()

            if result.success:
                store.mark_implemented(
                    proposal.id,
                    commit_sha=result.commit_sha,
                    summary=result.summary,
                )
                msg = (
                    f"✅ *Implemented & committed*\n"
                    f"_{proposal.title}_\n"
                    f"File: `{result.target_file}`\n"
                    f"Eval: passed  |  Commit: `{result.commit_sha}`"
                )
            else:
                store.mark_failed(proposal.id, reason=result.summary)
                msg = (
                    f"⚠️ *Auto-implementation reverted*\n"
                    f"_{proposal.title}_\n"
                    f"{result.summary}"
                )

            # Strip HTML from the telegram_message for Slack
            slack_msg = re.sub(r"<[^>]+>", "", result.telegram_message)
            post_message(slack_msg)
        except Exception:
            logger.exception("AutoImplementer thread failed")
            post_message(f"❌ AutoImplementer crashed for proposal `{proposal.id}`. Check logs.")

    threading.Thread(target=_impl, daemon=True).start()


# ---- Blocking approval loop (for serve-slack) ----------------------------

def run_approval_loop(poll_interval: int = 15) -> None:
    """Block forever, polling for approval messages every `poll_interval` seconds.

    This is the main loop for `ragbrain serve-slack`.
    Ctrl+C to stop.
    """
    logger.info(
        "Slack approval poller started — watching %s every %ds",
        _channel(), poll_interval,
    )
    print(f"  Watching Slack channel {_channel()} for approvals every {poll_interval}s.")
    print("  Reply to any proposal with:  approve <id>  |  skip <id>  |  explain <id>")
    print("  Press Ctrl+C to stop.\n")

    while True:
        try:
            n = poll_and_process_approvals()
            if n:
                logger.info("Processed %d approval action(s) from Slack", n)
        except Exception:
            logger.exception("Approval loop iteration failed — continuing")
        time.sleep(poll_interval)
