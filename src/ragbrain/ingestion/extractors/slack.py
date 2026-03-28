"""Slack message extractor.

Reads messages from a Slack channel or DM (typically AI news briefings),
parses the structured text, optionally fetches full articles from embedded
URLs via trafilatura, and returns Documents for ingestion.

Requires RAGBRAIN_SLACK_BOT_TOKEN and RAGBRAIN_SLACK_CHANNEL_ID in .env.
The bot must have im:history (for DMs) or channels:history (for channels).
"""

from __future__ import annotations

import logging
import re
import ssl
from datetime import datetime, timedelta, timezone

from ragbrain.config import settings
from ragbrain.models import Block, BlockType, Document, SourceType

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"<(https?://[^|>]+)(?:\|[^>]*)?>")


def _build_client():
    """Build a Slack WebClient with macOS-safe SSL context."""
    import certifi
    from slack_sdk import WebClient

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    return WebClient(token=settings.slack_bot_token, ssl=ssl_ctx)


def _strip_mrkdwn(text: str) -> str:
    """Convert Slack mrkdwn to plain text, replacing <url|label> with label."""
    text = re.sub(r"<(https?://[^|>]+)\|([^>]+)>", r"\2 (\1)", text)
    text = re.sub(r"<(https?://[^>]+)>", r"\1", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return text


class SlackExtractor:
    """Extract documents from Slack channel/DM messages.

    Two extraction modes:

    1. **Briefing mode** (default): treats each message as a self-contained
       document. Good for structured AI news digests like OpenClaw briefings.

    2. **URL mode** (``fetch_urls=True``): also extracts embedded URLs from
       messages and fetches full article text via ``trafilatura``. Each URL
       becomes an additional Document.
    """

    def __init__(self, fetch_urls: bool = False) -> None:
        self._fetch_urls = fetch_urls
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = _build_client()
        return self._client

    def extract_recent(
        self,
        lookback_hours: int | None = None,
        channel_id: str | None = None,
        limit: int = 50,
    ) -> list[Document]:
        """Fetch recent messages and convert them to Documents.

        Args:
            lookback_hours: How far back to look. Defaults to config value.
            channel_id: Override the default channel/DM.
            limit: Max messages to retrieve.

        Returns:
            List of Documents (one per message, plus optionally one per URL).
        """
        hours = lookback_hours or settings.slack_lookback_hours
        channel = channel_id or settings.slack_channel_id

        if not settings.slack_bot_token or not channel:
            logger.warning("Slack bot token or channel ID not configured.")
            return []

        oldest_ts = str((datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp())

        try:
            result = self.client.conversations_history(
                channel=channel, oldest=oldest_ts, limit=limit
            )
        except Exception:
            logger.exception("Failed to read Slack messages")
            return []

        messages = result.get("messages", [])
        if not messages:
            logger.info("No new Slack messages in the last %d hours.", hours)
            return []

        documents: list[Document] = []
        for msg in messages:
            doc = self._message_to_document(msg)
            if doc:
                documents.append(doc)

            if self._fetch_urls:
                url_docs = self._extract_urls(msg)
                documents.extend(url_docs)

        logger.info("Extracted %d documents from %d Slack messages.", len(documents), len(messages))
        return documents

    def _message_to_document(self, msg: dict) -> Document | None:
        """Convert a single Slack message to a Document."""
        text = msg.get("text", "")
        if not text or len(text) < 50:
            return None

        clean = _strip_mrkdwn(text)
        ts = msg.get("ts", "")
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc) if ts else None

        # Try to extract a title from the first bold/header line
        title = "Slack AI Briefing"
        for line in clean.split("\n"):
            stripped = line.strip()
            if stripped and len(stripped) > 10:
                title = stripped[:100]
                break

        return Document(
            source_type=SourceType.RSS,  # reuse RSS type for news-like content
            source_url=f"slack://channel/{settings.slack_channel_id}/{ts}",
            title=title,
            author="Tuk-AI-OpenClaw",
            published_at=dt,
            blocks=[Block(block_type=BlockType.TEXT, content=clean)],
            raw_text=clean,
            metadata={"slack_ts": ts, "source": "slack"},
        )

    def _extract_urls(self, msg: dict) -> list[Document]:
        """Find URLs in a message and fetch full articles."""
        text = msg.get("text", "")
        urls = _URL_RE.findall(text)
        if not urls:
            return []

        from ragbrain.ingestion.extractors.web import WebExtractor
        web = WebExtractor()
        docs: list[Document] = []

        for url in urls:
            if isinstance(url, tuple):
                url = url[0]
            try:
                doc = web.extract(url)
                doc.metadata["discovered_via"] = "slack"
                doc.metadata["slack_ts"] = msg.get("ts", "")
                docs.append(doc)
            except Exception:
                logger.debug("Could not fetch article from %s", url)

        return docs
