"""RSS feed extractor using feedparser.

Fetches all entries from an RSS/Atom feed and returns them as Document objects.
For paywalled content (Substack free-tier), the full article URL is preserved
so the WebExtractor can fetch the full text on demand.
"""

from __future__ import annotations

from datetime import UTC, datetime

import feedparser

from ragbrain.ingestion.extractors.base import BaseExtractor
from ragbrain.models import Block, BlockType, Document, SourceType


class RSSExtractor(BaseExtractor):
    """Fetch and parse RSS/Atom feeds."""

    def can_handle(self, source: str) -> bool:
        # RSS feeds are URLs; we identify them by trying feedparser
        return source.startswith("http://") or source.startswith("https://")

    def extract(self, source: str) -> Document:
        """Extract a single article from its URL using feedparser metadata.

        For bulk feed fetching, use `fetch_feed()` instead.
        """
        feed = feedparser.parse(source)
        if feed.entries:
            entry = feed.entries[0]
            return self._entry_to_document(entry, source)
        return Document(source_type=SourceType.RSS, source_url=source)

    def fetch_feed(
        self, feed_url: str, since: datetime | None = None
    ) -> list[Document]:
        """Fetch all entries from a feed, optionally filtered by date.

        Args:
            feed_url: The RSS/Atom feed URL.
            since: Only return entries published after this timestamp (UTC).

        Returns:
            List of Document objects, one per feed entry.
        """
        feed = feedparser.parse(feed_url)
        documents: list[Document] = []

        for entry in feed.entries:
            doc = self._entry_to_document(entry, feed_url)
            if since and doc.published_at:
                pub = doc.published_at
                if pub.tzinfo is None:
                    pub = pub.replace(tzinfo=UTC)
                since_aware = since.replace(tzinfo=UTC) if since.tzinfo is None else since
                if pub < since_aware:
                    continue
            documents.append(doc)

        return documents

    def _entry_to_document(self, entry: feedparser.FeedParserDict, feed_url: str) -> Document:
        title = getattr(entry, "title", "")
        link = getattr(entry, "link", "")
        author = getattr(entry, "author", "")
        summary = getattr(entry, "summary", "") or getattr(entry, "description", "")

        published_at: datetime | None = None
        published_parsed = getattr(entry, "published_parsed", None)
        if published_parsed:
            try:
                published_at = datetime(*published_parsed[:6], tzinfo=UTC)
            except (TypeError, ValueError):
                pass

        # Strip HTML from summary
        import html as html_module
        import re

        clean_summary = re.sub(r"<[^>]+>", " ", summary)
        clean_summary = html_module.unescape(clean_summary).strip()

        blocks = [Block(block_type=BlockType.TEXT, content=clean_summary)] if clean_summary else []

        return Document(
            source_type=SourceType.RSS,
            source_url=link or feed_url,
            title=title,
            author=author,
            published_at=published_at,
            blocks=blocks,
            raw_text=clean_summary,
            metadata={"feed_url": feed_url},
        )
