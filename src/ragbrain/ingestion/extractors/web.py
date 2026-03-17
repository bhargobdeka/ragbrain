"""Web article extractor using trafilatura.

Handles Substack, Medium, and general blog pages. Detects inline code
blocks by looking for <code>/<pre> tags in the original HTML before trafilatura
strips them, then re-labels those segments in the extracted text.
"""

from __future__ import annotations

import re
from datetime import datetime

import httpx
import trafilatura
from trafilatura.settings import use_config

from ragbrain.ingestion.extractors.base import BaseExtractor
from ragbrain.models import Block, BlockType, Document, SourceType

# Pattern to detect code fences in markdown-like output
_CODE_FENCE = re.compile(r"```[\s\S]+?```", re.MULTILINE)
_INLINE_CODE = re.compile(r"`[^`\n]+`")

_TRAFILATURA_CONFIG = use_config()
_TRAFILATURA_CONFIG.set("DEFAULT", "EXTRACTION_TIMEOUT", "30")


class WebExtractor(BaseExtractor):
    """Extract full-text articles from web URLs using trafilatura."""

    def __init__(self, timeout: int = 30) -> None:
        self._timeout = timeout

    def can_handle(self, source: str) -> bool:
        return source.startswith("http://") or source.startswith("https://")

    def extract(self, source: str) -> Document:
        html = self._fetch_html(source)
        return self._parse_html(html, source)

    def _fetch_html(self, url: str) -> str:
        response = httpx.get(url, timeout=self._timeout, follow_redirects=True)
        response.raise_for_status()
        return response.text

    def _parse_html(self, html: str, url: str) -> Document:
        metadata = trafilatura.extract_metadata(html)
        text = trafilatura.extract(
            html,
            config=_TRAFILATURA_CONFIG,
            include_comments=False,
            include_tables=True,
            output_format="markdown",
        ) or ""

        title = (metadata.title if metadata else "") or ""
        author = (metadata.author if metadata else "") or ""
        date_str = (metadata.date if metadata else "") or ""

        published_at: datetime | None = None
        if date_str:
            try:
                published_at = datetime.fromisoformat(date_str)
            except ValueError:
                pass

        blocks = self._segment_blocks(text)
        raw_text = "\n\n".join(b.content for b in blocks if b.block_type == BlockType.TEXT)

        return Document(
            source_type=SourceType.WEB,
            source_url=url,
            title=title,
            author=author,
            published_at=published_at,
            blocks=blocks,
            raw_text=raw_text,
        )

    def _segment_blocks(self, markdown_text: str) -> list[Block]:
        """Split markdown text into text and code blocks."""
        if not markdown_text:
            return []

        blocks: list[Block] = []
        last_end = 0

        for match in _CODE_FENCE.finditer(markdown_text):
            # Text before this code fence
            pre_text = markdown_text[last_end : match.start()].strip()
            if pre_text:
                blocks.append(Block(block_type=BlockType.TEXT, content=pre_text))

            code_content = match.group(0)
            # Strip the ``` markers
            inner = re.sub(r"^```\w*\n?", "", code_content)
            inner = re.sub(r"\n?```$", "", inner).strip()
            lang = re.match(r"^```(\w+)", code_content)
            blocks.append(
                Block(
                    block_type=BlockType.CODE,
                    content=inner,
                    language=lang.group(1) if lang else None,
                )
            )
            last_end = match.end()

        # Remaining text after last code fence
        remaining = markdown_text[last_end:].strip()
        if remaining:
            blocks.append(Block(block_type=BlockType.TEXT, content=remaining))

        return blocks or [Block(block_type=BlockType.TEXT, content=markdown_text.strip())]
