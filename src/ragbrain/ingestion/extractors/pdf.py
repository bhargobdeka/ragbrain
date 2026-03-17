"""PDF extractor using PyMuPDF (free).

Detects text blocks, code blocks (monospace fonts), and images (captioned via
a VLM when an Anthropic key is available).

Premium upgrade path: replace this with LandingAI ADE or LLMWhisperer by
subclassing BaseExtractor and dropping it into the extractor registry.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF

from ragbrain.ingestion.extractors.base import BaseExtractor
from ragbrain.models import Block, BlockType, Document, SourceType

# Heuristic: font names that suggest monospace / code
_CODE_FONT_PATTERNS = re.compile(
    r"(Courier|Mono|Code|Consol|Inconsolata|Source.?Code|Fira.?Mono)",
    re.IGNORECASE,
)

# Minimum characters for a block to be worth keeping
_MIN_BLOCK_CHARS = 30


class PDFExtractor(BaseExtractor):
    """Extract text and code blocks from PDF files using PyMuPDF."""

    def __init__(self, caption_images: bool = False) -> None:
        # Caption images requires an LLM call (optional, off by default)
        self._caption_images = caption_images

    def can_handle(self, source: str) -> bool:
        return Path(source).suffix.lower() == ".pdf"

    def extract(self, source: str) -> Document:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {source}")

        doc_pdf = fitz.open(str(path))
        title = doc_pdf.metadata.get("title", "") or path.stem
        author = doc_pdf.metadata.get("author", "")

        blocks: list[Block] = []

        for page_num, page in enumerate(doc_pdf, start=1):
            # Extract text blocks with font info
            blocks.extend(self._extract_text_blocks(page, page_num))

            # Extract images and optionally caption them
            if self._caption_images:
                blocks.extend(self._extract_image_blocks(page, page_num))

        doc_pdf.close()

        raw_text = "\n\n".join(b.content for b in blocks if b.block_type == BlockType.TEXT)

        return Document(
            source_type=SourceType.PDF,
            source_url=str(path.resolve()),
            title=title,
            author=author,
            blocks=blocks,
            raw_text=raw_text,
            metadata={"file_name": path.name, "page_count": page_num},
        )

    def _extract_text_blocks(self, page: fitz.Page, page_num: int) -> list[Block]:
        blocks: list[Block] = []
        # dict format gives font-level detail
        page_dict = page.get_text("dict")

        for raw_block in page_dict.get("blocks", []):
            if raw_block.get("type") != 0:  # 0 = text block
                continue

            lines_text: list[str] = []
            is_code_block = False

            for line in raw_block.get("lines", []):
                for span in line.get("spans", []):
                    font_name: str = span.get("font", "")
                    if _CODE_FONT_PATTERNS.search(font_name):
                        is_code_block = True
                    text = span.get("text", "").strip()
                    if text:
                        lines_text.append(text)

            content = " ".join(lines_text).strip()
            if len(content) < _MIN_BLOCK_CHARS:
                continue

            bbox_raw = raw_block.get("bbox", (0, 0, 0, 0))
            blocks.append(
                Block(
                    block_type=BlockType.CODE if is_code_block else BlockType.TEXT,
                    content=content,
                    page_number=page_num,
                    bbox={"x0": bbox_raw[0], "y0": bbox_raw[1], "x1": bbox_raw[2], "y1": bbox_raw[3]},
                )
            )

        return blocks

    def _extract_image_blocks(self, page: fitz.Page, page_num: int) -> list[Block]:
        """Placeholder: returns empty list unless VLM captioning is wired up."""
        # To enable: extract image bytes, send to Claude vision, return caption Block
        return []
