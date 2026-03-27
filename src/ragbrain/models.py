"""Core data models shared across the entire application."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class BlockType(StrEnum):
    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    TABLE = "table"


class SourceType(StrEnum):
    PDF = "pdf"
    WEB = "web"
    RSS = "rss"
    UNKNOWN = "unknown"


class Block(BaseModel):
    """A single content block extracted from a raw document (before chunking)."""

    block_id: str = Field(default_factory=lambda: str(uuid4()))
    block_type: BlockType = BlockType.TEXT
    content: str
    language: str | None = None        # for code blocks
    page_number: int | None = None     # for PDFs
    bbox: dict[str, float] | None = None   # bounding box {x0, y0, x1, y1}
    caption: str | None = None         # for image blocks (VLM-generated)


class Document(BaseModel):
    """A raw document before chunking -- one per ingest call."""

    doc_id: str = Field(default_factory=lambda: str(uuid4()))
    source_type: SourceType = SourceType.UNKNOWN
    source_url: str = ""
    title: str = ""
    author: str = ""
    published_at: datetime | None = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    blocks: list[Block] = Field(default_factory=list)
    raw_text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A text chunk ready for embedding and storage in Qdrant."""

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    doc_id: str
    content: str
    block_type: BlockType = BlockType.TEXT
    source_type: SourceType = SourceType.UNKNOWN
    source_url: str = ""
    title: str = ""
    page_number: int | None = None
    chunk_index: int = 0
    # Code-specific structural metadata (populated by ASTCodeChunker)
    language: str | None = None          # e.g. "python", "typescript"
    scope_chain: str | None = None       # e.g. "Trainer.train_step"
    docstring: str | None = None         # extracted docstring / leading comment
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def payload(self) -> dict[str, Any]:
        """Serialize to Qdrant payload format."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "block_type": self.block_type.value,
            "source_type": self.source_type.value,
            "source_url": self.source_url,
            "title": self.title,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "language": self.language,
            "scope_chain": self.scope_chain,
            "docstring": self.docstring,
            **self.metadata,
        }


class RetrievalResult(BaseModel):
    """A single retrieved chunk with its relevance score."""

    chunk: Chunk
    score: float                    # final score after reranking
    dense_rank: int | None = None   # original rank from dense search
    sparse_rank: int | None = None  # original rank from sparse search
    rrf_score: float | None = None  # intermediate RRF score


class ArticleSummary(BaseModel):
    """A summarized article produced by the articles pipeline."""

    article_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    source_url: str
    source_name: str = ""
    published_at: datetime | None = None
    summary: str
    key_takeaway: str
    relevance_score: int            # 1-10 from LLM-as-judge
    topics: list[str] = Field(default_factory=list)


class BookLesson(BaseModel):
    """A daily micro-lesson derived from a book chapter."""

    lesson_id: str = Field(default_factory=lambda: str(uuid4()))
    book_title: str
    chapter_title: str
    chapter_index: int
    lesson_bullets: list[str]
    reflection_question: str
    source_url: str = ""


class Digest(BaseModel):
    """The final daily digest delivered via Telegram or CLI."""

    date: datetime = Field(default_factory=datetime.utcnow)
    articles: list[ArticleSummary] = Field(default_factory=list)
    book_lesson: BookLesson | None = None
    delivery_channel: str = "cli"
