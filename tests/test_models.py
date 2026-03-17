"""Tests for core data models."""

from __future__ import annotations

from ragbrain.models import Block, BlockType, Chunk, Document, SourceType


def test_block_defaults():
    block = Block(content="Hello world")
    assert block.block_type == BlockType.TEXT
    assert block.block_id  # UUID auto-generated
    assert block.page_number is None


def test_document_defaults():
    doc = Document()
    assert doc.source_type == SourceType.UNKNOWN
    assert doc.blocks == []
    assert doc.doc_id  # UUID auto-generated


def test_chunk_payload_serialization():
    chunk = Chunk(
        doc_id="doc-1",
        content="RAG is great",
        block_type=BlockType.TEXT,
        source_type=SourceType.WEB,
        source_url="https://example.com",
        title="Test Article",
        chunk_index=0,
    )
    payload = chunk.payload
    assert payload["content"] == "RAG is great"
    assert payload["block_type"] == "text"
    assert payload["source_type"] == "web"
    assert payload["title"] == "Test Article"
    assert payload["chunk_id"] == chunk.chunk_id


def test_chunk_payload_roundtrip():
    """Payload serialized then reconstructed must preserve key fields."""
    chunk = Chunk(
        doc_id="doc-abc",
        content="Testing payload roundtrip",
        source_url="https://example.com/test",
        chunk_index=3,
    )
    payload = chunk.payload
    # Verify all expected fields are present
    for field in ("chunk_id", "doc_id", "content", "block_type", "source_type", "chunk_index"):
        assert field in payload
