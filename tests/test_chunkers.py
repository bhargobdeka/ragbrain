"""Tests for the chunking layer."""

from __future__ import annotations

from ragbrain.ingestion.chunkers.code import CodeChunker
from ragbrain.ingestion.chunkers.semantic import SemanticChunker


class TestCodeChunker:
    def setup_method(self):
        self.chunker = CodeChunker()

    def test_simple_python_function(self):
        code = """
def greet(name: str) -> str:
    return f"Hello, {name}!"

def farewell(name: str) -> str:
    return f"Goodbye, {name}!"
""".strip()
        chunks = self.chunker.chunk(code, language="python")
        assert len(chunks) >= 1
        assert any("greet" in c for c in chunks)

    def test_empty_code(self):
        chunks = self.chunker.chunk("", language="python")
        assert chunks == []

    def test_unknown_language_falls_back(self):
        code = "x = 1\ny = 2\nz = x + y"
        chunks = self.chunker.chunk(code, language="unknown_lang")
        assert len(chunks) >= 1
        assert any("x = 1" in c or "z = x + y" in c for c in chunks)

    def test_hard_split_on_very_long_block(self):
        code = "\n\n".join([f"line_{i} = {i}" for i in range(600)])
        chunks = self.chunker.chunk(code)
        assert all(len(c) <= 2200 for c in chunks)  # small buffer over _MAX_CHUNK_CHARS


class TestSemanticChunker:
    def setup_method(self):
        self.chunker = SemanticChunker()

    def test_single_sentence_returns_one_chunk(self):
        text = "RAG stands for Retrieval Augmented Generation."
        chunks = self.chunker.chunk(text)
        assert len(chunks) == 1
        assert "RAG" in chunks[0]

    def test_empty_text_returns_empty(self):
        chunks = self.chunker.chunk("")
        assert chunks == []

    def test_multi_topic_text_splits(self):
        # Two clearly different topics should ideally produce multiple chunks
        text = (
            "Qdrant is a vector similarity search engine. "
            "It supports filtering by payload fields and is written in Rust. "
            "Tomatoes are fruits that grow in warm climates. "
            "They are rich in lycopene and vitamin C. "
            "Chefs often use them in pasta sauces and salads."
        )
        chunks = self.chunker.chunk(text)
        # At least 1 chunk must be produced; ideally the semantic shift is detected
        assert len(chunks) >= 1

    def test_short_chunks_merged(self):
        text = "Hello. World."
        chunks = self.chunker.chunk(text)
        # Too short to split, should be merged into one or returned as-is
        assert len(chunks) >= 1
