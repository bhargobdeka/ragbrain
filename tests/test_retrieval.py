"""Tests for retrieval components (no live Qdrant required)."""

from __future__ import annotations

from ragbrain.models import Chunk, RetrievalResult, SourceType
from ragbrain.retrieval.fusion import reciprocal_rank_fusion


def _make_result(chunk_id: str, score: float, dense_rank: int | None = None, sparse_rank: int | None = None) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        content=f"Content of {chunk_id}",
        source_type=SourceType.WEB,
    )
    return RetrievalResult(chunk=chunk, score=score, dense_rank=dense_rank, sparse_rank=sparse_rank)


class TestRRF:
    def test_identical_lists_doubles_score(self):
        """When both lists agree, the same doc should score highest."""
        results = [_make_result(f"chunk-{i}", float(10 - i)) for i in range(5)]
        fused = reciprocal_rank_fusion(results, results, top_k=5)
        assert len(fused) == 5
        assert fused[0].chunk.chunk_id == "chunk-0"

    def test_complementary_lists_merged(self):
        """Docs appearing in both lists should score higher than doc in only one."""
        dense = [_make_result("A", 1.0), _make_result("B", 0.8)]
        sparse = [_make_result("B", 1.0), _make_result("C", 0.8)]
        fused = reciprocal_rank_fusion(dense, sparse)
        ids = [r.chunk.chunk_id for r in fused]
        # B appears in both, should be first
        assert ids[0] == "B"

    def test_empty_lists_return_empty(self):
        result = reciprocal_rank_fusion([], [])
        assert result == []

    def test_top_k_limit(self):
        dense = [_make_result(f"d-{i}", float(10 - i)) for i in range(10)]
        sparse = [_make_result(f"s-{i}", float(10 - i)) for i in range(10)]
        fused = reciprocal_rank_fusion(dense, sparse, top_k=3)
        assert len(fused) == 3

    def test_ranks_preserved(self):
        dense = [_make_result("X", 1.0, dense_rank=0)]
        sparse = [_make_result("X", 1.0, sparse_rank=0)]
        fused = reciprocal_rank_fusion(dense, sparse)
        assert fused[0].dense_rank == 0
        assert fused[0].sparse_rank == 0


class TestBM25Tokenizer:
    def test_tokenize_basic(self):
        from ragbrain.retrieval.sparse import _tokenize

        tokens = _tokenize("Hello World! RAG systems.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "rag" in tokens
        assert "systems" in tokens

    def test_tokenize_empty(self):
        from ragbrain.retrieval.sparse import _tokenize

        assert _tokenize("") == []
