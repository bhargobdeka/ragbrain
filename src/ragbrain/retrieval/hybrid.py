"""Hybrid retrieval orchestrator.

Combines dense (Qdrant) + sparse (BM25) retrieval, merges with RRF,
then reranks the top candidates with a CrossEncoder.

Usage::

    retriever = HybridRetriever()
    results = retriever.retrieve("What is RLHF?", top_n=5)
    for r in results:
        print(r.score, r.chunk.content[:100])
"""

from __future__ import annotations

from ragbrain.config import settings
from ragbrain.models import RetrievalResult
from ragbrain.retrieval.dense import DenseRetriever
from ragbrain.retrieval.fusion import reciprocal_rank_fusion
from ragbrain.retrieval.reranker import Reranker
from ragbrain.retrieval.sparse import BM25Retriever
from ragbrain.vectorstore.qdrant import QdrantStore


class HybridRetriever:
    """Full hybrid retrieval pipeline: dense + BM25 → RRF → CrossEncoder."""

    def __init__(self, store: QdrantStore | None = None) -> None:
        _store = store or QdrantStore()
        self._dense = DenseRetriever(store=_store)
        self._sparse = BM25Retriever(store=_store)
        self._reranker = Reranker()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        top_n: int | None = None,
        user_id: str | None = None,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        """End-to-end retrieval: dense + sparse → RRF → rerank.

        Args:
            query: Natural language question or search string.
            top_k: Candidates from each retriever (before fusion). Default from config.
            top_n: Final results after reranking. Default from config.
            user_id: Namespace for multi-tenant isolation.
            filters: Optional Qdrant payload filters (e.g. {"source_type": "pdf"}).

        Returns:
            Final ranked list of RetrievalResult objects.
        """
        k = top_k or settings.retrieval_top_k
        n = top_n or settings.rerank_top_n

        # Retrieve from both systems in parallel (simple sequential for now)
        dense_results = self._dense.retrieve(query, top_k=k, user_id=user_id, filters=filters)
        sparse_results = self._sparse.retrieve(query, top_k=k, user_id=user_id)

        # Fuse rankings
        fused = reciprocal_rank_fusion(dense_results, sparse_results, top_k=min(k * 2, 40))

        # Rerank top candidates
        return self._reranker.rerank(query, fused, top_n=n)

    def invalidate_sparse_index(self) -> None:
        """Force BM25 index rebuild (call after ingesting new documents)."""
        self._sparse.invalidate()
