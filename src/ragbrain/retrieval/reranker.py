"""CrossEncoder reranker for precise scoring of top-k candidates.

The CrossEncoder jointly encodes (query, document) pairs and produces a
scalar relevance score -- much more accurate than bi-encoder dot product
but ~10x slower, so we only apply it to the top candidates from RRF.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from ragbrain.config import settings
from ragbrain.models import RetrievalResult


class Reranker:
    """Rerank retrieval results using a CrossEncoder model."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or settings.reranker_model
        self._model: CrossEncoder | None = None  # lazy-loaded

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_n: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank results using CrossEncoder scores.

        Args:
            query: The original query string.
            results: Candidates from RRF, typically top 20.
            top_n: Number of results to return after reranking.

        Returns:
            Results sorted by descending CrossEncoder score.
        """
        if not results:
            return []

        n = top_n or settings.rerank_top_n
        pairs = [(query, r.chunk.content) for r in results]
        scores = self.model.predict(pairs)

        reranked = [
            RetrievalResult(
                chunk=result.chunk,
                score=float(score),
                dense_rank=result.dense_rank,
                sparse_rank=result.sparse_rank,
                rrf_score=result.rrf_score,
            )
            for result, score in zip(results, scores)
        ]

        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:n]
