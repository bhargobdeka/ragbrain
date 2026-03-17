"""Sparse BM25 retrieval using rank_bm25.

The BM25 index is built on-the-fly from chunks stored in Qdrant.
For large corpora, the index is cached in memory between calls.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from ragbrain.config import settings
from ragbrain.models import Chunk, RetrievalResult
from ragbrain.vectorstore.qdrant import QdrantStore


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    """BM25 sparse retriever over chunks loaded from Qdrant."""

    def __init__(self, store: QdrantStore | None = None) -> None:
        self._store = store or QdrantStore()
        self._index: BM25Okapi | None = None
        self._chunks: list[Chunk] = []
        self._user_id: str | None = None

    def _build_index(self, user_id: str | None = None) -> None:
        """Load all chunks and build (or rebuild) the BM25 index."""
        self._chunks = self._store.get_all_chunks(user_id=user_id)
        self._user_id = user_id
        if not self._chunks:
            self._index = None
            return
        tokenized = [_tokenize(c.content) for c in self._chunks]
        self._index = BM25Okapi(tokenized)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        user_id: str | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve documents using BM25 keyword matching.

        The index is rebuilt if the user_id changes or if it hasn't been
        built yet.  For production use with large corpora, consider persisting
        the index to disk.
        """
        if self._index is None or user_id != self._user_id:
            self._build_index(user_id)

        if self._index is None or not self._chunks:
            return []

        k = top_k or settings.retrieval_top_k
        tokens = _tokenize(query)
        scores = self._index.get_scores(tokens)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results: list[RetrievalResult] = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                break
            results.append(
                RetrievalResult(
                    chunk=self._chunks[idx],
                    score=float(scores[idx]),
                    sparse_rank=rank,
                )
            )

        return results

    def invalidate(self) -> None:
        """Force index rebuild on next call (e.g. after new ingestion)."""
        self._index = None
        self._chunks = []
