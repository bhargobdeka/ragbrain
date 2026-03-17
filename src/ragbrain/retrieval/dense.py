"""Dense retrieval using sentence-transformers + Qdrant."""

from __future__ import annotations

from ragbrain.config import settings
from ragbrain.models import RetrievalResult
from ragbrain.vectorstore.qdrant import QdrantStore


class DenseRetriever:
    """Retrieve documents using dense vector similarity."""

    def __init__(self, store: QdrantStore | None = None) -> None:
        self._store = store or QdrantStore()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        user_id: str | None = None,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        """Search Qdrant with a dense query vector.

        Returns results sorted by descending cosine similarity.
        """
        return self._store.dense_search(
            query=query,
            top_k=top_k or settings.retrieval_top_k,
            user_id=user_id,
            filters=filters,
        )
