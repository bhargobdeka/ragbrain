"""Qdrant vector store client.

Handles collection creation, vector upserts, and similarity search.
Namespacing is implemented via Qdrant's collection-per-user pattern
(collection name = "<base>_<user_id>"), keeping each user's data isolated.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

from ragbrain.config import settings
from ragbrain.models import Chunk, RetrievalResult


class QdrantStore:
    """Thin wrapper around QdrantClient with embedding support.

    Supports two modes (set via RAGBRAIN_QDRANT_MODE in .env):
      local  — embedded Qdrant, data persisted to RAGBRAIN_QDRANT_LOCAL_PATH.
               No Docker or external server required.
      server — connects to a running Qdrant instance at RAGBRAIN_QDRANT_URL.
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection: str | None = None,
    ) -> None:
        self._default_collection = collection or settings.qdrant_collection

        if settings.qdrant_mode == "local":
            import os
            local_path = settings.qdrant_local_path
            os.makedirs(local_path, exist_ok=True)
            self._client = QdrantClient(path=local_path)
        else:
            self._client = QdrantClient(
                url=url or settings.qdrant_url,
                api_key=api_key or settings.qdrant_api_key or None,
            )
        self._encoder: SentenceTransformer | None = None

    @property
    def encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            self._encoder = SentenceTransformer(settings.embedding_model)
        return self._encoder

    # ---- Collection management ----------------------------------------

    def collection_name(self, user_id: str | None = None) -> str:
        if user_id:
            return f"{self._default_collection}_{user_id}"
        return self._default_collection

    def ensure_collection(self, user_id: str | None = None) -> str:
        """Create the collection if it doesn't exist. Returns collection name."""
        name = self.collection_name(user_id)
        existing = {c.name for c in self._client.get_collections().collections}
        if name not in existing:
            self._client.create_collection(
                collection_name=name,
                vectors_config=qmodels.VectorParams(
                    size=settings.embedding_dim,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            # Create payload indexes for fast metadata filtering
            for field in ("source_type", "block_type", "doc_id"):
                self._client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
        return name

    # ---- Upsert -------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        user_id: str | None = None,
        batch_size: int = 64,
    ) -> int:
        """Embed and upsert chunks into Qdrant.

        Returns:
            Number of chunks upserted.
        """
        collection = self.ensure_collection(user_id)
        texts = [c.content for c in chunks]

        # Encode in batches
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vecs = self.encoder.encode(batch, normalize_embeddings=True)
            all_embeddings.extend(vecs.tolist())

        points = [
            qmodels.PointStruct(
                id=chunk.chunk_id,
                vector=embedding,
                payload=chunk.payload,
            )
            for chunk, embedding in zip(chunks, all_embeddings)
        ]

        for i in range(0, len(points), batch_size):
            self._client.upsert(
                collection_name=collection,
                points=points[i : i + batch_size],
            )

        return len(chunks)

    # ---- Search -------------------------------------------------------

    def dense_search(
        self,
        query: str,
        top_k: int | None = None,
        user_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Run dense (vector) similarity search.

        Args:
            query: Natural language query.
            top_k: Number of results to return.
            user_id: Namespace for multi-tenant isolation.
            filters: Optional Qdrant payload filters.

        Returns:
            List of RetrievalResult ordered by descending similarity.
        """
        k = top_k or settings.retrieval_top_k
        collection = self.collection_name(user_id)

        query_vec = self.encoder.encode([query], normalize_embeddings=True)[0].tolist()

        qdrant_filter = None
        if filters:
            conditions = [
                qmodels.FieldCondition(
                    key=k,
                    match=qmodels.MatchValue(value=v),
                )
                for k, v in filters.items()
            ]
            qdrant_filter = qmodels.Filter(must=conditions)

        response = self._client.query_points(
            collection_name=collection,
            query=query_vec,
            limit=k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        results: list[RetrievalResult] = []
        for rank, hit in enumerate(response.points):
            chunk = self._hit_to_chunk(hit)
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(hit.score),
                    dense_rank=rank,
                )
            )

        return results

    def get_all_chunks(
        self, user_id: str | None = None, limit: int = 10_000
    ) -> list[Chunk]:
        """Retrieve all chunk payloads (used for BM25 index construction)."""
        collection = self.collection_name(user_id)
        try:
            records, _ = self._client.scroll(
                collection_name=collection,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            return []
        return [self._record_to_chunk(r) for r in records]

    # ---- Helpers ------------------------------------------------------

    def _hit_to_chunk(self, hit: Any) -> Chunk:
        p = hit.payload or {}
        return self._payload_to_chunk(hit.id, p)

    def _record_to_chunk(self, record: Any) -> Chunk:
        p = record.payload or {}
        return self._payload_to_chunk(record.id, p)

    def _payload_to_chunk(self, chunk_id: Any, p: dict) -> Chunk:
        from ragbrain.models import BlockType, SourceType

        return Chunk(
            chunk_id=str(chunk_id),
            doc_id=p.get("doc_id", ""),
            content=p.get("content", ""),
            block_type=BlockType(p.get("block_type", "text")),
            source_type=SourceType(p.get("source_type", "unknown")),
            source_url=p.get("source_url", ""),
            title=p.get("title", ""),
            page_number=p.get("page_number"),
            chunk_index=p.get("chunk_index", 0),
            metadata={k: v for k, v in p.items() if k not in {
                "chunk_id", "doc_id", "content", "block_type",
                "source_type", "source_url", "title", "page_number", "chunk_index",
            }},
        )
