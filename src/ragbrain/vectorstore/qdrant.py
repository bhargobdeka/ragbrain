"""Qdrant vector store client.

Handles collection creation, vector upserts, and similarity search.
Namespacing is implemented via Qdrant's collection-per-user pattern
(collection name = "<base>_<user_id>"), keeping each user's data isolated.

Dual-encoder mode (RAGBRAIN_USE_CODE_ENCODER=true, the default):
  Each point stores TWO named vectors:
    "text" → all-mpnet-base-v2   (for prose / natural language queries)
    "code" → microsoft/unixcoder-base  (for code chunks / identifier search)

  TEXT blocks: text=mpnet, code=mpnet (same vector, mpnet handles prose well)
  CODE blocks: text=mpnet, code=unixcoder

  dense_search() queries both spaces, then merges with Reciprocal Rank Fusion.

Single-encoder mode (RAGBRAIN_USE_CODE_ENCODER=false):
  Behaves exactly as before — one vector per point, mpnet for everything.
  Use when you don't need code understanding or want to reduce memory.

NOTE: Changing use_code_encoder after ingestion requires deleting the
      collection and re-ingesting (the vector schema is incompatible).
"""

from __future__ import annotations

import warnings
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ragbrain.config import settings
from ragbrain.models import BlockType, Chunk, RetrievalResult


# ---------------------------------------------------------------------------
# Internal RRF helper
# ---------------------------------------------------------------------------

def _rrf_merge(result_lists: list[list[RetrievalResult]], top_k: int, k: int = 60) -> list[RetrievalResult]:
    """Merge ranked result lists using Reciprocal Rank Fusion.

    Args:
        result_lists: One list per retrieval source.
        top_k: Maximum results to return.
        k: RRF constant (60 works well empirically; Cormack et al. 2009).

    Returns:
        Merged, deduplicated, re-ranked list of RetrievalResult.
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}

    for results in result_lists:
        for rank, result in enumerate(results):
            cid = result.chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            chunk_map[cid] = result.chunk

    sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:top_k]
    return [
        RetrievalResult(chunk=chunk_map[cid], score=rrf_scores[cid])
        for cid in sorted_ids
    ]


# ---------------------------------------------------------------------------
# QdrantStore
# ---------------------------------------------------------------------------

class QdrantStore:
    """Thin wrapper around QdrantClient with dual-encoder embedding support.

    Supports two Qdrant connection modes (set via RAGBRAIN_QDRANT_MODE):
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
            local_path = os.path.expanduser(settings.qdrant_local_path)
            os.makedirs(local_path, exist_ok=True)
            self._client = QdrantClient(path=local_path)
        else:
            self._client = QdrantClient(
                url=url or settings.qdrant_url,
                api_key=api_key or settings.qdrant_api_key or None,
            )

    def close(self) -> None:
        """Close the underlying Qdrant client safely.

        Calling this prevents noisy interpreter-shutdown tracebacks from
        qdrant-client's internal __del__ handlers.
        """
        client = getattr(self, "_client", None)
        if client is None:
            return
        try:
            client.close()
        except Exception:
            # Avoid surfacing destructor-time shutdown noise to users.
            pass
        finally:
            self._client = None  # type: ignore[assignment]

    def __del__(self) -> None:
        """Best-effort cleanup."""
        try:
            self.close()
        except Exception:
            pass

    # ---- Collection management ----------------------------------------

    def collection_name(self, user_id: str | None = None) -> str:
        if user_id:
            return f"{self._default_collection}_{user_id}"
        return self._default_collection

    def ensure_collection(self, user_id: str | None = None) -> str:
        """Create the collection if it doesn't exist. Returns collection name."""
        name = self.collection_name(user_id)
        existing = {c.name for c in self._client.get_collections().collections}

        if name in existing:
            self._check_schema_compatibility(name)
            return name

        # Build vectors config based on encoder mode
        if settings.use_code_encoder:
            vectors_config: Any = {
                "text": qmodels.VectorParams(
                    size=settings.embedding_dim,
                    distance=qmodels.Distance.COSINE,
                ),
                "code": qmodels.VectorParams(
                    size=settings.code_embedding_dim,
                    distance=qmodels.Distance.COSINE,
                ),
            }
        else:
            vectors_config = qmodels.VectorParams(
                size=settings.embedding_dim,
                distance=qmodels.Distance.COSINE,
            )

        self._client.create_collection(
            collection_name=name,
            vectors_config=vectors_config,
        )

        # Payload indexes for fast metadata filtering
        for field in ("source_type", "block_type", "doc_id", "language"):
            self._client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
        return name

    def _check_schema_compatibility(self, name: str) -> None:
        """Warn if the existing collection schema doesn't match current config."""
        try:
            info = self._client.get_collection(name)
            is_named = isinstance(info.config.params.vectors, dict)
            needs_named = settings.use_code_encoder
            if is_named != needs_named:
                warnings.warn(
                    f"Collection '{name}' uses {'named' if is_named else 'single'} vectors "
                    f"but RAGBRAIN_USE_CODE_ENCODER={settings.use_code_encoder}. "
                    "Delete the collection and re-ingest to upgrade to the dual-encoder schema.",
                    stacklevel=3,
                )
        except Exception:
            pass

    # ---- Upsert -------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        user_id: str | None = None,
        batch_size: int = 32,
    ) -> int:
        """Embed and upsert chunks into Qdrant.

        Returns:
            Number of chunks upserted.
        """
        collection = self.ensure_collection(user_id)

        if settings.use_code_encoder:
            self._upsert_dual(chunks, collection, batch_size)
        else:
            self._upsert_single(chunks, collection, batch_size)

        return len(chunks)

    def _upsert_single(self, chunks: list[Chunk], collection: str, batch_size: int) -> None:
        """Upsert with a single mpnet vector per chunk."""
        from ragbrain.vectorstore.encoders import TextEncoder
        enc = TextEncoder.get()

        texts = [c.content for c in chunks]
        all_vecs: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            vecs = enc.encode(texts[i: i + batch_size])
            all_vecs.extend(vecs.tolist())

        points = [
            qmodels.PointStruct(id=chunk.chunk_id, vector=vec, payload=chunk.payload)
            for chunk, vec in zip(chunks, all_vecs)
        ]
        for i in range(0, len(points), batch_size):
            self._client.upsert(collection_name=collection, points=points[i: i + batch_size])

    def _upsert_dual(self, chunks: list[Chunk], collection: str, batch_size: int) -> None:
        """Upsert with named 'text' and 'code' vectors per chunk."""
        from ragbrain.vectorstore.encoders import TextEncoder

        text_enc = TextEncoder.get()

        texts = [c.content for c in chunks]

        # Encode all with text encoder (used for 'text' vector on every chunk)
        text_vecs: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            vecs = text_enc.encode(texts[i: i + batch_size])
            text_vecs.extend(vecs.tolist())

        # Start with text vecs for 'code' space; replace CODE chunks with UniXcoder
        code_vecs: list[list[float]] = list(text_vecs)
        code_indices = [i for i, c in enumerate(chunks) if c.block_type == BlockType.CODE]
        if code_indices:
            from ragbrain.vectorstore.encoders import CodeEncoder
            code_texts = [chunks[i].content for i in code_indices]
            try:
                code_enc = CodeEncoder.get()
                encoded_code: list[list[float]] = []
                for i in range(0, len(code_texts), batch_size):
                    vecs = code_enc.encode(code_texts[i: i + batch_size])
                    encoded_code.extend(vecs.tolist())
                for j, idx in enumerate(code_indices):
                    code_vecs[idx] = encoded_code[j]
            except Exception as exc:
                warnings.warn(
                    "Failed to load/use code embedding model "
                    f"({settings.code_embedding_model}): {exc}. "
                    "Falling back to text embeddings for code vectors.",
                    stacklevel=3,
                )

        points = [
            qmodels.PointStruct(
                id=chunk.chunk_id,
                vector={"text": text_vecs[i], "code": code_vecs[i]},
                payload=chunk.payload,
            )
            for i, chunk in enumerate(chunks)
        ]
        for i in range(0, len(points), batch_size):
            self._client.upsert(collection_name=collection, points=points[i: i + batch_size])

    # ---- Search -------------------------------------------------------

    def dense_search(
        self,
        query: str,
        top_k: int | None = None,
        user_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Run dense similarity search.

        In dual-encoder mode, searches both 'text' and 'code' vector spaces and
        merges results with RRF before returning.

        Args:
            query: Natural language or code query.
            top_k: Number of results to return after merging.
            user_id: Namespace for multi-tenant isolation.
            filters: Optional Qdrant payload key→value filters.

        Returns:
            List of :class:`RetrievalResult` sorted by descending relevance.
        """
        k = top_k or settings.retrieval_top_k

        if settings.use_code_encoder:
            return self._dual_dense_search(query, k, user_id, filters)
        return self._single_dense_search(query, k, user_id, filters)

    def _single_dense_search(
        self,
        query: str,
        top_k: int,
        user_id: str | None,
        filters: dict[str, Any] | None,
    ) -> list[RetrievalResult]:
        from ragbrain.vectorstore.encoders import TextEncoder
        enc = TextEncoder.get()
        query_vec = enc.encode([query])[0].tolist()
        return self._qdrant_search(query_vec, None, top_k, user_id, filters)

    def _dual_dense_search(
        self,
        query: str,
        top_k: int,
        user_id: str | None,
        filters: dict[str, Any] | None,
    ) -> list[RetrievalResult]:
        from ragbrain.vectorstore.encoders import CodeEncoder, TextEncoder
        text_enc = TextEncoder.get()
        code_enc = CodeEncoder.get()

        text_vec = text_enc.encode([query])[0].tolist()
        code_vec = code_enc.encode([query])[0].tolist()

        text_results = self._qdrant_search(text_vec, "text", top_k, user_id, filters)
        code_results = self._qdrant_search(code_vec, "code", top_k, user_id, filters)

        return _rrf_merge([text_results, code_results], top_k)

    def _qdrant_search(
        self,
        query_vec: list[float],
        vector_name: str | None,
        top_k: int,
        user_id: str | None,
        filters: dict[str, Any] | None,
    ) -> list[RetrievalResult]:
        collection = self.collection_name(user_id)
        qdrant_filter = self._build_filter(filters)

        kwargs: dict[str, Any] = {
            "collection_name": collection,
            "query": query_vec,
            "limit": top_k,
            "query_filter": qdrant_filter,
            "with_payload": True,
        }
        if vector_name:
            kwargs["using"] = vector_name

        response = self._client.query_points(**kwargs)

        results: list[RetrievalResult] = []
        for rank, hit in enumerate(response.points):
            chunk = self._hit_to_chunk(hit)
            results.append(RetrievalResult(chunk=chunk, score=float(hit.score), dense_rank=rank))
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

    @staticmethod
    def _build_filter(filters: dict[str, Any] | None) -> qmodels.Filter | None:
        if not filters:
            return None
        conditions = [
            qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v))
            for k, v in filters.items()
        ]
        return qmodels.Filter(must=conditions)

    def _hit_to_chunk(self, hit: Any) -> Chunk:
        return self._payload_to_chunk(hit.id, hit.payload or {})

    def _record_to_chunk(self, record: Any) -> Chunk:
        return self._payload_to_chunk(record.id, record.payload or {})

    def _payload_to_chunk(self, chunk_id: Any, p: dict) -> Chunk:
        _known = {
            "chunk_id", "doc_id", "content", "block_type", "source_type",
            "source_url", "title", "page_number", "chunk_index",
            "language", "scope_chain", "docstring",
        }
        return Chunk(
            chunk_id=str(chunk_id),
            doc_id=p.get("doc_id", ""),
            content=p.get("content", ""),
            block_type=BlockType(p.get("block_type", "text")),
            source_type=__import__("ragbrain.models", fromlist=["SourceType"]).SourceType(
                p.get("source_type", "unknown")
            ),
            source_url=p.get("source_url", ""),
            title=p.get("title", ""),
            page_number=p.get("page_number"),
            chunk_index=p.get("chunk_index", 0),
            language=p.get("language"),
            scope_chain=p.get("scope_chain"),
            docstring=p.get("docstring"),
            metadata={k: v for k, v in p.items() if k not in _known},
        )
