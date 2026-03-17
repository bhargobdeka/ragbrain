"""Reciprocal Rank Fusion (RRF) for merging dense and sparse ranked lists.

RRF formula: score(d) = Σ 1 / (k + rank(d))
where k=60 is the smoothing constant (standard from Cormack et al. 2009).
"""

from __future__ import annotations

from ragbrain.models import RetrievalResult

_RRF_K = 60  # smoothing constant


def reciprocal_rank_fusion(
    dense_results: list[RetrievalResult],
    sparse_results: list[RetrievalResult],
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """Merge dense and sparse result lists using RRF.

    Args:
        dense_results: Results from dense retrieval, sorted by descending score.
        sparse_results: Results from BM25, sorted by descending score.
        top_k: Number of results to return. None returns all.

    Returns:
        Merged list sorted by descending RRF score with both ranks annotated.
    """
    # Map chunk_id → accumulated RRF score and best result object
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievalResult] = {}

    for rank, result in enumerate(dense_results):
        cid = result.chunk.chunk_id
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        if cid not in chunk_map:
            chunk_map[cid] = result
        chunk_map[cid] = RetrievalResult(
            chunk=result.chunk,
            score=0.0,  # will be set below
            dense_rank=rank,
            sparse_rank=chunk_map[cid].sparse_rank,
        )

    for rank, result in enumerate(sparse_results):
        cid = result.chunk.chunk_id
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        if cid not in chunk_map:
            chunk_map[cid] = result
        existing = chunk_map[cid]
        chunk_map[cid] = RetrievalResult(
            chunk=result.chunk,
            score=0.0,
            dense_rank=existing.dense_rank,
            sparse_rank=rank,
        )

    # Assemble final results
    merged: list[RetrievalResult] = []
    for cid, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        r = chunk_map[cid]
        merged.append(
            RetrievalResult(
                chunk=r.chunk,
                score=rrf_score,
                dense_rank=r.dense_rank,
                sparse_rank=r.sparse_rank,
                rrf_score=rrf_score,
            )
        )

    return merged[:top_k] if top_k else merged
