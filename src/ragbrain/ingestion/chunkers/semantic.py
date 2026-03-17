"""Semantic chunker using sentence-transformers.

Embeds each sentence, computes cosine distances between consecutive sentences,
and splits wherever the distance exceeds a percentile threshold.  This preserves
topic coherence far better than fixed-size character splitting.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from ragbrain.config import settings


def _split_into_sentences(text: str) -> list[str]:
    """Simple sentence splitter (avoids NLTK dependency)."""
    import re

    # Split on . ! ? followed by whitespace and uppercase, preserving list items
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\-*•])", text)
    return [s.strip() for s in sentences if s.strip()]


class SemanticChunker:
    """Split text blocks into semantically coherent chunks.

    Algorithm:
    1. Split text into sentences.
    2. Encode all sentences with a local embedding model.
    3. Compute cosine distance between adjacent sentence embeddings.
    4. Find breakpoints where distance > percentile threshold.
    5. Merge sentences within each segment into a single chunk.
    """

    def __init__(
        self,
        model_name: str | None = None,
        breakpoint_percentile: float = 85.0,
        min_chunk_chars: int = 100,
    ) -> None:
        self._model_name = model_name or settings.embedding_model
        self._breakpoint_percentile = breakpoint_percentile
        self._min_chunk_chars = min_chunk_chars
        self._model: SentenceTransformer | None = None  # lazy-loaded

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def chunk(self, text: str) -> list[str]:
        """Split text into semantically coherent chunks.

        Args:
            text: The raw text to chunk.

        Returns:
            List of chunk strings.
        """
        sentences = _split_into_sentences(text)
        if len(sentences) <= 1:
            return [text.strip()] if text.strip() else []

        # Encode all sentences at once (batched, efficient)
        embeddings = self.model.encode(sentences, normalize_embeddings=True)

        # Cosine distance between consecutive sentences (1 - cosine similarity for normalized)
        distances = [
            float(1.0 - np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)
        ]

        threshold = float(np.percentile(distances, self._breakpoint_percentile))

        # Build chunks by grouping sentences between breakpoints
        chunks: list[str] = []
        current: list[str] = [sentences[0]]

        for i, dist in enumerate(distances):
            if dist > threshold:
                chunk_text = " ".join(current).strip()
                if len(chunk_text) >= self._min_chunk_chars:
                    chunks.append(chunk_text)
                elif chunks:
                    # Too short: merge into previous chunk
                    chunks[-1] = chunks[-1] + " " + chunk_text
                else:
                    chunks.append(chunk_text)
                current = [sentences[i + 1]]
            else:
                current.append(sentences[i + 1])

        # Flush remaining sentences
        if current:
            chunk_text = " ".join(current).strip()
            if len(chunk_text) >= self._min_chunk_chars:
                chunks.append(chunk_text)
            elif chunks:
                chunks[-1] = chunks[-1] + " " + chunk_text
            else:
                chunks.append(chunk_text)

        return chunks
