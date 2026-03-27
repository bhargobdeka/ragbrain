"""Encoder singletons for text and code embeddings.

TextEncoder  — wraps sentence-transformers ``all-mpnet-base-v2`` (general prose).
CodeEncoder  — wraps ``microsoft/unixcoder-base`` via HuggingFace transformers
               (code-specific: understands identifiers, ASTs, docstrings).

Both produce 768-dimensional L2-normalised vectors, matching the Qdrant
collection dimensions defined in ``config.py``.

References:
  - SBERT (TextEncoder): https://arxiv.org/abs/1908.10084
  - UniXcoder (CodeEncoder): https://arxiv.org/abs/2203.03850
"""

from __future__ import annotations

import numpy as np

from ragbrain.config import settings


class TextEncoder:
    """Sentence-transformers encoder for natural language prose."""

    _instance: TextEncoder | None = None

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(settings.embedding_model)

    @classmethod
    def get(cls) -> TextEncoder:
        """Return the module-level singleton (lazy init)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """Encode a batch of texts into embeddings.

        Args:
            texts: List of strings to encode.
            normalize: If True, L2-normalise the output vectors.

        Returns:
            ``(N, embedding_dim)`` float32 numpy array.
        """
        return self._model.encode(texts, normalize_embeddings=normalize)  # type: ignore[return-value]


class CodeEncoder:
    """UniXcoder encoder for source code (microsoft/unixcoder-base).

    Uses mean-pooling over non-padding token embeddings followed by L2
    normalisation — the standard approach for dense retrieval with BERT-style
    models (see DPR: https://arxiv.org/abs/2004.04906).

    The model is downloaded from HuggingFace on first use (~500 MB).
    Set ``RAGBRAIN_USE_CODE_ENCODER=false`` in .env to skip and use
    ``TextEncoder`` for all content instead.
    """

    _instance: CodeEncoder | None = None

    def __init__(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(settings.code_embedding_model)
        self._model = AutoModel.from_pretrained(settings.code_embedding_model)
        self._model.eval()
        self._torch = torch

    @classmethod
    def get(cls) -> CodeEncoder:
        """Return the module-level singleton (lazy init)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """Encode a batch of code strings into embeddings.

        Args:
            texts: List of source-code strings to encode.
            normalize: If True, L2-normalise the output vectors.

        Returns:
            ``(N, code_embedding_dim)`` float32 numpy array.
        """
        torch = self._torch
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Mean pooling — weight each token by its attention mask
        attn = inputs["attention_mask"].unsqueeze(-1).float()
        token_embs = outputs.last_hidden_state
        embeddings = (token_embs * attn).sum(1) / attn.sum(1).clamp(min=1e-9)

        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()
