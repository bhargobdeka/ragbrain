"""Abstract base class for all document extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ragbrain.models import Document


class BaseExtractor(ABC):
    """All extractors implement this interface.

    To add a premium extractor (LandingAI ADE, LLMWhisperer),
    subclass this and override `extract`.
    """

    @abstractmethod
    def can_handle(self, source: str) -> bool:
        """Return True if this extractor can handle the given source.

        Args:
            source: A file path or URL.
        """

    @abstractmethod
    def extract(self, source: str) -> Document:
        """Extract content from source into a Document with blocks.

        Args:
            source: A file path or URL.

        Returns:
            A Document with populated blocks.
        """
