"""Language-aware code chunker.

Splits code blocks at class/function boundaries rather than by character count,
so retrieved chunks are always syntactically complete units.
"""

from __future__ import annotations

import re

# Approximate max tokens per chunk (characters / 4 ≈ tokens)
_MAX_CHUNK_CHARS = 2048

# Patterns that mark top-level boundaries in common languages
_BOUNDARY_PATTERNS: dict[str, re.Pattern] = {
    "python": re.compile(r"^(class |def |async def )", re.MULTILINE),
    "javascript": re.compile(r"^(function |const |class |async function )", re.MULTILINE),
    "typescript": re.compile(r"^(function |const |class |interface |type |async function )", re.MULTILINE),
    "java": re.compile(r"^(public |private |protected |class )", re.MULTILINE),
    "go": re.compile(r"^func ", re.MULTILINE),
    "rust": re.compile(r"^(fn |pub fn |impl |struct |enum )", re.MULTILINE),
    "bash": re.compile(r"^(\w+\(\))", re.MULTILINE),
}
_GENERIC_BOUNDARY = re.compile(r"\n{2,}")  # blank lines for unknown languages


class CodeChunker:
    """Split code blocks at function/class boundaries."""

    def chunk(self, code: str, language: str | None = None) -> list[str]:
        """Split code into chunks respecting language structure.

        Args:
            code: The raw code string.
            language: Optional language hint (e.g. "python", "javascript").

        Returns:
            List of code chunk strings.
        """
        if not code.strip():
            return []

        pattern = _BOUNDARY_PATTERNS.get((language or "").lower(), _GENERIC_BOUNDARY)
        return self._split_on_pattern(code, pattern)

    def _split_on_pattern(self, code: str, pattern: re.Pattern) -> list[str]:
        """Split code on boundary pattern, respecting _MAX_CHUNK_CHARS."""
        matches = list(pattern.finditer(code))
        if not matches:
            return self._hard_split(code)

        chunks: list[str] = []
        boundaries = [m.start() for m in matches]

        # First segment (before first boundary)
        if boundaries[0] > 0:
            header = code[: boundaries[0]].strip()
            if header:
                chunks.append(header)

        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(code)
            segment = code[start:end].strip()
            if not segment:
                continue
            if len(segment) > _MAX_CHUNK_CHARS:
                chunks.extend(self._hard_split(segment))
            else:
                chunks.append(segment)

        return chunks or [code.strip()]

    def _hard_split(self, code: str) -> list[str]:
        """Fallback: split by character limit on blank lines."""
        parts = re.split(r"\n{2,}", code)
        chunks: list[str] = []
        current = ""
        for part in parts:
            if len(current) + len(part) > _MAX_CHUNK_CHARS and current:
                chunks.append(current.strip())
                current = part
            else:
                current = (current + "\n\n" + part) if current else part
        if current.strip():
            chunks.append(current.strip())
        return chunks
