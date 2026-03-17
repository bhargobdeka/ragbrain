"""Ingestion pipeline: extract → chunk → store.

Single entry point for ingesting any supported source (PDF path, web URL,
RSS feed URL).  The pipeline auto-detects the source type, routes to the
correct extractor, chunks the blocks, and upserts into Qdrant.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ragbrain.ingestion.chunkers.router import ChunkRouter
from ragbrain.ingestion.extractors.pdf import PDFExtractor
from ragbrain.ingestion.extractors.rss import RSSExtractor
from ragbrain.ingestion.extractors.web import WebExtractor
from ragbrain.models import Document
from ragbrain.vectorstore.qdrant import QdrantStore

console = Console()


class IngestionPipeline:
    """Orchestrate extract → chunk → embed → store."""

    def __init__(self, store: QdrantStore | None = None, user_id: str | None = None) -> None:
        self._store = store or QdrantStore()
        self._user_id = user_id
        self._router = ChunkRouter()
        self._pdf = PDFExtractor()
        self._web = WebExtractor()
        self._rss = RSSExtractor()

    def ingest(self, source: str) -> int:
        """Ingest a single source (file path or URL).

        Automatically selects the correct extractor based on the source.

        Args:
            source: A local file path or a URL.

        Returns:
            Number of chunks indexed.
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"[cyan]Ingesting {source[:60]}...", total=None)

            progress.update(task, description=f"[cyan]Extracting {source[:50]}...")
            document = self._extract(source)

            progress.update(task, description="[cyan]Chunking document...")
            chunks = self._router.chunk_document(document)

            if not chunks:
                console.print("[yellow]No chunks produced — source may be empty or unsupported.")
                return 0

            progress.update(task, description=f"[cyan]Storing {len(chunks)} chunks in Qdrant...")
            count = self._store.upsert_chunks(chunks, user_id=self._user_id)

        console.print(
            f"[green]✓ Ingested:[/green] {document.title or source} "
            f"[dim]({count} chunks)[/dim]"
        )
        return count

    def ingest_document(self, document: Document) -> int:
        """Chunk and store an already-extracted Document.

        Useful when the pipeline is called from the articles/books pipelines
        that have already done their own extraction.
        """
        chunks = self._router.chunk_document(document)
        if not chunks:
            return 0
        return self._store.upsert_chunks(chunks, user_id=self._user_id)

    def _extract(self, source: str) -> Document:
        """Route to the correct extractor."""
        if Path(source).exists() and source.lower().endswith(".pdf"):
            return self._pdf.extract(source)
        if source.startswith("http://") or source.startswith("https://"):
            # Try web extraction (full-text) over raw RSS for single URLs
            return self._web.extract(source)
        raise ValueError(
            f"Cannot determine extractor for source: {source!r}. "
            "Provide a .pdf file path or an http(s) URL."
        )
