"""Block-type-aware chunk router.

Routes each Block to the appropriate chunker and returns a flat list of Chunks
with correct metadata inherited from the parent Document.

CODE blocks → ASTCodeChunker → Chunk with scope_chain, docstring, language
TEXT/TABLE  → SemanticChunker
IMAGE       → passed through as-is (VLM caption already a short text)
"""

from __future__ import annotations

from ragbrain.ingestion.chunkers.code import ASTCodeChunker
from ragbrain.ingestion.chunkers.semantic import SemanticChunker
from ragbrain.models import Block, BlockType, Chunk, Document


class ChunkRouter:
    """Route document blocks to the correct chunker."""

    def __init__(self) -> None:
        self._semantic = SemanticChunker()
        self._code = ASTCodeChunker()

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Chunk all blocks in a document and return a flat list of Chunks.

        Args:
            document: The extracted Document with populated blocks.

        Returns:
            List of :class:`Chunk` objects ready for embedding and storage.
        """
        chunks: list[Chunk] = []
        chunk_index = 0
        published_iso = (
            document.published_at.isoformat() if document.published_at else None
        )

        for block in document.blocks:
            new_chunks = self._chunk_block(block, document, chunk_index, published_iso)
            chunks.extend(new_chunks)
            chunk_index += len(new_chunks)

        return chunks

    # ---- Private helpers -----------------------------------------------

    def _chunk_block(
        self,
        block: Block,
        document: Document,
        start_index: int,
        published_iso: str | None,
    ) -> list[Chunk]:
        if block.block_type == BlockType.CODE:
            return self._chunk_code(block, document, start_index, published_iso)
        if block.block_type == BlockType.IMAGE:
            if not block.content.strip():
                return []
            return [self._make_chunk(
                block.content, block, document, start_index, published_iso
            )]
        # TEXT, TABLE, fallback
        text_parts = self._semantic.chunk(block.content)
        return [
            self._make_chunk(text, block, document, start_index + i, published_iso)
            for i, text in enumerate(text_parts)
            if text.strip()
        ]

    def _chunk_code(
        self,
        block: Block,
        document: Document,
        start_index: int,
        published_iso: str | None,
    ) -> list[Chunk]:
        units = self._code.chunk(block.content, language=block.language)
        chunks: list[Chunk] = []
        for i, unit in enumerate(units):
            if not unit.content.strip():
                continue
            chunks.append(Chunk(
                doc_id=document.doc_id,
                content=unit.content,
                block_type=block.block_type,
                source_type=document.source_type,
                source_url=document.source_url,
                title=document.title,
                page_number=block.page_number,
                chunk_index=start_index + i,
                language=block.language,
                scope_chain=unit.scope_chain or None,
                docstring=unit.docstring or None,
                metadata={
                    "author": document.author,
                    "published_at": published_iso,
                    # Preserve import context so the LLM knows dependencies
                    "imports": unit.imports,
                },
            ))
        return chunks

    def _make_chunk(
        self,
        text: str,
        block: Block,
        document: Document,
        index: int,
        published_iso: str | None,
    ) -> Chunk:
        return Chunk(
            doc_id=document.doc_id,
            content=text,
            block_type=block.block_type,
            source_type=document.source_type,
            source_url=document.source_url,
            title=document.title,
            page_number=block.page_number,
            chunk_index=index,
            metadata={
                "author": document.author,
                "language": block.language,
                "published_at": published_iso,
            },
        )
