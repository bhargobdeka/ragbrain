"""Block-type-aware chunk router.

Routes each Block to the appropriate chunker and returns a flat list of Chunks
with correct metadata inherited from the parent Document.
"""

from __future__ import annotations

from ragbrain.ingestion.chunkers.code import CodeChunker
from ragbrain.ingestion.chunkers.semantic import SemanticChunker
from ragbrain.models import Block, BlockType, Chunk, Document


class ChunkRouter:
    """Route document blocks to the correct chunker."""

    def __init__(self) -> None:
        self._semantic = SemanticChunker()
        self._code = CodeChunker()

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Chunk all blocks in a document and return a flat list of Chunks.

        Args:
            document: The extracted Document with populated blocks.

        Returns:
            List of Chunk objects ready for embedding and storage.
        """
        chunks: list[Chunk] = []
        chunk_index = 0

        for block in document.blocks:
            block_chunks = self._chunk_block(block)
            for text in block_chunks:
                if not text.strip():
                    continue
                chunks.append(
                    Chunk(
                        doc_id=document.doc_id,
                        content=text,
                        block_type=block.block_type,
                        source_type=document.source_type,
                        source_url=document.source_url,
                        title=document.title,
                        page_number=block.page_number,
                        chunk_index=chunk_index,
                        metadata={
                            "author": document.author,
                            "language": block.language,
                            "published_at": (
                                document.published_at.isoformat()
                                if document.published_at
                                else None
                            ),
                        },
                    )
                )
                chunk_index += 1

        return chunks

    def _chunk_block(self, block: Block) -> list[str]:
        if block.block_type == BlockType.CODE:
            return self._code.chunk(block.content, language=block.language)
        if block.block_type == BlockType.IMAGE:
            # Image blocks contain the VLM-generated caption; treat as short text
            return [block.content] if block.content.strip() else []
        # TEXT, TABLE, fallback
        return self._semantic.chunk(block.content)
