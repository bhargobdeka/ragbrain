"""Books pipeline: PDF → chapter detection → daily micro-lesson rotation.

State is persisted as JSON at settings.book_state_file.
Each book tracks the current chapter index so the system delivers
one chapter lesson per day, cycling through all chapters.

Flow:
1. Detect chapters from a PDF (by heading heuristics)
2. Track progress per book in a JSON state file
3. On each invocation, return the next chapter's micro-lesson
4. Also index the chapter text into Qdrant for RAG queries
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ragbrain.config import settings
from ragbrain.ingestion.extractors.pdf import PDFExtractor
from ragbrain.ingestion.pipeline import IngestionPipeline
from ragbrain.models import Block, BlockType, BookLesson

# ---- Prompts --------------------------------------------------------

_LESSON_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert teacher. Given a book chapter, create a concise micro-lesson:\n"
        "1. Three bullet points covering the key concepts\n"
        "2. One reflection question that connects the material to real-world application\n\n"
        "Return JSON with: lesson_bullets (list[str] of 3 items), reflection_question (str).",
    ),
    (
        "human",
        "Book: {book_title}\nChapter: {chapter_title}\n\nContent:\n{content}",
    ),
])


class LessonOutput(BaseModel):
    lesson_bullets: list[str] = Field(min_length=1)
    reflection_question: str


# ---- Chapter detection heuristics -----------------------------------

# Tier 1: explicit "Chapter N" heading
_EXPLICIT_CHAPTER = re.compile(r"^chapter\s+\d+[\.:]\s*.+", re.IGNORECASE)

# Tier 2: "N  Title" (2+ spaces, Manning / O'Reilly style, no sub-section dot)
_SPACED_CHAPTER = re.compile(r"^\d{1,2}\s{2,}[A-Z][a-zA-Z ,&:\-]{3,60}$")

# Tier 3: "N Title" (single space, short, only if N changes from previous)
_SHORT_CHAPTER = re.compile(r"^\d{1,2}\s+[A-Z][a-zA-Z ,&:\-]{3,60}$")

# Tier 4: "N.1 Title" — first section of each chapter (most reliable in MEAP/technical books)
_FIRST_SECTION = re.compile(r"^(\d{1,2})\.1[\s:]+[A-Z].{3,70}$")

# Noise: all-caps appendix math, list items starting with digits
_NOISE = re.compile(
    r"DERIVING|GRADIENT|OBJECTIVE|PROOF|THEOREM|^[A-Z\s]{15,}$|^\d+\.\s+[a-z]",
    re.IGNORECASE,
)


def _chapter_number(text: str) -> int | None:
    """Return chapter number if text looks like a top-level chapter heading."""
    text = text.strip()
    if len(text) > 90 or len(text) < 4:
        return None
    if _NOISE.search(text):
        return None
    if _EXPLICIT_CHAPTER.match(text):
        m = re.search(r"\d+", text)
        return int(m.group()) if m else None
    if _SPACED_CHAPTER.match(text):
        return int(re.match(r"(\d+)", text).group(1))
    m = _FIRST_SECTION.match(text)
    if m:
        return int(m.group(1))
    return None


def _split_into_chapters(blocks: list[Block]) -> list[tuple[str, str]]:
    """Split blocks into (chapter_title, chapter_text) tuples.

    Strategy (in order of preference):
    1. Detect explicit chapter headings / first-section headings.
    2. If fewer than 3 chapters found, fall back to equal-size page splits
       (~15 pages each) so every book gets reasonable granularity.
    """
    chapters: list[tuple[str, str]] = []
    current_title = "Introduction"
    current_parts: list[str] = []
    last_chapter_num = -1

    for block in blocks:
        if block.block_type != BlockType.TEXT:
            continue
        text = block.content.strip()
        ch_num = _chapter_number(text)

        # Only advance if chapter number increases (avoids TOC duplicates)
        if ch_num is not None and ch_num > last_chapter_num:
            if current_parts:
                chapters.append((current_title, "\n\n".join(current_parts)))
            current_title = text
            current_parts = []
            last_chapter_num = ch_num
        else:
            current_parts.append(text)

    if current_parts:
        chapters.append((current_title, "\n\n".join(current_parts)))

    # Fallback: split into equal page-based segments
    if len(chapters) < 3:
        return _page_based_split(blocks, pages_per_chunk=15)

    return chapters


def _page_based_split(blocks: list[Block], pages_per_chunk: int = 15) -> list[tuple[str, str]]:
    """Fallback: split blocks into roughly equal page-based chunks."""
    page_groups: dict[int, list[str]] = {}
    for block in blocks:
        if block.block_type == BlockType.TEXT:
            pg = block.page_number or 0
            page_groups.setdefault(pg, []).append(block.content)

    pages = sorted(page_groups.keys())
    chunks: list[tuple[str, str]] = []
    for i in range(0, len(pages), pages_per_chunk):
        chunk_pages = pages[i : i + pages_per_chunk]
        text = "\n\n".join(
            t for pg in chunk_pages for t in page_groups[pg]
        )
        label = f"Pages {chunk_pages[0]}–{chunk_pages[-1]}"
        chunks.append((label, text))
    return chunks if chunks else [("Full Book", "\n\n".join(b.content for b in blocks))]


# ---- State management -----------------------------------------------

def _load_state(state_file: Path) -> dict:
    if state_file.exists():
        try:
            return json.loads(state_file.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_state(state: dict, state_file: Path) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2))


# ---- Pipeline -------------------------------------------------------

class BooksPipeline:
    """Generate daily micro-lessons from PDF books."""

    def __init__(self, user_id: str | None = None) -> None:
        self._user_id = user_id
        self._pdf = PDFExtractor()
        self._ingestor = IngestionPipeline(user_id=user_id)
        self._state_file = Path(settings.book_state_file)

    def close(self) -> None:
        """Close underlying ingestion resources."""
        try:
            self._ingestor.close()
        except Exception:
            pass

    def ingest_book(self, pdf_path: str) -> int:
        """Index a PDF book into Qdrant and register it in the lesson tracker.

        Returns:
            Number of chunks indexed.
        """
        doc = self._pdf.extract(pdf_path)
        chunks = self._ingestor.ingest_document(doc)

        # Register book in state
        state = _load_state(self._state_file)
        book_key = Path(pdf_path).stem
        if book_key not in state:
            chapters = _split_into_chapters(doc.blocks)
            state[book_key] = {
                "title": doc.title or book_key,
                "pdf_path": str(Path(pdf_path).resolve()),
                "chapters": [{"title": t, "content": c[:2000]} for t, c in chapters],
                "current_chapter_index": 0,
            }
            _save_state(state, self._state_file)

        return chunks

    def get_next_lesson(self, book_key: str | None = None) -> BookLesson | None:
        """Return today's micro-lesson for a book (rotating through chapters).

        If book_key is None, picks the first registered book.
        """
        state = _load_state(self._state_file)
        if not state:
            return None

        key = book_key or next(iter(state))
        book = state.get(key)
        if not book:
            return None

        chapters = book.get("chapters", [])
        if not chapters:
            return None

        idx = book.get("current_chapter_index", 0) % len(chapters)
        chapter = chapters[idx]

        lesson = self._generate_lesson(
            book_title=book["title"],
            chapter_title=chapter["title"],
            chapter_index=idx,
            content=chapter["content"],
        )

        # Advance to next chapter
        state[key]["current_chapter_index"] = (idx + 1) % len(chapters)
        _save_state(state, self._state_file)

        return lesson

    def get_all_book_keys(self) -> list[str]:
        """Return keys of all registered books."""
        return list(_load_state(self._state_file).keys())

    def _generate_lesson(
        self,
        book_title: str,
        chapter_title: str,
        chapter_index: int,
        content: str,
    ) -> BookLesson | None:
        llm = settings.get_llm()
        chain = _LESSON_PROMPT | llm.with_structured_output(LessonOutput)

        try:
            output = chain.invoke({
                "book_title": book_title,
                "chapter_title": chapter_title,
                "content": content[:3000],
            })
        except Exception as e:
            print(f"[warn] Lesson generation failed: {e}")
            return None

        return BookLesson(
            book_title=book_title,
            chapter_title=chapter_title,
            chapter_index=chapter_index,
            lesson_bullets=output.lesson_bullets,
            reflection_question=output.reflection_question,
        )
