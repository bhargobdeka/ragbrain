"""Digest formatter for Telegram messages and CLI output.

Telegram supports a limited subset of HTML or MarkdownV2 for formatting.
We use HTML mode here since it's more predictable than MarkdownV2's
aggressive escaping requirements.
"""

from __future__ import annotations

from ragbrain.models import ArticleSummary, BookLesson, Digest


class DigestFormatter:
    """Format Digest objects for different delivery channels."""

    def format_telegram(self, digest: Digest) -> str:
        """Format digest as Telegram HTML message.

        Telegram HTML supports: <b>, <i>, <code>, <pre>, <a href="">.
        """
        parts: list[str] = []
        date_str = digest.date.strftime("%B %d, %Y")

        # Header
        parts.append(f"<b>RAGBrain Daily Digest — {date_str}</b>")

        # Articles section
        if digest.articles:
            parts.append("")
            parts.append("📰 <b>Article Summaries</b>")
            for article in digest.articles[:5]:  # cap at 5 articles
                parts.append("")
                parts.append(self._format_article_telegram(article))

        # Book lesson section
        if digest.book_lesson:
            parts.append("")
            parts.append("📚 <b>Today's Book Lesson</b>")
            parts.append(self._format_book_lesson_telegram(digest.book_lesson))

        if not digest.articles and not digest.book_lesson:
            parts.append("\n<i>No new content today.</i>")

        return "\n".join(parts)

    def format_cli(self, digest: Digest) -> str:
        """Format digest for terminal (plain text with Rich markup)."""
        parts: list[str] = []
        date_str = digest.date.strftime("%B %d, %Y")

        parts.append(f"\n{'='*60}")
        parts.append(f"  RAGBrain Daily Digest — {date_str}")
        parts.append(f"{'='*60}")

        if digest.articles:
            parts.append("\n📰  ARTICLE SUMMARIES\n")
            for i, article in enumerate(digest.articles[:5], 1):
                parts.append(self._format_article_cli(article, i))

        if digest.book_lesson:
            parts.append("\n📚  TODAY'S BOOK LESSON\n")
            parts.append(self._format_book_lesson_cli(digest.book_lesson))

        if not digest.articles and not digest.book_lesson:
            parts.append("\n  No new content today.")

        parts.append(f"\n{'='*60}\n")
        return "\n".join(parts)

    def format_query_result(self, answer: str, sources: list[dict]) -> str:
        """Format a RAG query result for terminal output."""
        parts = [f"\n[bold]Answer:[/bold]\n{answer}"]
        if sources:
            parts.append("\n[bold]Sources:[/bold]")
            for s in sources:
                title = s.get("title", "Unknown")
                url = s.get("url", "")
                parts.append(f"  • {title}" + (f"\n    {url}" if url else ""))
        return "\n".join(parts)

    # ---- Private helpers -------------------------------------------

    def _format_article_telegram(self, article: ArticleSummary) -> str:
        topic_tags = " ".join(f"#{t.replace(' ', '_')}" for t in article.topics[:3])
        source = article.source_name or article.source_url

        lines = [
            f'🔵 <b><a href="{article.source_url}">{article.title}</a></b>',
        ]
        if source:
            lines.append(f"<i>{source}</i>")
        if topic_tags:
            lines.append(topic_tags)

        # Summary bullets
        for bullet in article.summary.split("\n"):
            bullet = bullet.strip().lstrip("•-*").strip()
            if bullet:
                lines.append(f"  → {bullet}")

        lines.append(f"💡 <b>Takeaway:</b> {article.key_takeaway}")
        return "\n".join(lines)

    def _format_book_lesson_telegram(self, lesson: BookLesson) -> str:
        lines = [
            f"<b>{lesson.book_title}</b>",
            f"<i>Chapter {lesson.chapter_index + 1}: {lesson.chapter_title}</i>",
            "",
        ]
        for bullet in lesson.lesson_bullets:
            lines.append(f"  • {bullet}")
        lines.append("")
        lines.append(f"💭 <b>Reflect:</b> <i>{lesson.reflection_question}</i>")
        return "\n".join(lines)

    def _format_article_cli(self, article: ArticleSummary, index: int) -> str:
        source = article.source_name or article.source_url
        lines = [
            f"  [{index}] {article.title}",
            f"      {source}",
        ]
        for bullet in article.summary.split("\n"):
            bullet = bullet.strip().lstrip("•-*").strip()
            if bullet:
                lines.append(f"      → {bullet}")
        lines.append(f"      💡 {article.key_takeaway}")
        lines.append("")
        return "\n".join(lines)

    def _format_book_lesson_cli(self, lesson: BookLesson) -> str:
        lines = [
            f"  Book: {lesson.book_title}",
            f"  Chapter {lesson.chapter_index + 1}: {lesson.chapter_title}",
            "",
        ]
        for bullet in lesson.lesson_bullets:
            lines.append(f"    • {bullet}")
        lines.append("")
        lines.append(f"  💭 Reflect: {lesson.reflection_question}")
        lines.append("")
        return "\n".join(lines)
