"""Tests for the digest formatter."""

from __future__ import annotations

from datetime import datetime

import pytest

from ragbrain.delivery.formatter import DigestFormatter
from ragbrain.models import ArticleSummary, BookLesson, Digest


@pytest.fixture
def sample_article():
    return ArticleSummary(
        title="Understanding RLHF",
        source_url="https://example.substack.com/p/rlhf",
        source_name="example.substack.com",
        summary="• RLHF trains models using human preferences\n• Reward models score outputs\n• PPO optimizes against the reward",
        key_takeaway="RLHF is the key technique behind ChatGPT's conversational quality.",
        relevance_score=9,
        topics=["RLHF", "LLM training"],
    )


@pytest.fixture
def sample_lesson():
    return BookLesson(
        book_title="Designing ML Systems",
        chapter_title="Data Pipelines",
        chapter_index=3,
        lesson_bullets=[
            "Data pipelines are the backbone of any ML system",
            "Batch vs streaming depends on latency requirements",
            "Data validation catches distribution shift early",
        ],
        reflection_question="How would you redesign your current ETL pipeline with these principles?",
    )


class TestDigestFormatter:
    def setup_method(self):
        self.fmt = DigestFormatter()

    def test_telegram_format_has_header(self, sample_article):
        digest = Digest(
            date=datetime(2026, 3, 16),
            articles=[sample_article],
        )
        output = self.fmt.format_telegram(digest)
        assert "RAGBrain Daily Digest" in output
        assert "March 16, 2026" in output

    def test_telegram_format_includes_article(self, sample_article):
        digest = Digest(date=datetime(2026, 3, 16), articles=[sample_article])
        output = self.fmt.format_telegram(digest)
        assert "Understanding RLHF" in output
        assert "RLHF" in output

    def test_telegram_format_includes_book_lesson(self, sample_lesson):
        digest = Digest(date=datetime(2026, 3, 16), book_lesson=sample_lesson)
        output = self.fmt.format_telegram(digest)
        assert "Designing ML Systems" in output
        assert "Data Pipelines" in output
        assert "Reflect" in output

    def test_cli_format_has_separator(self, sample_article):
        digest = Digest(date=datetime(2026, 3, 16), articles=[sample_article])
        output = self.fmt.format_cli(digest)
        assert "=" * 20 in output
        assert "ARTICLE" in output

    def test_empty_digest_shows_no_content_message(self):
        digest = Digest(date=datetime(2026, 3, 16))
        telegram_output = self.fmt.format_telegram(digest)
        assert "No new content" in telegram_output

    def test_split_message_short(self):
        from ragbrain.delivery.telegram import _split_message

        msg = "Short message"
        parts = _split_message(msg)
        assert parts == ["Short message"]

    def test_split_message_long(self):
        from ragbrain.delivery.telegram import _split_message

        msg = "Line\n" * 1000
        parts = _split_message(msg, limit=100)
        assert len(parts) > 1
        assert all(len(p) <= 100 for p in parts)
