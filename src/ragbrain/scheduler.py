"""APScheduler-based scheduler for automated digest delivery.

Runs two jobs:
  - Morning (default 8 AM): fetch articles and send digest
  - Evening (default 7 PM): generate and send book lesson

Both jobs push to Telegram if configured, otherwise just log to stdout.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from ragbrain.config import settings
from ragbrain.delivery.formatter import DigestFormatter
from ragbrain.models import Digest
from ragbrain.pipelines.articles import ArticlesPipeline
from ragbrain.pipelines.books import BooksPipeline

logger = logging.getLogger(__name__)
formatter = DigestFormatter()


async def _send_telegram(message: str) -> None:
    """Send a message via Telegram bot if configured."""
    token = settings.telegram_bot_token
    chat_id = settings.telegram_chat_id
    if not token or not chat_id:
        return

    from telegram import Bot
    from telegram.constants import ParseMode

    bot = Bot(token=token)
    # Split message if too long
    limit = 4000
    text = message
    async with bot:
        while len(text) > limit:
            split_at = text.rfind("\n", 0, limit)
            if split_at == -1:
                split_at = limit
            await bot.send_message(chat_id=chat_id, text=text[:split_at], parse_mode=ParseMode.HTML)
            text = text[split_at:].lstrip("\n")
        if text:
            await bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.HTML)


def morning_digest_job() -> None:
    """Fetch articles, summarize, and deliver morning digest."""
    logger.info("Running morning digest job...")
    try:
        pipeline = ArticlesPipeline()
        articles = pipeline.run()

        digest = Digest(date=datetime.utcnow(), articles=articles)
        cli_output = formatter.format_cli(digest)
        print(cli_output)

        if settings.telegram_bot_token and settings.telegram_chat_id:
            telegram_msg = formatter.format_telegram(digest)
            asyncio.run(_send_telegram(telegram_msg))
            logger.info("Morning digest sent to Telegram.")
    except Exception:
        logger.exception("Morning digest job failed")


def evening_lesson_job() -> None:
    """Generate and deliver the daily book lesson."""
    logger.info("Running evening lesson job...")
    try:
        pipeline = BooksPipeline()
        book_lesson = pipeline.get_next_lesson()

        if not book_lesson:
            logger.info("No book lesson available (add books with 'ragbrain ingest book.pdf')")
            return

        digest = Digest(date=datetime.utcnow(), book_lesson=book_lesson)
        cli_output = formatter.format_cli(digest)
        print(cli_output)

        if settings.telegram_bot_token and settings.telegram_chat_id:
            telegram_msg = formatter.format_telegram(digest)
            asyncio.run(_send_telegram(telegram_msg))
            logger.info("Evening lesson sent to Telegram.")
    except Exception:
        logger.exception("Evening lesson job failed")


def _parse_cron(cron_str: str) -> dict:
    """Parse '0 8 * * *' into CronTrigger kwargs."""
    parts = cron_str.split()
    if len(parts) != 5:
        raise ValueError(f"Invalid cron expression: {cron_str!r} (expected 5 fields)")
    return {
        "minute": parts[0],
        "hour": parts[1],
        "day": parts[2],
        "month": parts[3],
        "day_of_week": parts[4],
    }


def run_scheduler() -> None:
    """Start the blocking scheduler for automated delivery."""
    scheduler = BlockingScheduler(timezone="UTC")

    morning_cron = _parse_cron(settings.morning_cron)
    evening_cron = _parse_cron(settings.evening_cron)

    scheduler.add_job(
        morning_digest_job,
        CronTrigger(**morning_cron),
        id="morning_digest",
        name="Morning Article Digest",
        misfire_grace_time=300,
    )
    scheduler.add_job(
        evening_lesson_job,
        CronTrigger(**evening_cron),
        id="evening_lesson",
        name="Evening Book Lesson",
        misfire_grace_time=300,
    )

    logger.info(
        f"Scheduler started. Morning digest: {settings.morning_cron}, "
        f"Evening lesson: {settings.evening_cron} (UTC)"
    )
    print(f"  Morning digest:  {settings.morning_cron} (UTC)")
    print(f"  Evening lesson:  {settings.evening_cron} (UTC)")
    print("  Press Ctrl+C to stop.")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        scheduler.shutdown()
        print("\nScheduler stopped.")
