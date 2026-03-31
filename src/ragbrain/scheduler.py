"""APScheduler-based scheduler for automated digest delivery.

Jobs:
  - Morning (8 AM UTC, default):
      * daily_automation_job  — ingest Slack news, send daily briefing,
                                run UpgradePlanner, post proposals to Telegram
      * morning_digest_job    — existing article digest
  - Evening (7 PM UTC, default):
      * architecture_snapshot_job — plain-English snapshot sent to Telegram
      * evening_lesson_job        — existing book lesson
  - 8:30 AM UTC: architecture_review_job (if Slack configured)

The daily_automation_job and architecture_snapshot_job only run when
RAGBRAIN_AUTOMATION_ENABLED=true is set in your .env file.
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


async def _send_proposal_telegram(proposal) -> None:
    """Send a proposal to Telegram with Approve/Skip/Explain inline buttons."""
    token = settings.telegram_bot_token
    chat_id = settings.telegram_chat_id
    if not token or not chat_id:
        return

    from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.constants import ParseMode

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Approve", callback_data=f"approve:{proposal.id}"),
            InlineKeyboardButton("⏭ Skip",    callback_data=f"skip:{proposal.id}"),
            InlineKeyboardButton("🔍 Explain", callback_data=f"explain:{proposal.id}"),
        ]
    ])

    async with Bot(token=token) as bot:
        await bot.send_message(
            chat_id=chat_id,
            text=proposal.telegram_detail(),
            parse_mode=ParseMode.HTML,
            reply_markup=keyboard,
        )


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


def daily_automation_job() -> None:
    """Daily automation loop — runs at 8 AM UTC when automation is enabled.

    Steps:
    1. Ingest recent Slack news into the knowledge base.
    2. Generate a mobile-friendly learning briefing and send to Telegram.
    3. Run the UpgradePlanner to generate 1-3 proposals.
    4. Persist proposals in ProposalStore and send each to Telegram with buttons.
    """
    if not settings.automation_enabled:
        logger.debug("daily_automation_job: automation disabled — skipping.")
        return

    logger.info("Running daily automation job...")

    # ---- Step 1: Ingest Slack news -----------------------------------
    try:
        from ragbrain.ingestion.extractors.slack import SlackExtractor
        from ragbrain.ingestion.pipeline import IngestionPipeline

        extractor = SlackExtractor(fetch_urls=False)
        docs = extractor.extract_recent()
        if docs:
            pipeline = IngestionPipeline()
            for doc in docs:
                pipeline.ingest_document(doc)
            logger.info("Ingested %d Slack news messages.", len(docs))
        else:
            logger.info("No new Slack messages found.")
    except Exception:
        logger.exception("Slack ingestion failed in daily_automation_job")

    # ---- Step 2: Daily briefing -------------------------------------
    try:
        from ragbrain.delivery.slack_delivery import post_briefing
        from ragbrain.pipelines.daily_briefing import generate_daily_briefing

        briefing = generate_daily_briefing()
        post_briefing(briefing)
        logger.info("Daily briefing sent to Slack.")
    except Exception:
        logger.exception("Daily briefing failed in daily_automation_job")

    # ---- Step 3 + 4: Upgrade proposals --------------------------------
    try:
        from ragbrain.pipelines.proposals import Proposal, get_store
        from ragbrain.pipelines.upgrade_planner import get_upgrade_recommendations

        recs = get_upgrade_recommendations()
        if not recs:
            logger.info("UpgradePlanner returned no recommendations.")
            return

        store = get_store()
        token = settings.telegram_bot_token
        chat_id = settings.telegram_chat_id

        for rec in recs[:3]:   # send top 3
            suggestion = rec.get("suggestion", "")
            proposal = Proposal(
                title=suggestion[:80] if suggestion else "Architecture Upgrade",
                description=rec.get("rationale", ""),
                implementation_plan=suggestion,
                component=rec.get("component", ""),
                priority=rec.get("priority", "MEDIUM"),
                news_signal=rec.get("news_signal", ""),
            )
            store.add(proposal)

            from ragbrain.delivery.slack_delivery import post_proposal
            post_proposal(proposal)

        logger.info("Sent %d proposals to Slack.", min(len(recs), 3))
    except Exception:
        logger.exception("Upgrade planner failed in daily_automation_job")


def architecture_snapshot_job() -> None:
    """Send a plain-English architecture snapshot at 7 PM UTC."""
    if not settings.automation_enabled:
        logger.debug("architecture_snapshot_job: automation disabled — skipping.")
        return

    logger.info("Running architecture snapshot job...")
    try:
        from ragbrain.delivery.slack_delivery import post_briefing
        from ragbrain.pipelines.daily_briefing import architecture_snapshot

        snapshot = architecture_snapshot()
        post_briefing(snapshot)
        logger.info("Architecture snapshot sent to Slack.")
    except Exception:
        logger.exception("Architecture snapshot job failed")


def architecture_review_job() -> None:
    """Ingest Slack news and run architecture review."""
    logger.info("Running architecture review job...")
    try:
        from ragbrain.ingestion.extractors.slack import SlackExtractor
        from ragbrain.ingestion.pipeline import IngestionPipeline
        from ragbrain.pipelines.architecture_review import post_to_slack, run_review

        # Step 1: ingest recent Slack messages into the knowledge base
        extractor = SlackExtractor(fetch_urls=False)
        docs = extractor.extract_recent()
        if docs:
            pipeline = IngestionPipeline()
            for doc in docs:
                pipeline.ingest_document(doc)
            logger.info("Ingested %d Slack news messages.", len(docs))

        # Step 2: run the architecture review
        report = run_review(post_slack=False)
        print(report)

        # Step 3: post to Slack if configured
        if settings.slack_bot_token and (settings.slack_post_channel_id or settings.slack_channel_id):
            post_to_slack(report)
            logger.info("Architecture review posted to Slack.")
    except Exception:
        logger.exception("Architecture review job failed")


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

    # Architecture review runs 30 minutes after morning digest
    if settings.slack_bot_token and settings.slack_channel_id:
        review_cron = dict(morning_cron)
        review_cron["minute"] = "30"
        scheduler.add_job(
            architecture_review_job,
            CronTrigger(**review_cron),
            id="architecture_review",
            name="Architecture Review",
            misfire_grace_time=300,
        )

    # ---- Vacation automation jobs (only when enabled) ----------------
    if settings.automation_enabled:
        # Daily automation at 8 AM UTC (same hour as morning digest)
        scheduler.add_job(
            daily_automation_job,
            CronTrigger(**morning_cron),
            id="daily_automation",
            name="Daily Automation (Briefing + Proposals)",
            misfire_grace_time=600,
        )

        # Architecture snapshot at 7 PM UTC
        scheduler.add_job(
            architecture_snapshot_job,
            CronTrigger(**evening_cron),
            id="architecture_snapshot",
            name="Evening Architecture Snapshot",
            misfire_grace_time=300,
        )

    logger.info(
        f"Scheduler started. Morning digest: {settings.morning_cron}, "
        f"Evening lesson: {settings.evening_cron} (UTC)"
    )
    print(f"  Morning digest:     {settings.morning_cron} (UTC)")
    print(f"  Evening lesson:     {settings.evening_cron} (UTC)")
    if settings.slack_bot_token and settings.slack_channel_id:
        print(f"  Arch review:        {review_cron.get('minute', '30')} {morning_cron.get('hour', '8')} * * * (UTC)")
    if settings.automation_enabled:
        print(f"  Daily automation:   {settings.morning_cron} (UTC)  [ENABLED]")
        print(f"  Arch snapshot:      {settings.evening_cron} (UTC)  [ENABLED]")
    else:
        print("  Daily automation:   DISABLED (set RAGBRAIN_AUTOMATION_ENABLED=true to enable)")
    print("  Press Ctrl+C to stop.")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        scheduler.shutdown()
        print("\nScheduler stopped.")
