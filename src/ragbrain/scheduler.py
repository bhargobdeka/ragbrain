"""APScheduler-based scheduler for automated digest delivery.

Jobs:
  - Morning (8 AM UTC, default):
      * daily_automation_job  — ingest RSS articles, send daily briefing,
                                run UpgradePlanner, post proposals to Telegram
      * morning_digest_job    — existing article digest
  - Evening (7 PM UTC, default):
      * architecture_snapshot_job — plain-English snapshot sent to Telegram (automation on)
      * evening_lesson_job        — book lesson to Telegram (only if
        RAGBRAIN_EVENING_BOOK_LESSON_ENABLED=true)
  - 8:30 AM UTC: architecture_review_job (only if explicitly enabled)

The daily_automation_job and architecture_snapshot_job only run when
RAGBRAIN_AUTOMATION_ENABLED=true is set in your .env file.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from ragbrain.config import settings
from ragbrain.delivery.formatter import DigestFormatter
from ragbrain.models import Digest
from ragbrain.pipelines.articles import ArticlesPipeline
from ragbrain.pipelines.books import BooksPipeline

logger = logging.getLogger(__name__)
formatter = DigestFormatter()


def _notify_status(message: str) -> None:
    """Best-effort status notification via Telegram when configured."""
    try:
        if settings.telegram_bot_token and settings.telegram_chat_id:
            asyncio.run(_send_telegram(message))
        else:
            logger.info("Status (no Telegram): %s", message)
    except Exception:
        logger.exception("Failed to send status notification: %s", message)


def scheduler_heartbeat_job() -> None:
    """Periodic heartbeat so we can confirm scheduler liveness in logs."""
    logger.info("Scheduler heartbeat: alive at %s", datetime.utcnow().isoformat())


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
    from ragbrain.delivery.telegram import notify_telegram_html

    await notify_telegram_html(message)


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


def _planner_subprocess_worker(queue) -> None:
    """Subprocess worker: runs the planner and sends results via Queue."""
    try:
        from ragbrain.pipelines.upgrade_planner import get_upgrade_recommendations
        queue.put(("ok", get_upgrade_recommendations()))
    except Exception as exc:
        queue.put(("error", str(exc)))


def _run_planner_subprocess(timeout: int, retries: int = 2) -> tuple[list, str | None]:
    """Run upgrade planner in a subprocess with a hard kill timeout.

    Uses multiprocessing.Process so a hung planner is SIGKILL'd after
    `timeout` seconds — threads cannot be force-killed in Python.
    Retries once by default for transient network/API timeouts.
    """
    import multiprocessing

    last_reason: str | None = None

    for attempt in range(1, retries + 1):
        ctx = multiprocessing.get_context("spawn")
        q: multiprocessing.Queue = ctx.Queue()
        p = ctx.Process(target=_planner_subprocess_worker, args=(q,), daemon=True)
        p.start()
        p.join(timeout=timeout)

        if p.is_alive():
            p.kill()
            p.join(timeout=5)
            last_reason = f"timeout after {timeout}s (attempt {attempt}/{retries})"
            logger.warning("UpgradePlanner %s", last_reason)
            continue

        if not q.empty():
            status, payload = q.get_nowait()
            if status == "ok" and isinstance(payload, list):
                return payload, None
            last_reason = str(payload)
            logger.warning("UpgradePlanner error (attempt %d/%d): %s", attempt, retries, last_reason)
            continue

        last_reason = f"no result returned (attempt {attempt}/{retries})"
        logger.warning("UpgradePlanner %s", last_reason)

    return [], last_reason


def daily_automation_job() -> None:
    """Daily automation loop — runs at 8 AM UTC when automation is enabled.

    Steps:
    1. Fetch and ingest recent RSS articles into the knowledge base.
    2. Generate a mobile-friendly learning briefing and send to Telegram.
    3. Run the UpgradePlanner to generate 1-3 proposals.
    4. Persist proposals in ProposalStore and send each to Telegram with buttons.
    """
    if not settings.automation_enabled:
        logger.debug("daily_automation_job: automation disabled — skipping.")
        return

    logger.info("Running daily automation job...")
    _notify_status(
        "RAGBrain: Daily automation started. Ingesting RSS articles, generating briefing, and planning proposals."
    )

    # ---- Step 1: RSS articles → knowledge base ----------------------
    try:
        from ragbrain.pipelines.articles import ArticlesPipeline

        ap = ArticlesPipeline()
        try:
            summaries = ap.run(also_ingest=True)
            logger.info("Articles pipeline: %d summaries, KB updated.", len(summaries))
        finally:
            ap.close()
    except Exception:
        logger.exception("RSS/article ingestion failed in daily_automation_job")

    # ---- Step 2: Daily briefing -------------------------------------
    try:
        import html
        import re as _re
        from ragbrain.pipelines.daily_briefing import (
            generate_daily_briefing,
            generate_social_posts,
            get_daily_inputs,
        )

        news_content, arch_summary = get_daily_inputs()
        briefing = generate_daily_briefing(
            news_content=news_content,
            architecture_summary=arch_summary,
        )
        social_posts = generate_social_posts(
            news_content=news_content,
            architecture_summary=arch_summary,
        )

        # Prefer Telegram (richer mobile experience)
        if settings.telegram_bot_token and settings.telegram_chat_id:
            plain = _re.sub(r"<[^>]+>", "", briefing)
            asyncio.run(_send_telegram(plain))
            if social_posts:
                social_msg = (
                    "--- LinkedIn Draft ---\n"
                    f"{social_posts.get('linkedin', '').strip()}\n\n"
                    "--- X/Twitter Thread ---\n"
                    f"{social_posts.get('twitter', '').strip()}"
                )
                asyncio.run(_send_telegram(html.escape(social_msg)))
            logger.info("Daily briefing sent to Telegram.")
        else:
            logger.warning(
                "Daily briefing generated but Telegram not configured — nowhere to send."
            )
    except Exception:
        logger.exception("Daily briefing failed in daily_automation_job")
        _notify_status("RAGBrain: Daily automation warning — briefing generation failed. Check logs.")

    # ---- Step 3 + 4: Upgrade proposals --------------------------------
    try:
        from ragbrain.pipelines.proposals import Proposal, get_store

        _PLANNER_TIMEOUT = 180  # 3 minutes hard kill via subprocess
        recs, planner_reason = _run_planner_subprocess(_PLANNER_TIMEOUT, retries=2)

        if not recs:
            logger.info("UpgradePlanner returned no recommendations.")
            if planner_reason:
                fallback = (
                    "Planner fallback: Upgrade recommendations unavailable this run "
                    f"({planner_reason}). Briefing was still delivered."
                )
                _notify_status(fallback)
            _notify_status("RAGBrain: Daily automation finished with no proposals.")
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

            if settings.telegram_bot_token and settings.telegram_chat_id:
                asyncio.run(_send_proposal_telegram(proposal))
            else:
                logger.warning("Proposal %s created but Telegram not configured.", proposal.id)

        logger.info("Sent %d proposals to delivery channel.", min(len(recs), 3))
        _notify_status(
            f"RAGBrain: Daily automation finished successfully. Sent {min(len(recs), 3)} proposal(s)."
        )
    except Exception:
        logger.exception("Upgrade planner failed in daily_automation_job")
        _notify_status("RAGBrain: Daily automation warning — planner failed. Briefing was still delivered.")


def architecture_snapshot_job() -> None:
    """Send a plain-English architecture snapshot at evening cron (UTC)."""
    if not settings.automation_enabled:
        logger.debug("architecture_snapshot_job: automation disabled — skipping.")
        return

    logger.info("Running architecture snapshot job...")
    try:
        from ragbrain.pipelines.daily_briefing import architecture_snapshot

        snapshot = architecture_snapshot()
        if settings.telegram_bot_token and settings.telegram_chat_id:
            asyncio.run(_send_telegram(snapshot))
            logger.info("Architecture snapshot sent to Telegram.")
        else:
            logger.warning("Architecture snapshot generated but Telegram not configured.")
    except Exception:
        logger.exception("Architecture snapshot job failed")


def news_commit_job() -> None:
    """Auto-commit any new daily briefing files in news/ (runs nightly at 9 PM UTC).

    Checks for untracked or modified files under news/*.md, commits them with a
    descriptive message, and sends a Telegram confirmation.  Safe to run even
    when there is nothing new — it simply exits without committing.
    """
    repo_root = Path(__file__).resolve().parents[3]
    news_dir = repo_root / "news"

    if not news_dir.exists():
        return

    # Find any *.md files in news/ that are untracked or modified
    result = subprocess.run(
        ["git", "status", "--porcelain", "news/"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    changed_lines = [l for l in result.stdout.splitlines() if l.strip()]
    if not changed_lines:
        logger.info("news_commit_job: nothing new in news/ — skipping commit.")
        return

    dated_files = sorted(
        p.name for p in news_dir.glob("????-??-??.md")
        if any(p.name in line for line in changed_lines)
    )
    dates_label = ", ".join(f.replace(".md", "") for f in dated_files) or "updates"
    commit_msg = f"news: {dates_label}"

    try:
        subprocess.run(["git", "add", "news/"], cwd=repo_root, check=True)
        subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("news_commit_job: committed — %s", commit_msg)
        _notify_status(
            f"📰 News auto-committed: <code>{commit_msg}</code>\n"
            f"Files: {', '.join(dated_files)}"
        )
    except subprocess.CalledProcessError as exc:
        logger.exception("news_commit_job: git commit failed")
        _notify_status(f"⚠️ News auto-commit failed: {exc.stderr or exc}")


def architecture_review_job() -> None:
    """Run architecture review (RSS news + gap analysis); post to Telegram if configured."""
    logger.info("Running architecture review job...")
    try:
        from ragbrain.pipelines.architecture_review import run_review

        post_tg = bool(settings.telegram_bot_token and settings.telegram_chat_id)
        report = run_review(post_slack=False, post_telegram=post_tg)
        print(report)
        if post_tg:
            logger.info("Architecture review posted to Telegram.")
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
    if settings.evening_book_lesson_enabled:
        scheduler.add_job(
            evening_lesson_job,
            CronTrigger(**evening_cron),
            id="evening_lesson",
            name="Evening Book Lesson",
            misfire_grace_time=300,
        )
    scheduler.add_job(
        scheduler_heartbeat_job,
        CronTrigger(minute="*/30"),
        id="scheduler_heartbeat",
        name="Scheduler Heartbeat",
        misfire_grace_time=120,
    )

    # Architecture review runs 30 minutes after morning digest only when enabled.
    if settings.architecture_review_enabled:
        review_cron = dict(morning_cron)
        review_cron["minute"] = "30"
        scheduler.add_job(
            architecture_review_job,
            CronTrigger(**review_cron),
            id="architecture_review",
            name="Architecture Review",
            misfire_grace_time=300,
        )

    # News auto-commit — runs every night at 9 PM UTC regardless of other flags.
    scheduler.add_job(
        news_commit_job,
        CronTrigger(hour=21, minute=0),
        id="news_commit",
        name="News Auto-Commit",
        misfire_grace_time=3600,
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
        f"Evening book lesson: {'enabled ' + settings.evening_cron if settings.evening_book_lesson_enabled else 'disabled'} (UTC)"
    )
    print(f"  Morning digest:     {settings.morning_cron} (UTC)")
    print("  News auto-commit:   0 21 * * * (UTC, 9 PM)")
    if settings.evening_book_lesson_enabled:
        print(f"  Evening book lesson: {settings.evening_cron} (UTC)  [ENABLED]")
    else:
        print("  Evening book lesson: DISABLED (set RAGBRAIN_EVENING_BOOK_LESSON_ENABLED=true)")
    if settings.architecture_review_enabled:
        _rc = dict(morning_cron)
        _rc["minute"] = "30"
        print(f"  Arch review:        {_rc.get('minute', '30')} {_rc.get('hour', '8')} * * * (UTC)")
    else:
        print("  Arch review:        DISABLED (set RAGBRAIN_ARCHITECTURE_REVIEW_ENABLED=true)")
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
