"""Telegram bot delivery and interactive handler.

Bot commands:
  /start          — Welcome message and help
  /digest         — Generate and send today's digest (articles + book lesson)
  /query          — Ask a question: /query What is RLHF?
  /ingest         — Index a URL or local path: /ingest https://...
  /books          — List all registered books and chapter progress
  /status         — Show pending proposals and their state
  /architecture   — Generate a plain-English architecture snapshot on demand
  /help           — Show all commands

Plain text messages (without a /) are routed to the RAG query pipeline,
so you can ask questions directly on your phone without typing /query.

Proposal approval flow:
  - Each proposal is sent with Approve / Skip / Explain inline buttons.
  - Tapping Approve triggers AutoImplementer on the Mac; result is sent back.
  - Tapping Skip marks the proposal as skipped in ProposalStore.
  - Tapping Explain shows the full implementation plan.

The bot runs in polling mode (no webhook required for personal use).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from pathlib import Path

from ragbrain.agents.graph import query as rag_query
from ragbrain.config import settings
from ragbrain.delivery.formatter import DigestFormatter
from ragbrain.ingestion.pipeline import IngestionPipeline
from ragbrain.models import Digest
from ragbrain.pipelines.articles import ArticlesPipeline
from ragbrain.pipelines.books import BooksPipeline
from ragbrain.pipelines.proposals import Proposal, get_store

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NEWS_DIR = _REPO_ROOT / "news"
_DRAFTS_DIR = _REPO_ROOT / "_planning" / "drafts"
_BRIEFING_SIGNATURES = ("Daily AI Briefing", "───", "Tuk's Daily")

logger = logging.getLogger(__name__)


async def notify_telegram_html(message: str) -> None:
    """Send an HTML message to the configured Telegram chat (scheduler / automation).

    Splits on 4000-char Telegram limits. No-op if bot token or chat id is missing.
    """
    token = settings.telegram_bot_token
    chat_id = settings.telegram_chat_id
    if not token or not chat_id:
        return

    from telegram import Bot
    from telegram.constants import ParseMode

    limit = 4000
    text = message
    bot = Bot(token=token)
    async with bot:
        while len(text) > limit:
            split_at = text.rfind("\n", 0, limit)
            if split_at == -1:
                split_at = limit
            await bot.send_message(
                chat_id=chat_id, text=text[:split_at], parse_mode=ParseMode.HTML
            )
            text = text[split_at:].lstrip("\n")
        if text:
            await bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.HTML)


def notify_telegram_html_sync(message: str) -> None:
    """Sync wrapper for :func:`notify_telegram_html` (for non-async call sites)."""
    asyncio.run(notify_telegram_html(message))


formatter = DigestFormatter()


# ---- Inline keyboard helpers ---------------------------------------------

def _proposal_keyboard(proposal_id: str) -> InlineKeyboardMarkup:
    """Build Approve / Skip / Explain inline keyboard for a proposal."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Approve", callback_data=f"approve:{proposal_id}"),
            InlineKeyboardButton("⏭ Skip",    callback_data=f"skip:{proposal_id}"),
            InlineKeyboardButton("🔍 Explain", callback_data=f"explain:{proposal_id}"),
        ]
    ])


# ---- Proposal sending helper ---------------------------------------------

async def send_proposal(bot, chat_id: int, proposal: Proposal) -> None:
    """Send a single proposal message with Approve/Skip/Explain buttons."""
    text = proposal.telegram_detail()
    await bot.send_message(
        chat_id=chat_id,
        text=text,
        parse_mode=ParseMode.HTML,
        reply_markup=_proposal_keyboard(proposal.id),
    )


# ---- Command handlers ------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Log the chat ID so it's easy to find for RAGBRAIN_TELEGRAM_CHAT_ID
    chat_id = update.effective_chat.id
    logger.info("Chat ID hint — add to .env: RAGBRAIN_TELEGRAM_CHAT_ID=%s", chat_id)
    print(f"\n  Chat ID hint: RAGBRAIN_TELEGRAM_CHAT_ID={chat_id}\n")

    await update.message.reply_text(
        "👋 <b>Welcome to RAGBrain!</b>\n\n"
        "Your personal document intelligence assistant.\n\n"
        "<b>Commands:</b>\n"
        "/digest — Today's article summaries + book lesson\n"
        "/query &lt;question&gt; — Ask anything from your knowledge base\n"
        "/ingest &lt;url or path&gt; — Index a new document\n"
        "/books — Show book progress\n"
        "/status — View upgrade proposals\n"
        "/architecture — Current architecture snapshot\n"
        "/help — Show this message\n\n"
        "<i>Tip: Just type a question (no /query needed) to search your KB.</i>",
        parse_mode=ParseMode.HTML,
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_start(update, context)


async def cmd_digest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("⏳ Generating your digest...", parse_mode=ParseMode.HTML)

    try:
        articles_pipeline = ArticlesPipeline()
        books_pipeline = BooksPipeline()

        try:
            articles = articles_pipeline.run()
        finally:
            articles_pipeline.close()
        book_lesson = books_pipeline.get_next_lesson()

        digest = Digest(
            date=datetime.utcnow(),
            articles=articles,
            book_lesson=book_lesson,
            delivery_channel="telegram",
        )

        message = formatter.format_telegram(digest)
        for chunk in _split_message(message):
            await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.exception("Digest failed")
        await update.message.reply_text(f"❌ Error generating digest: {e}")


async def cmd_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            "Usage: /query &lt;your question&gt;\n\n"
            "<i>Or just type your question without any command.</i>",
            parse_mode=ParseMode.HTML,
        )
        return

    question = " ".join(context.args)
    await _do_query(update, question)


async def cmd_ingest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            "Usage: /ingest &lt;url or file path&gt;\n"
            "Example: /ingest https://example.substack.com/p/article",
            parse_mode=ParseMode.HTML,
        )
        return

    source = context.args[0]
    await update.message.reply_text(
        f"⏳ Ingesting: <code>{source}</code>...", parse_mode=ParseMode.HTML
    )

    try:
        pipeline = IngestionPipeline()
        count = pipeline.ingest(source)
        await update.message.reply_text(
            f"✅ Indexed <b>{count}</b> chunks from:\n<code>{source}</code>",
            parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        logger.exception("Ingest failed")
        await update.message.reply_text(f"❌ Ingest failed: {e}")


async def cmd_books(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pipeline = BooksPipeline()
    keys = pipeline.get_all_book_keys()

    if not keys:
        await update.message.reply_text(
            "No books registered yet.\n"
            "Use <code>ragbrain ingest path/to/book.pdf</code> to add one.",
            parse_mode=ParseMode.HTML,
        )
        return

    lines = ["<b>📚 Registered Books:</b>", ""]
    for key in keys:
        lines.append(f"  • <code>{key}</code>")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show pending proposals and their current status."""
    store = get_store()
    status_text = store.status_summary()
    pending = store.list_pending()

    await update.message.reply_text(status_text, parse_mode=ParseMode.HTML)

    # Resend any pending proposals with action buttons
    if pending:
        await update.message.reply_text(
            f"<b>Pending ({len(pending)}) — tap to act:</b>",
            parse_mode=ParseMode.HTML,
        )
        for proposal in pending[:5]:   # max 5 to avoid spam
            await send_proposal(context.bot, update.effective_chat.id, proposal)


async def cmd_architecture(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate and send a plain-English architecture snapshot."""
    await update.message.reply_text(
        "⏳ Generating architecture snapshot...", parse_mode=ParseMode.HTML
    )
    try:
        from ragbrain.pipelines.daily_briefing import architecture_snapshot
        snapshot = architecture_snapshot()
        for chunk in _split_message(snapshot):
            await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.exception("Architecture snapshot failed")
        await update.message.reply_text(f"❌ Error: {e}")


# ---- Briefing detection helpers ------------------------------------------

def _strip_preamble(text: str) -> str:
    """Remove Tuk's opener sentence before the first ─── separator.

    The raw forwarded message starts with a conversational line like
    "Good — solid signal from all three searches. Here's today's briefing:"
    followed by a blank line and the first ─── divider.  We drop everything
    before (and not including) the first ─── so the saved file starts cleanly.
    """
    idx = text.find("───")
    return text[idx:].strip() if idx != -1 else text.strip()


def _is_forwarded_briefing(message) -> bool:
    """Return True if the message is a forwarded OpenClaw/Tuk briefing."""
    is_forwarded = bool(
        getattr(message, "forward_date", None)
        or getattr(message, "forward_origin", None)
    )
    if not is_forwarded:
        return False
    text = message.text or ""
    return any(sig in text for sig in _BRIEFING_SIGNATURES)


async def _save_briefing(update: Update) -> None:
    """Save a forwarded briefing to news/YYYY-MM-DD.md and confirm."""
    message = update.message
    text = _strip_preamble(message.text or "")

    # Use the original message date (forward_date) so the filename matches
    # the day the briefing was actually published, not when it was forwarded.
    origin_date = getattr(message, "forward_date", None) or message.date
    date_str = origin_date.strftime("%Y-%m-%d")

    _NEWS_DIR.mkdir(parents=True, exist_ok=True)
    news_dest = _NEWS_DIR / f"{date_str}.md"
    news_dest.write_text(text, encoding="utf-8")
    logger.info("Saved briefing to %s", news_dest)

    # Generate X thread draft (async-safe: single LLM call, fast model)
    draft_msg = ""
    try:
        from ragbrain.pipelines.x_thread import format_draft_md, generate_x_thread

        tweets = generate_x_thread(text)
        _DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
        draft_dest = _DRAFTS_DIR / f"{date_str}-x-draft.md"
        draft_dest.write_text(format_draft_md(tweets, date_str), encoding="utf-8")
        logger.info("X thread draft saved to %s (%d tweets)", draft_dest, len(tweets))
        draft_msg = (
            f"\n\n🐦 X thread draft → <code>_planning/drafts/{date_str}-x-draft.md</code>"
            f" ({len(tweets)} tweets)"
        )
    except Exception:
        logger.exception("X thread draft generation failed (news still saved)")
        draft_msg = "\n\n⚠️ X draft generation failed — check logs."

    await update.message.reply_text(
        f"📰 Briefing saved to <code>news/{date_str}.md</code>{draft_msg}",
        parse_mode=ParseMode.HTML,
    )


# ---- Message handler (plain text → RAG query) ----------------------------

async def handle_plain_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route plain text messages to the RAG query pipeline.

    Forwarded OpenClaw/Tuk briefings are intercepted first and saved to
    news/YYYY-MM-DD.md without going through the RAG pipeline.
    """
    if not update.message:
        return
    if _is_forwarded_briefing(update.message):
        await _save_briefing(update)
        return
    text = update.message.text or ""
    if not text.strip():
        return
    await _do_query(update, text.strip())


# ---- Callback query handler (button taps) --------------------------------

async def handle_callback_query(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle inline button taps from proposal messages.

    Telegram callback queries expire after 60 seconds.  If the user taps a
    button on an older message, query.answer() raises BadRequest.  We catch
    that and continue processing the action anyway — the important part is
    updating ProposalStore and running AutoImplementer, not the spinner ack.
    """
    query = update.callback_query

    # Acknowledge the button tap (stops the spinner on the user's phone).
    # Swallow BadRequest/NetworkError so a stale query doesn't abort the action.
    try:
        await query.answer()
    except Exception as e:
        logger.debug("query.answer() failed (query may be expired): %s", e)

    data: str = query.data or ""
    if ":" not in data:
        return

    action, proposal_id = data.split(":", 1)
    chat_id = query.message.chat_id if query.message else settings.telegram_chat_id
    store = get_store()
    proposal = store.get(proposal_id)

    if proposal is None:
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"⚠️ Proposal <code>{proposal_id}</code> not found.",
            parse_mode=ParseMode.HTML,
        )
        return

    # Remove buttons to prevent double-taps (best-effort — may fail on old messages).
    try:
        await query.edit_message_reply_markup(reply_markup=None)
    except Exception:
        pass

    if action == "skip":
        store.skip(proposal_id)
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"⏭ Skipped: <i>{proposal.title}</i>",
            parse_mode=ParseMode.HTML,
        )

    elif action == "explain":
        from ragbrain.pipelines.proposals import Proposal as _P
        p = proposal
        explanation = (
            f"<b>{p.title}</b>\n\n"
            f"<i>Description:</i>\n{p.description}\n\n"
            f"<i>Implementation plan:</i>\n{p.implementation_plan}\n\n"
            f"<i>News signal:</i>\n{p.news_signal}"
        )
        for chunk in _split_message(explanation):
            await context.bot.send_message(
                chat_id=chat_id,
                text=chunk,
                parse_mode=ParseMode.HTML,
            )
        # Resend with buttons so user can still approve
        await send_proposal(context.bot, chat_id, proposal)

    elif action == "approve":
        store.approve(proposal_id)
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"✅ Approved: <i>{proposal.title}</i>\n\n"
                "⏳ Starting auto-implementation on your Mac…"
            ),
            parse_mode=ParseMode.HTML,
        )

        # Run AutoImplementer in a thread to avoid blocking the bot event loop
        loop = asyncio.get_event_loop()
        bot = context.bot

        async def _run_impl() -> None:
            from ragbrain.pipelines.auto_implement import implement_proposal
            from ragbrain.pipelines.proposals import get_store as _gs

            result = await loop.run_in_executor(
                None, implement_proposal, proposal
            )

            # Update proposal store
            _store = _gs()
            if result.success:
                _store.mark_implemented(
                    proposal_id,
                    commit_sha=result.commit_sha,
                    summary=result.summary,
                )
            else:
                _store.mark_failed(proposal_id, reason=result.summary)

            # Send result back to Telegram
            for chunk in _split_message(result.telegram_message):
                await bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode=ParseMode.HTML,
                )

        asyncio.ensure_future(_run_impl())


# ---- Shared query helper -------------------------------------------------

async def _do_query(update: Update, question: str) -> None:
    await update.message.reply_text(
        f"🔍 <i>Searching: {question[:100]}…</i>",
        parse_mode=ParseMode.HTML,
    )
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, rag_query, question)
        answer = result.get("answer", "No answer generated.")
        sources = result.get("sources", [])

        response_parts = [f"<b>Answer:</b>\n{answer}"]
        if sources:
            response_parts.append("\n<b>Sources:</b>")
            for s in sources[:3]:
                title = s.get("title", "Source")
                url = s.get("url", "")
                if url:
                    response_parts.append(f'  • <a href="{url}">{title}</a>')
                else:
                    response_parts.append(f"  • {title}")

        message = "\n".join(response_parts)
        for chunk in _split_message(message):
            await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.exception("Query failed")
        await update.message.reply_text(f"❌ Error: {e}")


# ---- Helpers ---------------------------------------------------------

def _split_message(text: str, limit: int = 4000) -> list[str]:
    """Split a long message into Telegram-safe chunks."""
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    while len(text) > limit:
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        parts.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    if text:
        parts.append(text)
    return parts


# ---- Global error handler -------------------------------------------

async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global error handler — logs cleanly instead of printing full tracebacks.

    Suppresses:
      - NetworkError / ConnectError: transient network blip, bot auto-retries
      - BadRequest 'Query is too old': user tapped a button after 60s timeout
    These are expected in long-running vacation mode and should not alarm.
    """
    from telegram.error import BadRequest, NetworkError

    err = context.error
    if isinstance(err, NetworkError):
        logger.debug("Telegram network blip (auto-retry): %s", err)
        return
    if isinstance(err, BadRequest) and "too old" in str(err).lower():
        logger.debug("Stale callback query ignored: %s", err)
        return
    # Log anything else as a warning (not a full traceback)
    logger.warning("Unhandled Telegram error: %s", err)


# ---- Bot entry point -------------------------------------------------

def run_bot() -> None:
    """Start the Telegram bot in polling mode."""
    token = settings.telegram_bot_token
    if not token:
        raise ValueError(
            "RAGBRAIN_TELEGRAM_BOT_TOKEN is not set. "
            "Add it to your .env file. Get a token from @BotFather on Telegram."
        )

    app = Application.builder().token(token).build()

    # Command handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("digest", cmd_digest))
    app.add_handler(CommandHandler("query", cmd_query))
    app.add_handler(CommandHandler("ingest", cmd_ingest))
    app.add_handler(CommandHandler("books", cmd_books))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("architecture", cmd_architecture))

    # Inline button callback handler
    app.add_handler(CallbackQueryHandler(handle_callback_query))

    # Global error handler — suppresses noisy network/stale-query tracebacks
    app.add_error_handler(handle_error)

    # Plain text → RAG query (must be last to not catch commands)
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_plain_text)
    )

    logger.info("RAGBrain Telegram bot started. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)
