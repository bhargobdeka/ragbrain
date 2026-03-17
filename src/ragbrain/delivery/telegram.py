"""Telegram bot delivery and interactive handler.

Bot commands:
  /start     — Welcome message and help
  /digest    — Generate and send today's digest (articles + book lesson)
  /query     — Ask a question: /query What is RLHF?
  /ingest    — Index a URL or local path: /ingest https://...
  /books     — List all registered books and chapter progress
  /help      — Show all commands

The bot runs in polling mode (no webhook required for personal use).
For production, consider switching to webhooks via Application.run_webhook().
"""

from __future__ import annotations

import logging
from datetime import datetime

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from ragbrain.agents.graph import query as rag_query
from ragbrain.config import settings
from ragbrain.delivery.formatter import DigestFormatter
from ragbrain.ingestion.pipeline import IngestionPipeline
from ragbrain.models import Digest
from ragbrain.pipelines.articles import ArticlesPipeline
from ragbrain.pipelines.books import BooksPipeline

logger = logging.getLogger(__name__)
formatter = DigestFormatter()


# ---- Command handlers ------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 <b>Welcome to RAGBrain!</b>\n\n"
        "Your personal document intelligence assistant.\n\n"
        "<b>Commands:</b>\n"
        "/digest — Today's article summaries + book lesson\n"
        "/query &lt;question&gt; — Ask anything from your knowledge base\n"
        "/ingest &lt;url or path&gt; — Index a new document\n"
        "/books — Show book progress\n"
        "/help — Show this message",
        parse_mode=ParseMode.HTML,
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_start(update, context)


async def cmd_digest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("⏳ Generating your digest...", parse_mode=ParseMode.HTML)

    try:
        articles_pipeline = ArticlesPipeline()
        books_pipeline = BooksPipeline()

        articles = articles_pipeline.run()
        book_lesson = books_pipeline.get_next_lesson()

        digest = Digest(
            date=datetime.utcnow(),
            articles=articles,
            book_lesson=book_lesson,
            delivery_channel="telegram",
        )

        message = formatter.format_telegram(digest)
        # Telegram messages are capped at 4096 chars; split if needed
        for chunk in _split_message(message):
            await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.exception("Digest failed")
        await update.message.reply_text(f"❌ Error generating digest: {e}")


async def cmd_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /query &lt;your question&gt;", parse_mode=ParseMode.HTML)
        return

    question = " ".join(context.args)
    await update.message.reply_text(f"🔍 Searching for: <i>{question}</i>...", parse_mode=ParseMode.HTML)

    try:
        result = rag_query(question)
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


async def cmd_ingest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            "Usage: /ingest &lt;url or file path&gt;\n"
            "Example: /ingest https://example.substack.com/p/article",
            parse_mode=ParseMode.HTML,
        )
        return

    source = context.args[0]
    await update.message.reply_text(f"⏳ Ingesting: <code>{source}</code>...", parse_mode=ParseMode.HTML)

    try:
        pipeline = IngestionPipeline()
        count = pipeline.ingest(source)
        await update.message.reply_text(f"✅ Indexed <b>{count}</b> chunks from:\n<code>{source}</code>", parse_mode=ParseMode.HTML)
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

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("digest", cmd_digest))
    app.add_handler(CommandHandler("query", cmd_query))
    app.add_handler(CommandHandler("ingest", cmd_ingest))
    app.add_handler(CommandHandler("books", cmd_books))

    logger.info("RAGBrain Telegram bot started. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)
