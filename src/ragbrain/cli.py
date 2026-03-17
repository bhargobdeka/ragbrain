"""RAGBrain CLI entry point.

Commands:
  ragbrain ingest <path_or_url>   Index a document or URL
  ragbrain query  <question>      Ask a question from your knowledge base
  ragbrain digest                 Generate and print today's digest
  ragbrain serve                  Start the Telegram bot (interactive)
  ragbrain schedule               Start the automated scheduler
  ragbrain fetch-articles         Fetch and summarize articles now
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    name="ragbrain",
    help="RAGBrain — document intelligence with agentic RAG",
    add_completion=False,
    pretty_exceptions_enable=False,
)

console = Console()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


# ---- ingest ----------------------------------------------------------

@app.command()
def ingest(
    source: str = typer.Argument(..., help="File path (.pdf) or URL to ingest"),
    user_id: str = typer.Option(None, "--user-id", "-u", help="Namespace for multi-tenant use"),
) -> None:
    """Index a document or URL into the knowledge base."""
    from ragbrain.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline(user_id=user_id)

    # Check if it looks like a PDF to handle books specially
    if source.lower().endswith(".pdf") and Path(source).exists():
        ingest_as_book = typer.confirm("Register this PDF as a book for daily lessons?", default=False)
        if ingest_as_book:
            from ragbrain.pipelines.books import BooksPipeline

            books = BooksPipeline(user_id=user_id)
            count = books.ingest_book(source)
            console.print(f"[green]✓ Book registered and indexed ({count} chunks)[/green]")
            return

    count = pipeline.ingest(source)
    if count == 0:
        console.print("[yellow]No chunks indexed. Check the source is accessible.[/yellow]")
        raise typer.Exit(1)


# ---- query -----------------------------------------------------------

@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    user_id: str = typer.Option(None, "--user-id", "-u", help="Namespace for multi-tenant use"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show retrieval details"),
) -> None:
    """Ask a question from your indexed knowledge base."""
    from ragbrain.agents.graph import query as rag_query
    with console.status("[cyan]Searching your knowledge base...[/cyan]"):
        result = rag_query(question, user_id=user_id)

    answer = result.get("answer", "")
    sources = result.get("sources", [])

    if not answer:
        console.print("[yellow]No answer generated. Make sure you have indexed some documents.[/yellow]")
        raise typer.Exit(1)

    console.print()
    console.print(Panel(answer, title="[bold]Answer[/bold]", border_style="green"))

    if sources:
        console.print("\n[bold]Sources:[/bold]")
        for s in sources:
            title = s.get("title", "Unknown")
            url = s.get("url", "")
            console.print(f"  • [cyan]{title}[/cyan]" + (f"\n    [dim]{url}[/dim]" if url else ""))

    if verbose:
        console.print(f"\n[dim]Retrieval attempts: {result.get('retrieval_attempts', 1)}[/dim]")
        console.print(f"[dim]Hallucination check: {result.get('hallucination_check', 'n/a')}[/dim]")


# ---- digest ----------------------------------------------------------

@app.command()
def digest(
    articles_only: bool = typer.Option(False, "--articles-only", help="Only fetch articles"),
    lesson_only: bool = typer.Option(False, "--lesson-only", help="Only generate book lesson"),
    book: str = typer.Option(None, "--book", "-b", help="Specific book key for lesson"),
) -> None:
    """Generate and print today's digest."""
    from ragbrain.delivery.formatter import DigestFormatter
    from ragbrain.models import Digest
    from ragbrain.pipelines.articles import ArticlesPipeline
    from ragbrain.pipelines.books import BooksPipeline

    fmt = DigestFormatter()
    articles = []
    book_lesson = None

    if not lesson_only:
        with console.status("[cyan]Fetching articles...[/cyan]"):
            try:
                articles = ArticlesPipeline().run()
            except Exception as e:
                console.print(f"[yellow]Articles fetch failed: {e}[/yellow]")

    if not articles_only:
        with console.status("[cyan]Generating book lesson...[/cyan]"):
            try:
                book_lesson = BooksPipeline().get_next_lesson(book_key=book)
            except Exception as e:
                console.print(f"[yellow]Book lesson failed: {e}[/yellow]")

    d = Digest(date=datetime.utcnow(), articles=articles, book_lesson=book_lesson)
    output = fmt.format_cli(d)
    console.print(output)


# ---- serve -----------------------------------------------------------

@app.command()
def serve() -> None:
    """Start the Telegram bot for interactive use."""
    from ragbrain.config import settings

    if not settings.telegram_bot_token:
        console.print(
            "[red]Error:[/red] RAGBRAIN_TELEGRAM_BOT_TOKEN is not set.\n"
            "Add it to your .env file. Get a bot token from @BotFather on Telegram."
        )
        raise typer.Exit(1)

    from ragbrain.delivery.telegram import run_bot

    console.print("[green]Starting RAGBrain Telegram bot...[/green]")
    console.print("[dim]Press Ctrl+C to stop.[/dim]")
    run_bot()


# ---- schedule --------------------------------------------------------

@app.command()
def schedule() -> None:
    """Start the automated digest scheduler (morning articles + evening lessons)."""
    from ragbrain.scheduler import run_scheduler

    console.print("[green]Starting RAGBrain scheduler...[/green]")
    run_scheduler()


# ---- fetch-articles --------------------------------------------------

@app.command(name="fetch-articles")
def fetch_articles(
    feed_url: list[str] = typer.Option(
        None, "--feed", "-f", help="RSS feed URL (repeatable). Defaults to .env feeds."
    ),
    no_ingest: bool = typer.Option(False, "--no-ingest", help="Skip indexing fetched articles"),
) -> None:
    """Fetch and summarize articles from RSS feeds."""
    from ragbrain.delivery.formatter import DigestFormatter
    from ragbrain.models import Digest
    from ragbrain.pipelines.articles import ArticlesPipeline

    fmt = DigestFormatter()
    feeds = list(feed_url) or None

    with console.status("[cyan]Fetching and summarizing articles...[/cyan]"):
        summaries = ArticlesPipeline().run(feed_urls=feeds, also_ingest=not no_ingest)

    d = Digest(date=datetime.utcnow(), articles=summaries)
    console.print(fmt.format_cli(d))


# ---- Main entry point ------------------------------------------------

def main() -> None:
    app()


if __name__ == "__main__":
    main()
