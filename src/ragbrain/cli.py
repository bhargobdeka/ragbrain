"""RAGBrain CLI entry point.

Commands:
  ragbrain ingest <path_or_url>       Index a document or URL
  ragbrain query  <question>          Ask a question from your knowledge base
  ragbrain digest                     Generate and print today's digest
  ragbrain serve                      Start the Telegram bot (interactive)
  ragbrain schedule                   Start the automated scheduler
  ragbrain fetch-articles             Fetch and summarize articles now
  ragbrain ingest-slack               Ingest recent Slack news into knowledge base
  ragbrain review-architecture        Run the self-improvement architecture review
  ragbrain plan-upgrades              Run the Deep Agents upgrade planner
  ragbrain eval                       Run quality evaluation suites
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
    # Check if it looks like a PDF to handle books specially
    if source.lower().endswith(".pdf") and Path(source).exists():
        ingest_as_book = typer.confirm("Register this PDF as a book for daily lessons?", default=False)
        if ingest_as_book:
            from ragbrain.pipelines.books import BooksPipeline

            books = BooksPipeline(user_id=user_id)
            try:
                count = books.ingest_book(source)
            finally:
                books.close()
            console.print(f"[green]✓ Book registered and indexed ({count} chunks)[/green]")
            return

    from ragbrain.ingestion.pipeline import IngestionPipeline
    pipeline = IngestionPipeline(user_id=user_id)
    try:
        count = pipeline.ingest(source)
    finally:
        pipeline.close()
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


# ---- ingest-slack ----------------------------------------------------

@app.command(name="ingest-slack")
def ingest_slack(
    lookback: int = typer.Option(None, "--lookback", "-l", help="Hours to look back (default: config value)"),
    fetch_urls: bool = typer.Option(False, "--fetch-urls", help="Also fetch full articles from URLs in messages"),
    user_id: str = typer.Option(None, "--user-id", "-u", help="Namespace for multi-tenant use"),
) -> None:
    """Ingest recent Slack news messages into the knowledge base."""
    from ragbrain.config import settings as _settings

    if not _settings.slack_bot_token or not _settings.slack_channel_id:
        console.print(
            "[red]Error:[/red] RAGBRAIN_SLACK_BOT_TOKEN and RAGBRAIN_SLACK_CHANNEL_ID must be set in .env"
        )
        raise typer.Exit(1)

    from ragbrain.ingestion.extractors.slack import SlackExtractor
    from ragbrain.ingestion.pipeline import IngestionPipeline

    extractor = SlackExtractor(fetch_urls=fetch_urls)
    pipeline = IngestionPipeline(user_id=user_id)

    with console.status("[cyan]Reading Slack messages...[/cyan]"):
        docs = extractor.extract_recent(lookback_hours=lookback)

    if not docs:
        console.print("[yellow]No new messages found in Slack.[/yellow]")
        raise typer.Exit(0)

    total = 0
    for doc in docs:
        count = pipeline.ingest_document(doc)
        total += count

    console.print(f"[green]Ingested {len(docs)} messages ({total} chunks) from Slack.[/green]")


# ---- review-architecture ---------------------------------------------

@app.command(name="review-architecture")
def review_architecture(
    post_slack: bool = typer.Option(False, "--post-slack", help="Post the review back to Slack"),
) -> None:
    """Run the architecture review agent — analyses recent AI news against RAGBrain's design."""
    from ragbrain.config import settings as _settings

    if not _settings.slack_bot_token or not _settings.slack_channel_id:
        console.print(
            "[red]Error:[/red] Slack integration required. Set RAGBRAIN_SLACK_BOT_TOKEN and "
            "RAGBRAIN_SLACK_CHANNEL_ID in .env"
        )
        raise typer.Exit(1)

    from ragbrain.pipelines.architecture_review import run_review

    with console.status("[cyan]Running architecture review...[/cyan]"):
        report = run_review(post_slack=post_slack)

    console.print()
    console.print(Panel(report, title="[bold]Architecture Review[/bold]", border_style="blue"))

    if post_slack:
        console.print("[green]Review posted to Slack.[/green]")


# ---- eval ------------------------------------------------------------

@app.command()
def eval(
    suite: str = typer.Option(
        None, "--suite", "-s",
        help="Run a single suite by filename stem (e.g. 'rag_basic'). Default: all suites.",
    ),
    feature: str = typer.Option(
        None, "--feature", "-f",
        help="Only run test cases tagged with this feature (e.g. 'rag-core').",
    ),
    red_team: bool = typer.Option(
        False, "--red-team", help="Run adversarial red-team probes instead of quality suites.",
    ),
    no_auto: bool = typer.Option(
        False, "--no-auto",
        help="(Red team only) Skip LLM-generated probes; run static YAML cases only.",
    ),
    output: str = typer.Option(
        None, "--output", "-o",
        help="Save results as JSON to this path (useful for CI/regression comparison).",
    ),
    baseline: str = typer.Option(
        None, "--baseline",
        help="Compare results against a previously saved JSON baseline.",
    ),
    fail_fast: bool = typer.Option(
        False, "--fail-fast", help="Stop after the first failing test case.",
    ),
    eval_dir: str = typer.Option(
        "tests/eval", "--eval-dir",
        help="Directory containing YAML eval suites (default: tests/eval).",
    ),
) -> None:
    """Run evaluation suites or red-team adversarial probes against the RAG pipeline.

    Quality evaluation (default):
        ragbrain eval                          # all suites
        ragbrain eval --suite rag_basic        # one suite
        ragbrain eval --feature rlhf-book      # filter by feature tag
        ragbrain eval --output results.json    # save for regression tracking
        ragbrain eval --baseline prev.json     # compare against saved baseline

    Red-team adversarial probing:
        ragbrain eval --red-team               # static + LLM-generated probes
        ragbrain eval --red-team --no-auto     # static YAML cases only
    """
    eval_path = Path(eval_dir)

    if not eval_path.exists():
        console.print(f"[red]Eval directory not found:[/red] {eval_path}")
        console.print("[dim]Run from the project root or pass --eval-dir.[/dim]")
        raise typer.Exit(1)

    # ---- Red-team mode -----------------------------------------------
    if red_team:
        from ragbrain.eval.red_team import print_red_team_report, run_red_team

        static = eval_path / "red_team.yaml"
        console.print("[bold red]Running red-team adversarial probes...[/bold red]")
        if not no_auto:
            console.print("[dim]Auto-generating additional probes via LLM (--no-auto to skip).[/dim]")

        with console.status("[red]Probing for vulnerabilities...[/red]"):
            result = run_red_team(
                static_path=static if static.exists() else None,
                auto_generate=not no_auto,
            )

        console.print()
        print_red_team_report(result)

        vuln_count = sum(1 for v in result.vulnerabilities if v.vulnerable)
        high_count = len(result.high_severity_vulns)
        if high_count:
            console.print(f"\n[red]Found {high_count} HIGH severity vulnerabilities — review above.[/red]")
            raise typer.Exit(1)
        elif vuln_count:
            console.print(f"\n[yellow]{vuln_count} vulnerability/ies detected — review above.[/yellow]")
        else:
            console.print("\n[green]No vulnerabilities detected.[/green]")
        return

    # ---- Quality evaluation mode -------------------------------------
    from ragbrain.eval.report import print_overall_summary, print_suite_report
    from ragbrain.eval.runner import EvalRunner

    runner = EvalRunner(fail_fast=fail_fast)

    console.print("[bold cyan]Running RAGBrain evaluation suites...[/bold cyan]")
    if feature:
        console.print(f"[dim]Filtering to feature tag: [cyan]{feature}[/cyan][/dim]")

    if suite:
        suite_path = eval_path / f"{suite}.yaml"
        if not suite_path.exists():
            console.print(f"[red]Suite file not found:[/red] {suite_path}")
            raise typer.Exit(1)
        with console.status(f"[cyan]Running suite '{suite}'...[/cyan]"):
            results = [runner.run_suite(suite_path, feature_filter=feature)]
    else:
        with console.status("[cyan]Running all suites (excluding red_team.yaml)...[/cyan]"):
            results = runner.run_all(
                eval_path,
                feature_filter=feature,
                exclude_files=["red_team.yaml"],
            )

    # Print per-suite tables
    console.print()
    for sr in results:
        if sr.case_results:
            print_suite_report(sr)

    # Regression check
    regressions: list[str] = []
    if baseline:
        baseline_path = Path(baseline)
        regressions = EvalRunner.compare_baseline(results, baseline_path)

    print_overall_summary(results, regressions=regressions or None)

    # Save JSON output
    if output:
        out_path = Path(output)
        EvalRunner.save_results(results, out_path)
        console.print(f"[dim]Results saved to [cyan]{out_path}[/cyan][/dim]")

    # Exit 1 if any failures (useful for CI)
    total_failed = sum(
        sum(1 for r in sr.case_results if not r.passed) for sr in results
    )
    if total_failed or regressions:
        raise typer.Exit(1)


# ---- plan-upgrades ---------------------------------------------------

@app.command(name="plan-upgrades")
def plan_upgrades(
    post_slack: bool = typer.Option(False, "--post-slack", help="Post the plan back to Slack"),
    lookback: int = typer.Option(
        24, "--lookback", "-l", help="Hours of Slack news to consider (default: 24)"
    ),
) -> None:
    """Run the Deep Agents upgrade planner.

    The agent reads ARCHITECTURE.md + upgrade history, fetches recent Slack
    news, searches the knowledge base, and produces a prioritised upgrade plan.
    Results are persisted to architecture-state.md for future runs.
    """
    from ragbrain.config import settings as _settings

    if not _settings.slack_bot_token or not _settings.slack_channel_id:
        console.print(
            "[red]Error:[/red] Slack integration required for news fetching.\n"
            "Set RAGBRAIN_SLACK_BOT_TOKEN and RAGBRAIN_SLACK_CHANNEL_ID in .env"
        )
        raise typer.Exit(1)

    from ragbrain.pipelines.upgrade_planner import run_upgrade_planner

    console.print("[cyan]Starting Deep Agents upgrade planner...[/cyan]")
    console.print("[dim]The agent will call multiple tools — this may take a minute.[/dim]\n")

    with console.status("[cyan]Planning architecture upgrades...[/cyan]"):
        report = run_upgrade_planner(post_slack=post_slack)

    console.print()
    console.print(Panel(report, title="[bold]Upgrade Plan[/bold]", border_style="magenta"))

    from pathlib import Path
    state_path = Path("architecture-state.md")
    if state_path.exists():
        console.print(
            f"\n[dim]Plan appended to [cyan]architecture-state.md[/cyan] "
            f"for cross-run memory.[/dim]"
        )

    if post_slack:
        console.print("[green]Plan posted to Slack.[/green]")


# ---- Main entry point ------------------------------------------------

def main() -> None:
    app()


if __name__ == "__main__":
    main()
