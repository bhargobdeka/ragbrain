"""RAGBrain CLI entry point.

Commands:
  ragbrain ingest <path_or_url>       Index a document or URL
  ragbrain query  <question>          Ask a question from your knowledge base
  ragbrain digest                     Generate and print today's digest
  ragbrain serve                      Start Telegram bot (add --scheduler for vacation mode)
  ragbrain schedule                   Start the automated scheduler
  ragbrain fetch-articles             Fetch and summarize articles now
  ragbrain ingest-slack               Ingest recent Slack news into knowledge base
  ragbrain review-architecture        Run the self-improvement architecture review
  ragbrain plan-upgrades              Run the Deep Agents upgrade planner
  ragbrain slack-setup                Find bot DM channel and write to .env
  ragbrain run-automation             Run the full daily automation loop right now
  ragbrain serve-slack                Start Slack approval poller + scheduler
  ragbrain eval                       Run quality evaluation suites
  ragbrain tracing                    Show LangSmith tracing status
"""

from __future__ import annotations

import logging
import os
import warnings
from datetime import datetime
from pathlib import Path

# Suppress HuggingFace tokenizers fork-after-parallelism warning (harmless).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Suppress pydantic UserWarnings from deepagents' use of typing.NotRequired.
# These are library internals — not actionable by RAGBrain users.
warnings.filterwarnings(
    "ignore",
    message="typing.NotRequired is not a Python type",
    category=UserWarning,
)

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
    from ragbrain.config import settings as _s

    def _run() -> dict:
        return rag_query(question, user_id=user_id)

    with console.status("[cyan]Searching your knowledge base...[/cyan]"):
        if _s.query_timeout > 0:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_run)
                try:
                    result = future.result(timeout=_s.query_timeout)
                except concurrent.futures.TimeoutError:
                    console.print(
                        f"[red]Query timed out after {_s.query_timeout}s.[/red]\n"
                        "[dim]Try a more specific query, or raise "
                        "RAGBRAIN_QUERY_TIMEOUT in your .env.[/dim]"
                    )
                    raise typer.Exit(1)
        else:
            result = _run()

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
def serve(
    with_scheduler: bool = typer.Option(
        False,
        "--scheduler/--no-scheduler",
        help="Also run the cron scheduler in the background (use for vacation mode)",
    ),
) -> None:
    """Start the Telegram bot.

    For vacation mode — leave Mac running with both the bot AND the
    daily scheduler active — use the --scheduler flag:

      ragbrain serve --scheduler

    Without --scheduler (default): just the interactive bot, no cron jobs.
    """
    from ragbrain.config import settings as _s

    if not _s.telegram_bot_token:
        console.print(
            "[red]Error:[/red] RAGBRAIN_TELEGRAM_BOT_TOKEN is not set.\n\n"
            "Run [cyan]ragbrain telegram-setup[/cyan] first."
        )
        raise typer.Exit(1)

    lines = [
        f"Telegram bot:  [bold green]ON[/bold green]",
        f"Scheduler:     [bold]{'ON (cron jobs active)' if with_scheduler else 'OFF'}[/bold]",
        f"Automation:    [bold]{'ENABLED' if _s.automation_enabled else 'DISABLED (set RAGBRAIN_AUTOMATION_ENABLED=true)'}[/bold]",
    ]
    if with_scheduler and _s.automation_enabled:
        lines += [
            "",
            f"Daily briefing + proposals: [cyan]{_s.morning_cron}[/cyan] UTC",
            f"Architecture snapshot:      [cyan]{_s.evening_cron}[/cyan] UTC",
        ]

    console.print(Panel("\n".join(lines), title="[bold]RAGBrain[/bold]", border_style="green"))
    console.print("[dim]Press Ctrl+C to stop.[/dim]\n")

    if with_scheduler:
        import threading
        from ragbrain.scheduler import run_scheduler

        t = threading.Thread(target=run_scheduler, daemon=True, name="ragbrain-scheduler")
        t.start()
        console.print("[dim]Scheduler running in background.[/dim]\n")

    from ragbrain.delivery.telegram import run_bot
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


# ---- eval helpers ---------------------------------------------------

def _red_team_to_suite_result(rt_result: "object") -> "object":
    """Convert a RedTeamResult into a SuiteResult for uniform JSON storage."""
    from ragbrain.eval.assertions import AssertionResult
    from ragbrain.eval.runner import CaseResult, SuiteResult

    case_results = []
    for v in rt_result.vulnerabilities:
        ar = AssertionResult(
            assertion_type="not_vulnerable",
            passed=not v.vulnerable,
            message="; ".join(v.evidence),
        )
        case_results.append(CaseResult(
            case_id=v.case_id,
            query=v.query,
            answer=v.answer,
            sources=[],
            assertion_results=[ar],
            judge_results=[],
            passed=not v.vulnerable,
            latency_ms=v.latency_ms,
            retrieval_attempts=0,
            hallucination_check="",
        ))
    return SuiteResult(
        suite_name="Red Team",
        description="Adversarial probe results",
        case_results=case_results,
    )


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
    history: bool = typer.Option(
        False, "--history", help="Show a table of past eval runs from tests/eval/results/ and exit.",
    ),
) -> None:
    """Run evaluation suites or red-team adversarial probes against the RAG pipeline.

    Quality evaluation (default):
        ragbrain eval                          # all suites, auto-saved to tests/eval/results/
        ragbrain eval --suite rag_basic        # one suite
        ragbrain eval --feature rlhf-book      # filter by feature tag
        ragbrain eval --output results.json    # also save to a specific path
        ragbrain eval --baseline prev.json     # compare against saved baseline
        ragbrain eval --history                # show past run history

    Red-team adversarial probing:
        ragbrain eval --red-team               # static + LLM-generated probes
        ragbrain eval --red-team --no-auto     # static YAML cases only
    """
    eval_path = Path(eval_dir)

    if not eval_path.exists() and not history:
        console.print(f"[red]Eval directory not found:[/red] {eval_path}")
        console.print("[dim]Run from the project root or pass --eval-dir.[/dim]")
        raise typer.Exit(1)

    # ---- History mode ------------------------------------------------
    if history:
        from ragbrain.eval.runner import EvalRunner
        from rich import box
        from rich.table import Table

        label = "red_team" if red_team else "eval"
        past = EvalRunner.load_history(eval_path, label=label, last_n=20)
        if not past:
            console.print(
                f"[yellow]No saved results yet in {eval_path}/results/[/yellow]\n"
                "[dim]Run [cyan]ragbrain eval[/cyan] to create the first record.[/dim]"
            )
            raise typer.Exit(0)

        tbl = Table(
            box=box.ROUNDED, header_style="bold cyan",
            title=f"[bold]Eval History[/bold] — {eval_path}/results/",
        )
        tbl.add_column("Run", style="cyan")
        tbl.add_column("Suite")
        tbl.add_column("Pass rate", justify="right")
        tbl.add_column("Avg faith.", justify="right")
        tbl.add_column("Avg relev.", justify="right")
        tbl.add_column("Cases", justify="right")

        for run in past:
            for suite in run["suites"]:
                rate = suite.get("pass_rate", 0)
                color = "green" if rate >= 0.8 else "yellow" if rate >= 0.5 else "red"
                faith = suite.get("avg_faithfulness")
                relev = suite.get("avg_relevance")
                tbl.add_row(
                    run["file"].replace(f"{label}_", "").replace(".json", ""),
                    suite.get("suite", "?"),
                    f"[{color}]{rate:.0%}[/{color}]",
                    f"{faith:.2f}" if faith else "[dim]—[/dim]",
                    f"{relev:.2f}" if relev else "[dim]—[/dim]",
                    str(len(suite.get("cases", []))),
                )
        console.print(tbl)
        raise typer.Exit(0)

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

        # Auto-save red-team results
        from ragbrain.eval.runner import EvalRunner, SuiteResult, CaseResult
        rt_suite = _red_team_to_suite_result(result)
        saved_path = EvalRunner.auto_save([rt_suite], eval_path, label="red_team")
        console.print(f"[dim]Results auto-saved to [cyan]{saved_path}[/cyan][/dim]")

        if output:
            EvalRunner.save_results([rt_suite], Path(output))

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

    # Always auto-save to tests/eval/results/
    auto_path = EvalRunner.auto_save(results, eval_path, label="eval")
    console.print(f"[dim]Results auto-saved to [cyan]{auto_path}[/cyan][/dim]")
    console.print(f"[dim]View history with: [cyan]ragbrain eval --history[/cyan][/dim]")

    # Also save to explicit output path if requested
    if output:
        EvalRunner.save_results(results, Path(output))
        console.print(f"[dim]Also saved to [cyan]{output}[/cyan][/dim]")

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


# ---- run-automation --------------------------------------------------

@app.command(name="run-automation")
def run_automation(
    dry_run: bool = typer.Option(False, "--dry-run", help="Print output but don't send to Telegram"),
    skip_planner: bool = typer.Option(False, "--skip-planner", help="Skip upgrade planner (faster, briefing only)"),
) -> None:
    """Run the full daily automation loop right now.

    This is the same job the scheduler runs at 8 AM. Use it to:
    - Test the pipeline before vacation
    - Trigger manually when you want a fresh briefing + proposals
    - Debug issues without waiting for the cron to fire

    Steps:
      1. Ingest recent Slack news into knowledge base
      2. Generate daily briefing (what Tuk covered + RAGBrain relevance)
      3. Run UpgradePlanner → create Proposals → send to Telegram with buttons
    """
    from ragbrain.config import settings as _s

    if not _s.automation_enabled and not dry_run:
        console.print(
            "[yellow]RAGBRAIN_AUTOMATION_ENABLED is false in your .env.[/yellow]\n"
            "Running anyway (you triggered this manually).\n"
        )

    # ---- Step 1: Ingest Slack news -----------------------------------
    console.print("[cyan]Step 1/3 — Ingesting Slack news...[/cyan]")
    ingested_count = 0
    try:
        from ragbrain.ingestion.extractors.slack import SlackExtractor
        from ragbrain.ingestion.pipeline import IngestionPipeline

        extractor = SlackExtractor(fetch_urls=False)
        with console.status("Fetching from Slack..."):
            docs = extractor.extract_recent()

        if docs:
            pipeline = IngestionPipeline()
            try:
                for doc in docs:
                    pipeline.ingest_document(doc)
                ingested_count = len(docs)
            finally:
                pipeline.close()   # release Qdrant lock before step 2 opens it
            console.print(f"  [green]✓[/green] Ingested {ingested_count} Slack messages.")
        else:
            console.print("  [yellow]No new Slack messages found.[/yellow]")
    except Exception as e:
        console.print(f"  [red]Slack ingestion failed:[/red] {e}")

    # ---- Step 2: Daily briefing --------------------------------------
    console.print("\n[cyan]Step 2/3 — Generating daily briefing...[/cyan]")
    briefing = ""
    try:
        from ragbrain.pipelines.daily_briefing import generate_daily_briefing

        with console.status("Generating briefing with LLM..."):
            briefing = generate_daily_briefing()

        # Print a plain-text preview (strip HTML tags for CLI)
        import re
        plain = re.sub(r"<[^>]+>", "", briefing)
        console.print(Panel(plain[:1200] + ("..." if len(plain) > 1200 else ""),
                            title="[bold]Daily Briefing Preview[/bold]",
                            border_style="blue"))

        if not dry_run:
            sent_any = False
            # Try Slack first
            from ragbrain.delivery.slack_delivery import post_briefing
            if _s.slack_bot_token and (_s.slack_bot_channel_id or _s.slack_post_channel_id or _s.slack_channel_id):
                ok = post_briefing(briefing)
                if ok:
                    console.print("  [green]✓[/green] Briefing sent to Slack.")
                    sent_any = True
            # Try Telegram
            if _s.telegram_bot_token and _s.telegram_chat_id:
                import asyncio, re as _re
                plain = _re.sub(r"<[^>]+>", "", briefing)
                from ragbrain.scheduler import _send_telegram
                asyncio.run(_send_telegram(plain))
                console.print("  [green]✓[/green] Briefing sent to Telegram.")
                sent_any = True
            if not sent_any:
                console.print(
                    "  [yellow]No delivery channel configured.[/yellow]\n"
                    "  Run [cyan]ragbrain telegram-setup[/cyan] to connect Telegram (recommended),\n"
                    "  or [cyan]ragbrain slack-setup[/cyan] for Slack."
                )
        else:
            console.print("  [dim]Dry-run: send skipped.[/dim]")
    except Exception as e:
        console.print(f"  [red]Briefing failed:[/red] {e}")

    # ---- Step 3: Upgrade proposals -----------------------------------
    if not skip_planner:
        console.print("\n[cyan]Step 3/3 — Running UpgradePlanner (may take 1-2 min)...[/cyan]")
        try:
            from ragbrain.pipelines.proposals import Proposal, get_store
            from ragbrain.pipelines.upgrade_planner import get_upgrade_recommendations

            with console.status("Planning upgrades with Deep Agents..."):
                recs = get_upgrade_recommendations()

            if not recs:
                console.print("  [yellow]No recommendations returned.[/yellow]")
            else:
                store = get_store()
                sent = 0
                for rec in recs[:3]:
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
                    console.print(
                        f"  [green]✓[/green] Proposal #{proposal.id}: [{proposal.priority}] {proposal.title[:60]}"
                    )

                    if not dry_run:
                        # Send to Telegram (preferred — inline approve/skip buttons)
                        if _s.telegram_bot_token and _s.telegram_chat_id:
                            import asyncio
                            from ragbrain.delivery.telegram import send_proposal as tg_send
                            from telegram import Bot
                            async def _tg():
                                async with Bot(token=_s.telegram_bot_token) as bot:
                                    await tg_send(bot, int(_s.telegram_chat_id), proposal)
                            asyncio.run(_tg())
                            sent += 1
                        # Also send to Slack if configured (text-based)
                        elif _s.slack_bot_token and (_s.slack_bot_channel_id or _s.slack_channel_id):
                            from ragbrain.delivery.slack_delivery import post_proposal
                            if post_proposal(proposal):
                                sent += 1

                if sent:
                    channels = []
                    if _s.telegram_bot_token and _s.telegram_chat_id:
                        channels.append("[cyan]Telegram[/cyan] (tap Approve/Skip buttons)")
                    if _s.slack_bot_token and (_s.slack_bot_channel_id or _s.slack_channel_id):
                        channels.append("[cyan]Slack[/cyan] (reply approve <id>)")
                    console.print(
                        f"\n  [green]✓[/green] {sent} proposal(s) sent to {' + '.join(channels) or 'delivery channel'}."
                    )
                elif dry_run:
                    console.print("  [dim]Dry-run: send skipped.[/dim]")
                else:
                    console.print(
                        "  [yellow]Proposals saved locally. No delivery channel configured.[/yellow]\n"
                        "  Run [cyan]ragbrain telegram-setup[/cyan] to connect Telegram."
                    )
        except Exception as e:
            console.print(f"  [red]Upgrade planner failed:[/red] {e}")
    else:
        console.print("\n[dim]Step 3/3 — Skipped (--skip-planner).[/dim]")

    # ---- Summary -------------------------------------------------------
    console.print()
    from ragbrain.config import settings as _sf
    _has_tg = bool(_sf.telegram_bot_token and _sf.telegram_chat_id)
    _has_slack = bool(_sf.slack_bot_token and (_sf.slack_bot_channel_id or _sf.slack_channel_id))
    if _has_tg:
        _next_step = "Run [cyan]ragbrain serve[/cyan] to handle Telegram Approve/Skip button taps."
    elif _has_slack:
        _next_step = "Reply [cyan]approve <id>[/cyan] in Slack. Run [cyan]ragbrain serve-slack[/cyan]."
    else:
        _next_step = "Run [cyan]ragbrain telegram-setup[/cyan] to connect a delivery channel."

    console.print(Panel(
        f"Slack messages ingested: [bold]{ingested_count}[/bold]\n"
        f"Briefing generated:      [bold]{'yes' if briefing else 'no'}[/bold]\n"
        f"Proposals stored:        [cyan]~/.ragbrain/proposals.json[/cyan]\n\n"
        f"[dim]{_next_step}[/dim]",
        title="[bold]Automation Run Complete[/bold]",
        border_style="green",
    ))


# ---- telegram-setup --------------------------------------------------------

@app.command(name="telegram-setup")
def telegram_setup() -> None:
    """Interactive guide to connect your Telegram bot for approvals.

    Takes about 5 minutes. At the end you will have:
      - A Telegram bot that receives briefings and proposal buttons
      - RAGBRAIN_TELEGRAM_BOT_TOKEN and RAGBRAIN_TELEGRAM_CHAT_ID in .env
      - approve / skip / explain buttons that work from your phone anywhere

    Steps (follow the prompts):
      1. Create a bot via @BotFather in Telegram
      2. Paste the token here
      3. Start a chat with your bot and press Enter
      4. Your Chat ID is found automatically and saved to .env
    """
    console.print(Panel(
        "[bold]Telegram bot setup — ~5 minutes[/bold]\n\n"
        "You'll need the Telegram app open on your phone or desktop.\n"
        "This only needs to be done once.",
        title="[bold cyan]RAGBrain × Telegram[/bold cyan]",
        border_style="cyan",
    ))

    # ---- Step 1: Get bot token -------------------------------------------
    console.print("\n[bold]Step 1 — Create a bot[/bold]")
    console.print(
        "  1. Open Telegram and search for [cyan]@BotFather[/cyan]\n"
        "  2. Send: [cyan]/newbot[/cyan]\n"
        "  3. Follow the prompts (name + username, e.g. [dim]RAGBrain / ragbrain_myname_bot[/dim])\n"
        "  4. BotFather will give you a token like: [dim]7123456789:AAH...[/dim]\n"
    )
    token = typer.prompt("  Paste your bot token here").strip()

    if not token or ":" not in token:
        console.print("[red]That doesn't look like a valid token (expected format: 123456:ABC...)[/red]")
        raise typer.Exit(1)

    # Verify the token works
    console.print("\n  Verifying token...")
    try:
        import requests
        r = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
        data = r.json()
        if not data.get("ok"):
            console.print(f"[red]Token rejected by Telegram:[/red] {data.get('description')}")
            raise typer.Exit(1)
        bot_name = data["result"].get("first_name", "")
        bot_username = data["result"].get("username", "")
        console.print(f"  [green]✓[/green] Bot verified: [bold]{bot_name}[/bold] (@{bot_username})")
    except Exception as e:
        console.print(f"[red]Could not reach Telegram API:[/red] {e}")
        raise typer.Exit(1)

    # ---- Step 2: Get chat ID -----------------------------------------------
    console.print(
        f"\n[bold]Step 2 — Start a chat with your bot[/bold]\n\n"
        f"  1. In Telegram, search for [cyan]@{bot_username}[/cyan]\n"
        f"  2. Tap [cyan]START[/cyan] or send any message (e.g. 'hello')\n"
    )
    typer.confirm("  Done? (press Enter after you've sent a message)", default=True)

    # Poll getUpdates to find the chat ID
    console.print("  Looking up your Chat ID...")
    chat_id = ""
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{token}/getUpdates",
            timeout=10,
        )
        updates = r.json().get("result", [])
        if updates:
            # Take the most recent message
            last = updates[-1]
            msg = last.get("message") or last.get("my_chat_member", {})
            chat = msg.get("chat", {}) if isinstance(msg, dict) else {}
            chat_id = str(chat.get("id", ""))

        if not chat_id:
            console.print(
                "[yellow]Could not auto-detect Chat ID.[/yellow]\n\n"
                "Manual alternative:\n"
                f"  Visit: https://api.telegram.org/bot{token}/getUpdates\n"
                "  Look for: result[0].message.chat.id\n"
                "  It's a number like 123456789 (or negative for groups)\n"
            )
            chat_id = typer.prompt("  Paste your Chat ID manually").strip()
        else:
            console.print(f"  [green]✓[/green] Chat ID found: [bold cyan]{chat_id}[/bold cyan]")
    except Exception as e:
        console.print(f"[red]Could not fetch updates:[/red] {e}")
        chat_id = typer.prompt("  Paste your Chat ID manually").strip()

    if not chat_id:
        console.print("[red]Chat ID is required.[/red]")
        raise typer.Exit(1)

    # ---- Step 3: Write to .env --------------------------------------------
    env_path = Path(".env")
    import re as _re
    if env_path.exists():
        env_text = env_path.read_text(encoding="utf-8")
        # Update or append BOT_TOKEN
        if "RAGBRAIN_TELEGRAM_BOT_TOKEN=" in env_text:
            env_text = _re.sub(r"RAGBRAIN_TELEGRAM_BOT_TOKEN=.*", f"RAGBRAIN_TELEGRAM_BOT_TOKEN={token}", env_text)
        else:
            env_text += f"\nRAGBRAIN_TELEGRAM_BOT_TOKEN={token}\n"
        # Update or append CHAT_ID
        if "RAGBRAIN_TELEGRAM_CHAT_ID=" in env_text:
            env_text = _re.sub(r"RAGBRAIN_TELEGRAM_CHAT_ID=.*", f"RAGBRAIN_TELEGRAM_CHAT_ID={chat_id}", env_text)
        else:
            env_text += f"RAGBRAIN_TELEGRAM_CHAT_ID={chat_id}\n"
        env_path.write_text(env_text, encoding="utf-8")
        console.print(f"\n  [green]✓[/green] Saved to [cyan].env[/cyan]")
    else:
        console.print(
            f"\n[yellow].env not found — add these manually:[/yellow]\n"
            f"  RAGBRAIN_TELEGRAM_BOT_TOKEN={token}\n"
            f"  RAGBRAIN_TELEGRAM_CHAT_ID={chat_id}"
        )

    # ---- Step 4: Send test message ----------------------------------------
    try:
        import requests
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": (
                    "👋 *RAGBrain is connected\\!*\n\n"
                    "You'll receive daily briefings and upgrade proposals here\\.\n"
                    "Tap *Approve*, *Skip*, or *Explain* on each proposal\\.\n\n"
                    "Run `ragbrain serve` to start the bot\\."
                ),
                "parse_mode": "MarkdownV2",
            },
            timeout=10,
        )
        console.print("  [green]✓[/green] Test message sent — check Telegram!")
    except Exception as e:
        console.print(f"  [yellow]Could not send test message:[/yellow] {e}")

    console.print(Panel(
        f"Bot: [bold]@{bot_username}[/bold]\n"
        f"Chat ID: [bold cyan]{chat_id}[/bold cyan]\n\n"
        "Next step: run [bold cyan]ragbrain serve[/bold cyan]\n"
        "  This starts the bot and handles Approve/Skip button taps.\n\n"
        "To send proposals to Telegram, run:\n"
        "  [cyan]ragbrain run-automation[/cyan]",
        title="[bold green]Telegram Setup Complete[/bold green]",
        border_style="green",
    ))


# ---- slack-setup --------------------------------------------------------

@app.command(name="slack-setup")
def slack_setup() -> None:
    """Find the bot's own DM channel and write it to .env.

    The problem:
      Your news channel (RAGBRAIN_SLACK_CHANNEL_ID) is a DM with Tuk.
      The bot can *post* there but cannot *read* your replies — so
      approve/skip commands typed there are invisible to RAGBrain.

    The fix:
      RAGBrain needs its own DM channel with you, where it's a participant
      and can read your messages.  This command finds that channel ID and
      adds RAGBRAIN_SLACK_BOT_CHANNEL_ID to your .env automatically.

    How to run:
      1. In Slack, find your RAGBrain bot (search for its app name)
      2. Click 'Message' to open a DM with the bot — send any message
      3. Run this command — it will detect the DM and update .env
    """
    from ragbrain.config import settings as _s

    if not _s.slack_bot_token:
        console.print("[red]RAGBRAIN_SLACK_BOT_TOKEN not set in .env[/red]")
        raise typer.Exit(1)

    import ssl
    import certifi
    from slack_sdk import WebClient

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    client = WebClient(token=_s.slack_bot_token, ssl=ssl_ctx)

    console.print("[cyan]Looking up bot identity...[/cyan]")
    try:
        auth = client.auth_test()
        bot_user_id = auth["user_id"]
        bot_name = auth.get("user", "ragbrain-bot")
        console.print(f"  Bot: [bold]{bot_name}[/bold]  (user_id={bot_user_id})")
    except Exception as e:
        console.print(f"[red]auth.test failed:[/red] {e}")
        raise typer.Exit(1)

    console.print("[cyan]Searching for DM channels the bot is in...[/cyan]")
    try:
        # im.list returns all DMs opened with this bot
        resp = client.conversations_list(types="im", limit=200)
        ims = resp.get("channels", [])
    except Exception as e:
        console.print(f"[red]conversations.list failed:[/red] {e}")
        raise typer.Exit(1)

    if not ims:
        console.print(
            "\n[yellow]No DM channels found for this bot.[/yellow]\n\n"
            "To fix this:\n"
            "  1. Open Slack on your phone or desktop\n"
            "  2. Search for your bot (by its app name)\n"
            "  3. Tap 'Message' and send any message (e.g. 'hello')\n"
            "  4. Run [cyan]ragbrain slack-setup[/cyan] again\n"
        )
        raise typer.Exit(1)

    # Find the most recently active DM (most likely the one the user just opened)
    latest_im = max(ims, key=lambda c: float(c.get("last_read", "0") or "0"))
    bot_channel_id = latest_im["id"]

    console.print(f"\n  [green]✓[/green] Found bot DM channel: [bold cyan]{bot_channel_id}[/bold cyan]")

    # Send a test message so the user can verify in Slack
    try:
        client.chat_postMessage(
            channel=bot_channel_id,
            text=(
                "👋 *RAGBrain is set up!*\n\n"
                "This is your control channel. Reply here to act on proposals:\n"
                "  • `approve <id>` — implement a proposal\n"
                "  • `skip <id>` — skip a proposal\n"
                "  • `explain <id>` — show full proposal details\n\n"
                "Start the bot with: `ragbrain serve-slack`"
            ),
            mrkdwn=True,
        )
        console.print("  [green]✓[/green] Test message sent — check Slack to confirm.")
    except Exception as e:
        console.print(f"  [yellow]Could not send test message:[/yellow] {e}")

    # Write to .env
    env_path = Path(".env")
    if env_path.exists():
        env_text = env_path.read_text(encoding="utf-8")
        if "RAGBRAIN_SLACK_BOT_CHANNEL_ID=" in env_text:
            # Update existing line
            import re
            env_text = re.sub(
                r"RAGBRAIN_SLACK_BOT_CHANNEL_ID=.*",
                f"RAGBRAIN_SLACK_BOT_CHANNEL_ID={bot_channel_id}",
                env_text,
            )
        else:
            # Append after RAGBRAIN_SLACK_CHANNEL_ID line
            env_text = env_text.replace(
                f"RAGBRAIN_SLACK_CHANNEL_ID={_s.slack_channel_id}",
                f"RAGBRAIN_SLACK_CHANNEL_ID={_s.slack_channel_id}\n"
                f"RAGBRAIN_SLACK_BOT_CHANNEL_ID={bot_channel_id}",
            )
        env_path.write_text(env_text, encoding="utf-8")
        console.print(f"  [green]✓[/green] Updated [cyan].env[/cyan]: RAGBRAIN_SLACK_BOT_CHANNEL_ID={bot_channel_id}")
    else:
        console.print(
            f"\n[yellow].env not found — add this manually:[/yellow]\n"
            f"  RAGBRAIN_SLACK_BOT_CHANNEL_ID={bot_channel_id}"
        )

    console.print(Panel(
        f"Bot DM channel: [bold cyan]{bot_channel_id}[/bold cyan]\n\n"
        "RAGBrain will now post briefings and proposals to this DM.\n"
        "Reply [cyan]approve <id>[/cyan] / [cyan]skip <id>[/cyan] to act on proposals.\n\n"
        "Next step: run [bold cyan]ragbrain serve-slack[/bold cyan]",
        title="[bold green]Slack Setup Complete[/bold green]",
        border_style="green",
    ))


# ---- serve-slack --------------------------------------------------------

@app.command(name="serve-slack")
def serve_slack(
    poll_interval: int = typer.Option(15, "--poll", help="Seconds between approval polls"),
    with_scheduler: bool = typer.Option(True, "--scheduler/--no-scheduler",
                                        help="Also run the cron scheduler in the background"),
) -> None:
    """Start the Slack approval poller (and optionally the scheduler).

    This is your single command for leaving the Mac unattended.
    It does two things in parallel:

    1. Watches the Slack DM channel every N seconds for messages like:
         approve <id>   → triggers AutoImplementer on the approved proposal
         skip <id>      → marks the proposal as skipped
         explain <id>   → posts the full proposal details back to Slack

    2. (Optional) Runs the full cron scheduler so daily briefings and
       upgrade proposals are posted automatically at the configured times.

    Usage:
      ragbrain serve-slack                    # poller + scheduler
      ragbrain serve-slack --no-scheduler     # poller only
      ragbrain serve-slack --poll 30          # check every 30 seconds
    """
    from ragbrain.config import settings as _s

    if not _s.slack_bot_token or not (_s.slack_post_channel_id or _s.slack_channel_id):
        console.print(
            "[red]Slack not configured.[/red]\n\n"
            "Add these to your [cyan].env[/cyan]:\n"
            "  RAGBRAIN_SLACK_BOT_TOKEN=xoxb-...\n"
            "  RAGBRAIN_SLACK_CHANNEL_ID=<channel or DM ID>",
        )
        raise typer.Exit(1)

    if not _s.automation_enabled:
        console.print(
            "[yellow]RAGBRAIN_AUTOMATION_ENABLED is false.[/yellow]\n"
            "Approval poller will run, but the scheduler won't send daily automation jobs.\n"
            "Add RAGBRAIN_AUTOMATION_ENABLED=true to your .env to enable the full loop.\n"
        )

    console.print(Panel(
        f"Slack channel:   [cyan]{_s.slack_post_channel_id or _s.slack_channel_id}[/cyan]\n"
        f"Poll interval:   [bold]{poll_interval}s[/bold]\n"
        f"Scheduler:       [bold]{'yes' if with_scheduler else 'no'}[/bold]\n"
        f"Automation:      [bold]{'ENABLED' if _s.automation_enabled else 'DISABLED'}[/bold]\n\n"
        f"Reply in Slack:\n"
        f"  [cyan]approve <id>[/cyan] — implement the proposal\n"
        f"  [cyan]skip <id>[/cyan]    — skip the proposal\n"
        f"  [cyan]explain <id>[/cyan] — show full proposal details",
        title="[bold]RAGBrain Slack Bot[/bold]",
        border_style="cyan",
    ))

    if with_scheduler:
        import threading
        from ragbrain.scheduler import run_scheduler

        def _run_sched():
            try:
                run_scheduler()
            except Exception:
                logger.exception("Scheduler crashed")

        t = threading.Thread(target=_run_sched, daemon=True, name="ragbrain-scheduler")
        t.start()
        console.print("[dim]Scheduler started in background thread.[/dim]\n")

    # Run the approval poller in the main thread (blocking)
    try:
        from ragbrain.delivery.slack_delivery import run_approval_loop
        run_approval_loop(poll_interval=poll_interval)
    except KeyboardInterrupt:
        console.print("\n[yellow]Slack bot stopped.[/yellow]")


# ---- tracing --------------------------------------------------------

@app.command()
def tracing() -> None:
    """Show LangSmith tracing status and instructions to enable it."""
    from ragbrain.config import settings as _s

    enabled = _s.setup_tracing()
    has_key = bool(_s.langsmith_api_key)

    if enabled:
        console.print(Panel(
            f"[green]LangSmith tracing is [bold]ACTIVE[/bold][/green]\n\n"
            f"Project : [cyan]{_s.langsmith_project}[/cyan]\n"
            f"Endpoint: [dim]{_s.langsmith_endpoint}[/dim]\n\n"
            f"Open your dashboard at [link=https://smith.langchain.com]https://smith.langchain.com[/link] "
            f"and select the [cyan]{_s.langsmith_project}[/cyan] project.",
            title="[bold]Tracing[/bold]",
            border_style="green",
        ))
    elif has_key:
        console.print(Panel(
            "[yellow]API key is set but tracing is not enabled.[/yellow]\n\n"
            "Add this to your [cyan].env[/cyan]:\n"
            "  [bold]LANGCHAIN_TRACING_V2=true[/bold]\n"
            "  [bold]LANGCHAIN_PROJECT=ragbrain[/bold]",
            title="[bold]Tracing[/bold]",
            border_style="yellow",
        ))
    else:
        console.print(Panel(
            "[red]LangSmith tracing is [bold]OFF[/bold][/red] — no API key found.\n\n"
            "To enable:\n"
            "  1. Sign up at [link=https://smith.langchain.com]https://smith.langchain.com[/link]\n"
            "  2. Copy your API key from [bold]Settings → API Keys[/bold]\n"
            "  3. Add to your [cyan].env[/cyan]:\n"
            "       [bold]LANGCHAIN_API_KEY=lsv2_...[/bold]\n"
            "       [bold]LANGCHAIN_TRACING_V2=true[/bold]\n"
            "       [bold]LANGCHAIN_PROJECT=ragbrain[/bold]\n\n"
            "Every [cyan]ragbrain query[/cyan] and [cyan]ragbrain eval[/cyan] run will then "
            "appear as a named trace in your LangSmith dashboard.",
            title="[bold]Tracing[/bold]",
            border_style="red",
        ))


# ---- Main entry point ------------------------------------------------

def main() -> None:
    app()


if __name__ == "__main__":
    main()
