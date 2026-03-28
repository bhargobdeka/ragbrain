"""Rich terminal report for evaluation results."""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ragbrain.eval.runner import CaseResult, SuiteResult

console = Console()


def _pass_badge(passed: bool) -> str:
    return "[bold green]PASS[/bold green]" if passed else "[bold red]FAIL[/bold red]"


def _score_color(score: float) -> str:
    if score >= 0.8:
        return "green"
    if score >= 0.6:
        return "yellow"
    return "red"


def print_suite_report(suite: SuiteResult) -> None:
    """Print a Rich table for one eval suite."""
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Test ID", style="cyan", no_wrap=True, min_width=20)
    table.add_column("Result", justify="center")
    table.add_column("Faith.", justify="right")
    table.add_column("Relev.", justify="right")
    table.add_column("Ctx.", justify="right")
    table.add_column("Retries", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("Failures", max_width=55)

    for cr in suite.case_results:
        faith = next((jr.score for jr in cr.judge_results if jr.judge_type == "llm_faithfulness"), None)
        relev = next((jr.score for jr in cr.judge_results if jr.judge_type == "llm_relevance"), None)
        ctx   = next((jr.score for jr in cr.judge_results if jr.judge_type == "llm_context_relevance"), None)

        def _fmt(s: float | None) -> str:
            if s is None:
                return "[dim]—[/dim]"
            c = _score_color(s)
            return f"[{c}]{s:.2f}[/{c}]"

        failures = cr.failures
        failure_text = "\n".join(failures[:3]) if failures else "[dim]—[/dim]"
        if len(failures) > 3:
            failure_text += f"\n[dim]+{len(failures)-3} more[/dim]"

        table.add_row(
            cr.case_id,
            _pass_badge(cr.passed),
            _fmt(faith),
            _fmt(relev),
            _fmt(ctx),
            str(cr.retrieval_attempts),
            f"{cr.latency_ms:.0f}ms",
            failure_text,
        )

    # Subtitle stats line
    rate = suite.pass_rate
    rate_color = "green" if rate >= 0.8 else "yellow" if rate >= 0.5 else "red"
    stats = [
        f"Pass rate: [{rate_color}]{rate:.0%}[/{rate_color}] "
        f"({sum(1 for r in suite.case_results if r.passed)}/{len(suite.case_results)})"
    ]
    if suite.avg_faithfulness is not None:
        c = _score_color(suite.avg_faithfulness)
        stats.append(f"Avg faith: [{c}]{suite.avg_faithfulness:.2f}[/{c}]")
    if suite.avg_relevance is not None:
        c = _score_color(suite.avg_relevance)
        stats.append(f"Avg relev: [{c}]{suite.avg_relevance:.2f}[/{c}]")
    stats.append(f"Avg latency: {suite.avg_latency_ms:.0f}ms")

    title = f"[bold]{suite.suite_name}[/bold]"
    if suite.description:
        title += f"  [dim]{suite.description}[/dim]"

    console.print(Panel(table, title=title, subtitle="  |  ".join(stats)))


def print_overall_summary(suites: list[SuiteResult], regressions: list[str] | None = None) -> None:
    """Print the overall pass/fail summary across all suites."""
    total_cases  = sum(len(s.case_results) for s in suites)
    total_passed = sum(sum(1 for r in s.case_results if r.passed) for s in suites)
    overall = total_passed / total_cases if total_cases else 0.0
    color = "green" if overall >= 0.8 else "yellow" if overall >= 0.5 else "red"

    console.print(
        f"\n[bold]Overall:[/bold] [{color}]{overall:.0%}[/{color}] "
        f"({total_passed}/{total_cases} cases passed across {len(suites)} suite(s))\n"
    )

    if regressions:
        console.print("[bold red]Regressions detected:[/bold red]")
        for r in regressions:
            console.print(f"  [red]•[/red] {r}")
        console.print()
