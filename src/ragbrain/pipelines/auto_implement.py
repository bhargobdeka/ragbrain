"""AutoImplementer — autonomous code-change agent with eval gate.

Accepts an approved Proposal and:

1. Validates it is in the safe-change scope.
2. Uses Claude to generate the exact file diff.
3. Writes the change to disk.
4. Runs `ragbrain eval --suite rag_basic` and checks the pass rate.
5a. If pass rate ≥ threshold → git commit (tagged [ragbrain-auto]).
5b. If pass rate <  threshold → git checkout (revert) the file.
6. Returns a human-readable result string for Telegram.

Safe scope (only auto-implement these targets):
    - tests/eval/         New/updated eval YAML test cases
    - retrieval/intent.py Query intent keywords update
    - config.py           Config default value changes
    - ARCHITECTURE.md     Documentation update
    - architecture-state.md  Documentation/state update
"""

from __future__ import annotations

import json
import logging
import subprocess
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from ragbrain.config import settings

if TYPE_CHECKING:
    from ragbrain.pipelines.proposals import Proposal

logger = logging.getLogger(__name__)

# Root of the git repo (two levels above src/)
_REPO_ROOT = Path(__file__).resolve().parents[4]

# ---- Safe-scope definitions ----------------------------------------------

_SAFE_TARGETS: list[str] = [
    "tests/eval/",
    "src/ragbrain/retrieval/intent.py",
    "src/ragbrain/config.py",
    "ARCHITECTURE.md",
    "architecture-state.md",
]

_FORBIDDEN_PATTERNS: list[str] = [
    "src/ragbrain/agents/",
    "src/ragbrain/vectorstore/",
    "src/ragbrain/ingestion/",
    "src/ragbrain/agents/graph.py",
]


def _is_safe_path(rel_path: str) -> bool:
    """Return True if the relative path is in the safe-change scope."""
    # Reject anything matching forbidden patterns first
    for forbidden in _FORBIDDEN_PATTERNS:
        if rel_path.startswith(forbidden):
            return False
    # Must match at least one safe target
    for safe in _SAFE_TARGETS:
        if rel_path.startswith(safe) or rel_path == safe.rstrip("/"):
            return True
    return False


# ---- LLM code-generation prompt ------------------------------------------

_GENERATE_PROMPT = """You are an autonomous code-change agent for the RAGBrain project.

You will receive an approved architecture upgrade proposal. Your job is to
generate the exact file content for a SINGLE file that implements this change.

The change MUST be limited to one of these safe targets:
  - tests/eval/ (new or updated YAML eval test cases)
  - src/ragbrain/retrieval/intent.py (keyword updates only)
  - src/ragbrain/config.py (default value changes only)
  - ARCHITECTURE.md (documentation update)
  - architecture-state.md (state/note update)

Rules:
1. Output ONLY valid JSON with exactly two keys:
   - "target_file": relative path from repo root (e.g. "tests/eval/new_suite.yaml")
   - "content": the COMPLETE file content (not a diff, the entire new file)
2. If the proposal cannot be safely implemented in the allowed scope, output:
   {{"target_file": null, "content": "REASON: <why it cannot be auto-implemented>"}}
3. For tests/eval/ files: generate a complete YAML test suite with at least 2 test cases.
4. For retrieval/intent.py: only modify keyword sets, do not change function logic.
5. For config.py: only change Field default= values, do not restructure code.
6. For ARCHITECTURE.md / architecture-state.md: append a new dated section.
7. Make the change minimal and focused. Don't rewrite entire files unnecessarily.

Proposal to implement:

Title: {title}
Component: {component}
Description: {description}
Implementation plan: {implementation_plan}
News signal: {news_signal}

Current content of target file (if it exists):
{current_content}

Output your JSON response now. No markdown fences, no explanation — raw JSON only."""


def _get_current_content(target_file: str) -> str:
    """Read current file content, returning empty string if missing."""
    full_path = _REPO_ROOT / target_file
    if full_path.exists():
        content = full_path.read_text(encoding="utf-8")
        # Truncate very large files so the prompt stays manageable
        return content[:4000] + ("\n\n...[truncated]" if len(content) > 4000 else "")
    return "(File does not exist yet — you are creating it.)"


def _run_eval_suite(suite_name: str = "rag_basic") -> tuple[float, str]:
    """Run a named eval suite via the ragbrain CLI and return (pass_rate, output).

    Returns (0.0, error_message) if the suite cannot be run.
    """
    try:
        result = subprocess.run(
            ["ragbrain", "eval", "--suite", suite_name],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(_REPO_ROOT),
        )
        output = result.stdout + result.stderr
        # Parse pass rate from eval output: "Passed X / Y"
        import re
        m = re.search(r"Passed\s+(\d+)\s*/\s*(\d+)", output)
        if m:
            passed, total = int(m.group(1)), int(m.group(2))
            rate = passed / total if total > 0 else 0.0
            return rate, output
        # If the process exited cleanly but no match, treat as 0
        return 0.0, output or "No eval output."
    except subprocess.TimeoutExpired:
        return 0.0, "Eval timed out after 300 seconds."
    except FileNotFoundError:
        return 0.0, "ragbrain CLI not found — cannot run eval."
    except Exception as exc:
        return 0.0, f"Eval run error: {exc}"


def _git_commit(message: str) -> tuple[bool, str]:
    """Stage all changes and commit. Returns (success, output)."""
    try:
        add = subprocess.run(
            ["git", "add", "-A"],
            capture_output=True, text=True, cwd=str(_REPO_ROOT)
        )
        commit = subprocess.run(
            ["git", "commit", "-m", message],
            capture_output=True, text=True, cwd=str(_REPO_ROOT)
        )
        success = commit.returncode == 0
        output = commit.stdout + commit.stderr
        return success, output
    except Exception as exc:
        return False, str(exc)


def _git_checkout(file_path: str) -> tuple[bool, str]:
    """Revert a single file to the last committed state."""
    try:
        result = subprocess.run(
            ["git", "checkout", "--", file_path],
            capture_output=True, text=True, cwd=str(_REPO_ROOT)
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as exc:
        return False, str(exc)


def _get_short_diff(file_path: str) -> str:
    """Return a short git diff for the file (last commit)."""
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD~1", "--", file_path],
            capture_output=True, text=True, cwd=str(_REPO_ROOT)
        )
        diff = result.stdout
        return diff[:1000] + ("..." if len(diff) > 1000 else "")
    except Exception:
        return ""


# ---- Main entry point ----------------------------------------------------

class AutoImplementResult:
    def __init__(
        self,
        success: bool,
        target_file: str | None,
        commit_sha: str,
        summary: str,
        telegram_message: str,
    ) -> None:
        self.success = success
        self.target_file = target_file
        self.commit_sha = commit_sha
        self.summary = summary
        self.telegram_message = telegram_message


def implement_proposal(proposal: "Proposal") -> AutoImplementResult:
    """Auto-implement an approved proposal.

    Returns an AutoImplementResult with the outcome and a Telegram-ready message.
    """
    threshold = settings.eval_pass_threshold
    llm = settings.get_llm()

    # ---- Step 1: Ask LLM for the change ----------------------------------
    prompt_text = _GENERATE_PROMPT.format(
        title=proposal.title,
        component=proposal.component,
        description=proposal.description,
        implementation_plan=proposal.implementation_plan,
        news_signal=proposal.news_signal,
        current_content="(To be filled after target_file is known)",
    )

    logger.info("Generating implementation for proposal %s: %s", proposal.id, proposal.title)

    try:
        response = llm.invoke(prompt_text)
        raw = response.content if hasattr(response, "content") else str(response)

        # Strip accidental markdown fences
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)
        target_file: str | None = data.get("target_file")
        file_content: str = data.get("content", "")
    except json.JSONDecodeError as exc:
        msg = f"LLM returned invalid JSON: {exc}\nRaw: {raw[:300]}"
        logger.error(msg)
        return AutoImplementResult(
            success=False, target_file=None, commit_sha="",
            summary=msg,
            telegram_message=f"❌ <b>Auto-implementation failed</b>\n\nCould not parse LLM output.\n<i>{msg[:300]}</i>",
        )
    except Exception as exc:
        logger.exception("LLM generation failed")
        return AutoImplementResult(
            success=False, target_file=None, commit_sha="",
            summary=str(exc),
            telegram_message=f"❌ <b>Auto-implementation failed</b>\n\nLLM error: {exc}",
        )

    # ---- Step 2: LLM says it can't safely implement this -----------------
    if target_file is None:
        reason = file_content.replace("REASON:", "").strip()
        return AutoImplementResult(
            success=False, target_file=None, commit_sha="",
            summary=f"Out of scope: {reason}",
            telegram_message=(
                f"⚠️ <b>Auto-implementation skipped</b>\n\n"
                f"<i>Proposal #{proposal.id}: {proposal.title}</i>\n\n"
                f"This change requires manual implementation:\n{reason}"
            ),
        )

    # ---- Step 3: Safety check --------------------------------------------
    if not _is_safe_path(target_file):
        msg = f"Target file '{target_file}' is outside the safe-change scope."
        logger.warning(msg)
        return AutoImplementResult(
            success=False, target_file=target_file, commit_sha="",
            summary=msg,
            telegram_message=(
                f"🚫 <b>Blocked — out of safe scope</b>\n\n"
                f"<i>Proposal #{proposal.id}: {proposal.title}</i>\n\n"
                f"{msg}\n\nMake this change manually when you return."
            ),
        )

    # ---- Step 4: If we need to provide current content, do a second pass -
    # (The prompt above passes a placeholder; for a second iteration we'd
    # re-invoke with actual content.  For simplicity we apply the change
    # and let the eval gate decide.)
    full_path = _REPO_ROOT / target_file
    full_path.parent.mkdir(parents=True, exist_ok=True)

    # Back up original content
    original_content: str | None = None
    if full_path.exists():
        original_content = full_path.read_text(encoding="utf-8")

    # Write proposed change
    full_path.write_text(file_content, encoding="utf-8")
    logger.info("Wrote proposed change to %s", target_file)

    # ---- Step 5: Run eval gate -------------------------------------------
    pass_rate, eval_output = _run_eval_suite()
    logger.info("Eval pass rate: %.2f  (threshold: %.2f)", pass_rate, threshold)

    eval_summary = textwrap.shorten(eval_output, width=400, placeholder="...")

    if pass_rate >= threshold:
        # ---- Step 5a: Commit the change ----------------------------------
        commit_msg = (
            f"[ragbrain-auto] {proposal.title}\n\n"
            f"Proposal #{proposal.id} | Component: {proposal.component}\n"
            f"Eval pass rate: {pass_rate:.0%} (threshold: {threshold:.0%})"
        )
        success, commit_out = _git_commit(commit_msg)

        # Get short SHA
        sha_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(_REPO_ROOT)
        )
        sha = sha_result.stdout.strip()

        diff_text = _get_short_diff(target_file)
        diff_block = f"\n\n<pre>{diff_text}</pre>" if diff_text else ""

        tg_msg = (
            f"✅ <b>Auto-implemented & committed</b>\n\n"
            f"<i>{proposal.title}</i>\n"
            f"File: <code>{target_file}</code>\n"
            f"Eval: {pass_rate:.0%} pass rate  ✓\n"
            f"Commit: <code>{sha}</code>"
            f"{diff_block}"
        )
        return AutoImplementResult(
            success=True,
            target_file=target_file,
            commit_sha=sha,
            summary=f"Committed {target_file} — {pass_rate:.0%} eval pass rate",
            telegram_message=tg_msg,
        )
    else:
        # ---- Step 5b: Revert the change ----------------------------------
        if original_content is not None:
            full_path.write_text(original_content, encoding="utf-8")
            logger.info("Reverted %s — eval pass rate too low (%.2f)", target_file, pass_rate)
        else:
            full_path.unlink(missing_ok=True)

        tg_msg = (
            f"⚠️ <b>Auto-implementation reverted</b>\n\n"
            f"<i>{proposal.title}</i>\n"
            f"File: <code>{target_file}</code>\n"
            f"Eval: {pass_rate:.0%} pass rate — below {threshold:.0%} threshold\n\n"
            f"Change has been rolled back. Review and apply manually.\n\n"
            f"<i>Eval output:</i>\n<code>{eval_summary}</code>"
        )
        return AutoImplementResult(
            success=False,
            target_file=target_file,
            commit_sha="",
            summary=f"Reverted {target_file} — eval pass rate {pass_rate:.0%} < {threshold:.0%}",
            telegram_message=tg_msg,
        )
