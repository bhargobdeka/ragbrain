"""EvalRunner — loads YAML test suites and runs them against the RAG pipeline.

A test suite is a YAML file in tests/eval/ with this structure:

    suite: "Suite name"
    description: "What this tests"
    features: ["tag-a", "tag-b"]   # suite-level feature tags
    test_cases:
      - id: unique-case-id
        query: "Natural language question"
        features: ["tag-a"]        # case-level feature tags (optional)
        assertions:
          - type: answer_not_empty
          - type: has_sources
            min_count: 1
          - type: contains_keywords
            keywords: ["rlhf", "reward"]
            min_match: 1
          - type: llm_faithfulness
            threshold: 0.7
          - type: llm_relevance
            threshold: 0.7
          - type: llm_context_relevance
            threshold: 0.6
          - type: retrieval_attempts_max
            max: 2
          - type: hallucination_check_passes
          - type: not_contains
            strings: ["I don't know", "I cannot"]
          - type: regex
            pattern: "\\d+"
            flags: "i"
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ragbrain.eval.assertions import AssertionResult, run_assertion
from ragbrain.eval.judges import JudgeResult, run_judge

logger = logging.getLogger(__name__)

_LLM_ASSERTION_TYPES = {"llm_faithfulness", "llm_relevance", "llm_context_relevance"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvalCase:
    id: str
    query: str
    assertions: list[dict]
    features: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class CaseResult:
    case_id: str
    query: str
    answer: str
    sources: list[dict]
    assertion_results: list[AssertionResult]
    judge_results: list[JudgeResult]
    passed: bool
    latency_ms: float
    retrieval_attempts: int
    hallucination_check: str
    error: str | None = None

    @property
    def failures(self) -> list[str]:
        msgs = [
            f"{ar.assertion_type}: {ar.message}"
            for ar in self.assertion_results if not ar.passed
        ] + [
            f"{jr.judge_type}: {jr.score:.2f} < threshold ({jr.reason[:60]})"
            for jr in self.judge_results if not jr.passed
        ]
        return msgs


@dataclass
class SuiteResult:
    suite_name: str
    description: str
    case_results: list[CaseResult]

    @property
    def pass_rate(self) -> float:
        if not self.case_results:
            return 0.0
        return sum(1 for r in self.case_results if r.passed) / len(self.case_results)

    @property
    def avg_faithfulness(self) -> float | None:
        scores = [
            jr.score for r in self.case_results
            for jr in r.judge_results if jr.judge_type == "llm_faithfulness"
        ]
        return sum(scores) / len(scores) if scores else None

    @property
    def avg_relevance(self) -> float | None:
        scores = [
            jr.score for r in self.case_results
            for jr in r.judge_results if jr.judge_type == "llm_relevance"
        ]
        return sum(scores) / len(scores) if scores else None

    @property
    def avg_latency_ms(self) -> float:
        if not self.case_results:
            return 0.0
        return sum(r.latency_ms for r in self.case_results) / len(self.case_results)

    def to_dict(self) -> dict:
        return {
            "suite": self.suite_name,
            "description": self.description,
            "pass_rate": self.pass_rate,
            "avg_faithfulness": self.avg_faithfulness,
            "avg_relevance": self.avg_relevance,
            "avg_latency_ms": self.avg_latency_ms,
            "cases": [
                {
                    "id": cr.case_id,
                    "query": cr.query,
                    "passed": cr.passed,
                    "latency_ms": round(cr.latency_ms, 1),
                    "retrieval_attempts": cr.retrieval_attempts,
                    "hallucination_check": cr.hallucination_check,
                    "answer_excerpt": cr.answer[:300],
                    "error": cr.error,
                    "assertion_results": [
                        {"type": ar.assertion_type, "passed": ar.passed,
                         "score": ar.score, "message": ar.message}
                        for ar in cr.assertion_results
                    ],
                    "judge_results": [
                        {"type": jr.judge_type, "score": round(jr.score, 3),
                         "passed": jr.passed, "reason": jr.reason}
                        for jr in cr.judge_results
                    ],
                }
                for cr in self.case_results
            ],
        }


# ---------------------------------------------------------------------------
# EvalRunner
# ---------------------------------------------------------------------------

class EvalRunner:
    """Runs YAML-defined evaluation suites against the RAG pipeline."""

    def __init__(self, fail_fast: bool = False):
        self.fail_fast = fail_fast

    # ------------------------------------------------------------------
    # Suite loading
    # ------------------------------------------------------------------

    def _load_suite(self, path: Path) -> tuple[str, str, list[str], list[EvalCase]]:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        suite_name   = data.get("suite", path.stem)
        description  = data.get("description", "")
        suite_features = data.get("features", [])

        cases = []
        for tc in data.get("test_cases", []):
            cases.append(EvalCase(
                id=tc["id"],
                query=tc["query"],
                assertions=tc.get("assertions", []),
                features=tc.get("features", []) + suite_features,
                description=tc.get("description", ""),
            ))

        return suite_name, description, suite_features, cases

    # ------------------------------------------------------------------
    # Single case execution
    # ------------------------------------------------------------------

    def _run_case(self, case: EvalCase) -> CaseResult:
        from ragbrain.agents.graph import query as rag_query
        from ragbrain.config import settings

        # Give this trace a descriptive name in LangSmith so it's easy to find.
        settings.setup_tracing()

        t0 = time.perf_counter()
        error: str | None = None
        try:
            result = rag_query(case.query)
            error = result.get("error")
        except Exception as e:
            logger.exception("Query error on case %s", case.id)
            result = {
                "answer": "", "sources": [],
                "retrieval_attempts": 0, "hallucination_check": "",
                "grade_result": "",
            }
            error = str(e)
        latency_ms = (time.perf_counter() - t0) * 1000

        answer  = result.get("answer", "")
        sources = result.get("sources", [])

        assertion_results: list[AssertionResult] = []
        judge_results: list[JudgeResult] = []

        for ac in case.assertions:
            atype = ac.get("type", "")
            if atype in _LLM_ASSERTION_TYPES:
                threshold = ac.get("threshold", 0.7)
                jr = run_judge(atype, case.query, answer, sources, threshold)
                if jr is not None:
                    judge_results.append(jr)
            else:
                assertion_results.append(run_assertion(result, ac))

        passed = (
            all(ar.passed for ar in assertion_results)
            and all(jr.passed for jr in judge_results)
            and error is None
        )

        return CaseResult(
            case_id=case.id,
            query=case.query,
            answer=answer,
            sources=sources,
            assertion_results=assertion_results,
            judge_results=judge_results,
            passed=passed,
            latency_ms=latency_ms,
            retrieval_attempts=result.get("retrieval_attempts", 0),
            hallucination_check=result.get("hallucination_check", ""),
            error=error,
        )

    # ------------------------------------------------------------------
    # Suite and multi-suite runners
    # ------------------------------------------------------------------

    def run_suite(
        self,
        path: Path,
        feature_filter: str | None = None,
    ) -> SuiteResult:
        """Run a single YAML eval suite."""
        suite_name, description, suite_features, cases = self._load_suite(path)

        if feature_filter:
            cases = [c for c in cases if feature_filter in c.features]
            if not cases:
                logger.info(
                    "Suite %s has no cases matching feature %r — skipping.",
                    suite_name, feature_filter,
                )

        case_results: list[CaseResult] = []
        for case in cases:
            logger.info("  Running case: %s", case.id)
            cr = self._run_case(case)
            case_results.append(cr)
            if self.fail_fast and not cr.passed:
                logger.warning("Fail-fast triggered at case %s", case.id)
                break

        return SuiteResult(
            suite_name=suite_name,
            description=description,
            case_results=case_results,
        )

    def run_all(
        self,
        eval_dir: Path,
        feature_filter: str | None = None,
        exclude_files: list[str] | None = None,
    ) -> list[SuiteResult]:
        """Run all YAML suites in eval_dir (sorted, skipping excluded files)."""
        exclude = set(exclude_files or [])
        yaml_files = sorted(f for f in eval_dir.glob("*.yaml") if f.name not in exclude)

        if not yaml_files:
            logger.warning("No YAML eval suites found in %s", eval_dir)
            return []

        results: list[SuiteResult] = []
        for path in yaml_files:
            logger.info("Suite: %s", path.stem)
            suite_result = self.run_suite(path, feature_filter=feature_filter)
            results.append(suite_result)
            if self.fail_fast and any(not r.passed for r in suite_result.case_results):
                break

        return results

    # ------------------------------------------------------------------
    # Baseline comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare_baseline(
        current: list[SuiteResult],
        baseline_path: Path,
    ) -> list[str]:
        """Return a list of regression warnings vs a saved baseline JSON."""
        if not baseline_path.exists():
            return [f"Baseline file {baseline_path} not found — nothing to compare."]

        with open(baseline_path, encoding="utf-8") as f:
            baseline = json.load(f)

        warnings: list[str] = []
        baseline_by_suite = {s["suite"]: s for s in baseline}

        for suite in current:
            b = baseline_by_suite.get(suite.suite_name)
            if b is None:
                continue

            delta_pass = suite.pass_rate - b.get("pass_rate", 0.0)
            if delta_pass < -0.1:
                warnings.append(
                    f"[REGRESSION] {suite.suite_name}: pass rate dropped "
                    f"{b['pass_rate']:.0%} → {suite.pass_rate:.0%}"
                )

            if suite.avg_faithfulness is not None and b.get("avg_faithfulness"):
                delta_f = suite.avg_faithfulness - b["avg_faithfulness"]
                if delta_f < -0.1:
                    warnings.append(
                        f"[REGRESSION] {suite.suite_name}: faithfulness dropped "
                        f"{b['avg_faithfulness']:.2f} → {suite.avg_faithfulness:.2f}"
                    )

        return warnings

    # ------------------------------------------------------------------
    # JSON persistence
    # ------------------------------------------------------------------

    @staticmethod
    def default_results_dir(eval_dir: Path) -> Path:
        """Return the auto-save directory, creating it if needed."""
        results_dir = eval_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir

    @staticmethod
    def save_results(suites: list[SuiteResult], path: Path) -> None:
        """Save results as JSON for CI/CD or baseline comparison."""
        data = [s.to_dict() for s in suites]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def auto_save(suites: list[SuiteResult], eval_dir: Path, label: str = "eval") -> Path:
        """Auto-save results to tests/eval/results/<label>_YYYYMMDD_HHMMSS.json."""
        from datetime import datetime, timezone

        results_dir = EvalRunner.default_results_dir(eval_dir)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = results_dir / f"{label}_{ts}.json"
        EvalRunner.save_results(suites, path)
        return path

    @staticmethod
    def load_history(eval_dir: Path, label: str = "eval", last_n: int = 10) -> list[dict]:
        """Load the last N saved result files, newest first."""
        results_dir = eval_dir / "results"
        if not results_dir.exists():
            return []
        files = sorted(
            results_dir.glob(f"{label}_*.json"),
            reverse=True,
        )[:last_n]
        history = []
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                history.append({"file": f.name, "suites": data})
            except Exception:
                pass
        return history
