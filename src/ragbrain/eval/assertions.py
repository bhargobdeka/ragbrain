"""Deterministic assertion types for RAG evaluation.

Each assertion takes the raw result dict from agents.graph.query() and a
config dict from the YAML test case, and returns an AssertionResult.

Supported types:
    answer_not_empty         — answer string must be non-empty
    has_sources              — source count >= min_count
    contains_keywords        — answer contains >= min_match of keywords
    not_contains             — answer must NOT contain any of strings
    retrieval_attempts_max   — retrieval_attempts <= max
    hallucination_check_passes — hallucination_check == "grounded"
    grade_result             — grade_result == expected
    regex                    — answer matches regex pattern
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class AssertionResult:
    assertion_type: str
    passed: bool
    score: float | None = None
    message: str = ""


# ---------------------------------------------------------------------------
# Individual assertion functions
# ---------------------------------------------------------------------------

def _assert_answer_not_empty(result: dict, config: dict) -> AssertionResult:
    answer = result.get("answer", "").strip()
    passed = bool(answer)
    return AssertionResult(
        "answer_not_empty", passed,
        message="" if passed else "Answer is empty",
    )


def _assert_has_sources(result: dict, config: dict) -> AssertionResult:
    min_count = config.get("min_count", 1)
    sources = result.get("sources", [])
    passed = len(sources) >= min_count
    return AssertionResult(
        "has_sources", passed,
        message=f"Got {len(sources)} source(s), expected >= {min_count}",
    )


def _assert_contains_keywords(result: dict, config: dict) -> AssertionResult:
    keywords = config.get("keywords", [])
    min_match = config.get("min_match", 1)
    answer = result.get("answer", "").lower()
    matched = [kw for kw in keywords if kw.lower() in answer]
    passed = len(matched) >= min_match
    return AssertionResult(
        "contains_keywords", passed,
        score=len(matched) / max(len(keywords), 1),
        message=f"Matched {len(matched)}/{len(keywords)}: {matched}",
    )


def _assert_not_contains(result: dict, config: dict) -> AssertionResult:
    strings = config.get("strings", [])
    answer = result.get("answer", "").lower()
    found = [s for s in strings if s.lower() in answer]
    passed = not found
    return AssertionResult(
        "not_contains", passed,
        message=f"Found forbidden strings: {found}" if found else "",
    )


def _assert_retrieval_attempts_max(result: dict, config: dict) -> AssertionResult:
    max_attempts = config.get("max", 2)
    attempts = result.get("retrieval_attempts", 0)
    passed = attempts <= max_attempts
    return AssertionResult(
        "retrieval_attempts_max", passed,
        message=f"Used {attempts} retrieval attempt(s), max allowed {max_attempts}",
    )


def _assert_hallucination_check_passes(result: dict, config: dict) -> AssertionResult:
    check = result.get("hallucination_check", "")
    passed = check == "grounded"
    return AssertionResult(
        "hallucination_check_passes", passed,
        message=f"hallucination_check={check!r}",
    )


def _assert_grade_result(result: dict, config: dict) -> AssertionResult:
    expected = config.get("expected", "relevant")
    actual = result.get("grade_result", "")
    passed = actual == expected
    return AssertionResult(
        "grade_result", passed,
        message=f"Expected grade_result={expected!r}, got {actual!r}",
    )


def _assert_regex(result: dict, config: dict) -> AssertionResult:
    pattern = config.get("pattern", "")
    flags_str = config.get("flags", "")
    flags = re.IGNORECASE if "i" in flags_str else 0
    answer = result.get("answer", "")
    matched = bool(re.search(pattern, answer, flags))
    return AssertionResult(
        "regex", matched,
        message=f"Pattern {pattern!r} {'matched' if matched else 'not found'} in answer",
    )


# ---------------------------------------------------------------------------
# Registry and dispatcher
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, callable] = {
    "answer_not_empty":           _assert_answer_not_empty,
    "has_sources":                _assert_has_sources,
    "contains_keywords":          _assert_contains_keywords,
    "not_contains":               _assert_not_contains,
    "retrieval_attempts_max":     _assert_retrieval_attempts_max,
    "hallucination_check_passes": _assert_hallucination_check_passes,
    "grade_result":               _assert_grade_result,
    "regex":                      _assert_regex,
}


def run_assertion(result: dict, assertion_config: dict) -> AssertionResult:
    """Dispatch to the appropriate assertion function."""
    atype = assertion_config.get("type", "")
    fn = _REGISTRY.get(atype)
    if fn is None:
        return AssertionResult(
            atype, False,
            message=f"Unknown assertion type: {atype!r}. "
                    f"Valid types: {sorted(_REGISTRY)}",
        )
    return fn(result, assertion_config)
