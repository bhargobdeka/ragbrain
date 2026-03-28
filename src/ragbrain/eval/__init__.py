"""RAGBrain evaluation harness.

Provides declarative YAML-driven test suites for quality evaluation and
red-team adversarial probing of the RAG pipeline.

Usage:
    ragbrain eval                          # run all quality suites
    ragbrain eval --suite rag_basic        # run one suite
    ragbrain eval --feature rag-core       # filter by feature tag
    ragbrain eval --red-team               # run adversarial probes
    ragbrain eval --output results.json    # save for CI/regression
"""

__all__ = ["EvalRunner", "run_red_team"]


def __getattr__(name: str):
    if name == "EvalRunner":
        from ragbrain.eval.runner import EvalRunner
        return EvalRunner
    if name == "run_red_team":
        from ragbrain.eval.red_team import run_red_team
        return run_red_team
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
