"""High-level pipelines for articles, books, and architecture review.

Imports are lazy to avoid pulling heavy dependencies (sentence-transformers,
torch, etc.) when only one pipeline is needed.
"""

__all__ = ["ArticlesPipeline", "BooksPipeline", "run_review", "run_upgrade_planner"]


def __getattr__(name: str):
    if name == "ArticlesPipeline":
        from ragbrain.pipelines.articles import ArticlesPipeline
        return ArticlesPipeline
    if name == "BooksPipeline":
        from ragbrain.pipelines.books import BooksPipeline
        return BooksPipeline
    if name == "run_review":
        from ragbrain.pipelines.architecture_review import run_review
        return run_review
    if name == "run_upgrade_planner":
        from ragbrain.pipelines.upgrade_planner import run_upgrade_planner
        return run_upgrade_planner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
