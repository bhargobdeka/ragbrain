"""High-level pipelines for articles and books."""

from ragbrain.pipelines.articles import ArticlesPipeline
from ragbrain.pipelines.books import BooksPipeline

__all__ = ["ArticlesPipeline", "BooksPipeline"]
