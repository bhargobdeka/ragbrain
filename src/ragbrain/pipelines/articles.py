"""Articles pipeline: RSS fetch → dedup → relevance score → summarize → Digest.

Flow:
1. Fetch all configured RSS feeds for the last N hours
2. For each entry, fetch the full article text via WebExtractor
3. Deduplicate using cosine similarity on title embeddings
4. Score each article's relevance to user interests (LLM-as-judge, 1-10)
5. Keep articles above min_relevance_score
6. Summarize each kept article
7. Return a list of ArticleSummary objects
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from ragbrain.config import settings
from ragbrain.ingestion.extractors.rss import RSSExtractor
from ragbrain.ingestion.extractors.web import WebExtractor
from ragbrain.ingestion.pipeline import IngestionPipeline
from ragbrain.models import ArticleSummary, Document

if TYPE_CHECKING:
    pass

# ---- Prompts --------------------------------------------------------

_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a content relevance scorer. Given a user's interest profile and an article, "
        "rate the article's relevance on a scale of 1-10.\n\n"
        "User interests: {interests}\n\n"
        "Return a JSON with fields: score (int 1-10), reason (str).",
    ),
    (
        "human",
        "Article title: {title}\n\nArticle excerpt:\n{excerpt}",
    ),
])

_SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a technical content summarizer. Given an article, produce:\n"
        "1. A 3-bullet point summary (concise, information-dense)\n"
        "2. One key takeaway sentence\n"
        "3. A list of 1-3 topic tags\n\n"
        "Return JSON with: summary (str, use \\n between bullets), "
        "key_takeaway (str), topics (list[str]).",
    ),
    (
        "human",
        "Title: {title}\n\nContent:\n{content}",
    ),
])


# ---- Pydantic output schemas ----------------------------------------

class RelevanceScore(BaseModel):
    score: int = Field(ge=1, le=10)
    reason: str = ""


class ArticleSummaryOutput(BaseModel):
    summary: str
    key_takeaway: str
    topics: list[str] = Field(default_factory=list)


# ---- Pipeline -------------------------------------------------------

class ArticlesPipeline:
    """Fetch, filter, and summarize articles from RSS feeds."""

    def __init__(self, user_id: str | None = None) -> None:
        self._user_id = user_id
        self._rss = RSSExtractor()
        self._web = WebExtractor()
        self._ingestor = IngestionPipeline(user_id=user_id)
        self._encoder: SentenceTransformer | None = None

    @property
    def encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            self._encoder = SentenceTransformer(settings.embedding_model)
        return self._encoder

    def run(
        self,
        feed_urls: list[str] | None = None,
        lookback_hours: int | None = None,
        also_ingest: bool = True,
    ) -> list[ArticleSummary]:
        """Run the full articles pipeline.

        Args:
            feed_urls: RSS feed URLs to fetch. Defaults to settings.rss_feeds.
            lookback_hours: Hours to look back for new articles.
            also_ingest: If True, also index articles in Qdrant.

        Returns:
            List of ArticleSummary objects sorted by relevance score desc.
        """
        urls = feed_urls or settings.rss_feeds
        hours = lookback_hours or settings.article_lookback_hours
        since = datetime.now(tz=UTC) - timedelta(hours=hours)

        # Step 1: Fetch RSS entries
        documents = self._fetch_feeds(urls, since)
        if not documents:
            return []

        # Step 2: Deduplicate
        documents = self._deduplicate(documents)

        # Step 3: Fetch full text for each article
        documents = self._enrich_with_full_text(documents)

        # Step 4: Score relevance
        scored = self._score_relevance(documents)

        # Step 5: Filter and summarize
        summaries: list[ArticleSummary] = []
        for doc, score in scored:
            if score < settings.min_relevance_score:
                continue
            summary = self._summarize(doc, score)
            if summary:
                summaries.append(summary)

            # Optionally ingest into Qdrant for future RAG queries
            if also_ingest and doc.blocks:
                try:
                    self._ingestor.ingest_document(doc)
                except Exception:
                    pass

        summaries.sort(key=lambda s: s.relevance_score, reverse=True)
        return summaries

    def _fetch_feeds(self, feed_urls: list[str], since: datetime) -> list[Document]:
        docs: list[Document] = []
        for url in feed_urls:
            try:
                entries = self._rss.fetch_feed(url, since=since)
                docs.extend(entries)
            except Exception as e:
                print(f"[warn] Failed to fetch feed {url}: {e}")
        return docs

    def _deduplicate(self, documents: list[Document]) -> list[Document]:
        """Remove near-duplicate articles by title embedding similarity."""
        if len(documents) <= 1:
            return documents

        titles = [d.title or d.source_url for d in documents]
        embeddings = self.encoder.encode(titles, normalize_embeddings=True)

        keep = [True] * len(documents)
        for i in range(len(documents)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(documents)):
                if not keep[j]:
                    continue
                sim = float(np.dot(embeddings[i], embeddings[j]))
                if sim > settings.dedup_threshold:
                    keep[j] = False

        return [doc for doc, k in zip(documents, keep) if k]

    def _enrich_with_full_text(self, documents: list[Document]) -> list[Document]:
        """Fetch full article text for RSS entries that only have summaries."""
        enriched: list[Document] = []
        for doc in documents:
            if doc.source_url and (not doc.raw_text or len(doc.raw_text) < 200):
                try:
                    full_doc = self._web.extract(doc.source_url)
                    # Preserve RSS metadata
                    full_doc.published_at = doc.published_at or full_doc.published_at
                    full_doc.doc_id = doc.doc_id
                    enriched.append(full_doc)
                except Exception:
                    enriched.append(doc)
            else:
                enriched.append(doc)
        return enriched

    def _score_relevance(self, documents: list[Document]) -> list[tuple[Document, int]]:
        llm = settings.get_fast_llm()
        scorer = _RELEVANCE_PROMPT | llm.with_structured_output(RelevanceScore)
        results: list[tuple[Document, int]] = []

        for doc in documents:
            excerpt = (doc.raw_text or "")[:1000]
            if not excerpt:
                results.append((doc, 0))
                continue
            try:
                grade = scorer.invoke({
                    "interests": settings.interests_text,
                    "title": doc.title,
                    "excerpt": excerpt,
                })
                results.append((doc, grade.score))
            except Exception:
                results.append((doc, 5))  # neutral score on error

        return results

    def _summarize(self, doc: Document, relevance_score: int) -> ArticleSummary | None:
        llm = settings.get_llm()
        summarizer = _SUMMARIZE_PROMPT | llm.with_structured_output(ArticleSummaryOutput)

        content = (doc.raw_text or "")[:3000]
        if not content:
            return None

        try:
            output = summarizer.invoke({"title": doc.title, "content": content})
        except Exception as e:
            print(f"[warn] Summarization failed for {doc.title}: {e}")
            return None

        source_name = doc.source_url.split("/")[2] if doc.source_url else ""

        return ArticleSummary(
            title=doc.title,
            source_url=doc.source_url,
            source_name=source_name,
            published_at=doc.published_at,
            summary=output.summary,
            key_takeaway=output.key_takeaway,
            relevance_score=relevance_score,
            topics=output.topics,
        )
