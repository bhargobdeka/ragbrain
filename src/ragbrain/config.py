"""Central configuration using pydantic-settings.

All settings can be overridden via environment variables or a .env file.
Prefix: RAGBRAIN_  (e.g. RAGBRAIN_QDRANT_URL)

Note: rss_feeds and interests are stored as raw comma-separated strings
internally to avoid pydantic-settings v2 attempting json.loads() on them.
Access them via the .rss_feeds / .interests properties.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_comma_str(v: str) -> list[str]:
    """Parse a comma-separated env string into a list, handling empty values."""
    if not v or not v.strip():
        return []
    return [item.strip() for item in v.split(",") if item.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAGBRAIN_",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- LLM -------------------------------------------------------
    # ANTHROPIC_API_KEY has no prefix (standard env var name)
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    llm_fast_model: str = "claude-haiku-4-5"
    llm_temperature: float = 0.0

    # ---- Embeddings (local) ----------------------------------------
    embedding_model: str = "all-mpnet-base-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_dim: int = 768
    # Code-specific embeddings — uses microsoft/unixcoder-base (768-dim, ~500 MB)
    # Set use_code_encoder=false to skip and use mpnet for all content
    code_embedding_model: str = "microsoft/unixcoder-base"
    code_embedding_dim: int = 768
    use_code_encoder: bool = True

    # ---- Qdrant ----------------------------------------------------
    # mode: "local"  → no Docker, stores data on disk at qdrant_local_path
    #       "server" → connects to a running Qdrant server at qdrant_url
    qdrant_mode: Literal["local", "server"] = "local"
    qdrant_local_path: str = "~/.ragbrain/qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "ragbrain_default"

    # ---- Retrieval -------------------------------------------------
    retrieval_top_k: int = 20       # candidates from dense + sparse each
    rerank_top_n: int = 5           # final results after reranking
    dedup_threshold: float = 0.85   # cosine similarity for dedup
    min_relevance_score: int = 6    # 1-10 scale

    # ---- Agentic RAG -----------------------------------------------
    max_retries: int = 2
    max_hallucination_retries: int = 2

    # ---- Telegram --------------------------------------------------
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # ---- Slack (for reading AI news + posting review output) ------
    slack_bot_token: str = ""
    slack_channel_id: str = ""      # DM or channel ID where news arrives
    slack_post_channel_id: str = "" # where to post review results (defaults to same)
    slack_lookback_hours: int = 24

    # ---- Article pipeline ------------------------------------------
    # Stored as raw string to avoid pydantic-settings json.loads() on list fields
    _rss_feeds_raw: str = ""
    rss_feeds_str: str = ""            # set via RAGBRAIN_RSS_FEEDS in .env
    article_lookback_hours: int = 24
    morning_cron: str = "0 8 * * *"
    evening_cron: str = "0 19 * * *"

    # ---- Book pipeline ---------------------------------------------
    inbox_dir: str = "./inbox"
    book_state_file: str = "./data/book_state.json"

    # ---- User interests (for relevance scoring) --------------------
    # Stored as raw string to avoid pydantic-settings json.loads() on list fields
    interests_str: str = "RAG systems,LLM agents,machine learning"

    # ---- Parsed list properties ------------------------------------

    @property
    def rss_feeds(self) -> list[str]:
        return _parse_comma_str(self.rss_feeds_str)

    @property
    def interests(self) -> list[str]:
        return _parse_comma_str(self.interests_str)

    @property
    def interests_text(self) -> str:
        return ", ".join(self.interests)

    # ---- LLM factories ---------------------------------------------

    def get_llm(self):
        """Return a configured LangChain LLM instance."""
        if self.llm_provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=self.llm_model,
                temperature=self.llm_temperature,
                api_key=self.anthropic_api_key,  # type: ignore[arg-type]
            )
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=self.llm_model,
            temperature=self.llm_temperature,
            api_key=self.openai_api_key,  # type: ignore[arg-type]
        )

    def get_fast_llm(self):
        """Return a cheaper/faster LLM for grading and rewriting."""
        if self.llm_provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=self.llm_fast_model,
                temperature=self.llm_temperature,
                api_key=self.anthropic_api_key,  # type: ignore[arg-type]
            )
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=self.llm_fast_model,
            temperature=self.llm_temperature,
            api_key=self.openai_api_key,  # type: ignore[arg-type]
        )


# Module-level singleton -- import this everywhere
settings = Settings()
