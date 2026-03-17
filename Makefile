.PHONY: help setup install qdrant-up qdrant-down ingest query digest serve test lint format clean

PYTHON := python
PIP := pip
RAGBRAIN := ragbrain

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---- Setup --------------------------------------------------

setup: install qdrant-up data-dirs ## Full first-time setup: install deps + start Qdrant + create data dirs
	@echo ""
	@echo "\033[32m✓ RAGBrain setup complete!\033[0m"
	@echo "Next steps:"
	@echo "  1. Copy .env.example to .env and fill in your ANTHROPIC_API_KEY"
	@echo "  2. Run 'make ingest' to index your first document"
	@echo "  3. Run 'make query Q=\"your question\"' to query"
	@echo "  4. Run 'make serve' to start the Telegram bot"

install: ## Install Python dependencies
	$(PIP) install -e ".[dev]"

data-dirs: ## Create required local data directories
	@mkdir -p inbox data
	@touch inbox/.gitkeep data/.gitkeep
	@echo "Created inbox/ and data/ directories"

# ---- Qdrant ------------------------------------------------

qdrant-up: ## Start Qdrant vector database (Docker)
	docker compose up -d qdrant
	@echo "Waiting for Qdrant to be ready..."
	@sleep 3
	@echo "\033[32m✓ Qdrant running at http://localhost:6333\033[0m"

qdrant-down: ## Stop Qdrant
	docker compose down

qdrant-reset: ## Wipe Qdrant data (⚠ destructive)
	docker compose down -v
	@echo "\033[33m⚠ Qdrant data wiped\033[0m"

# ---- Core commands -----------------------------------------

ingest: ## Ingest a file or URL (usage: make ingest PATH=./mybook.pdf or make ingest URL=https://...)
ifdef PATH
	$(RAGBRAIN) ingest "$(PATH)"
else ifdef URL
	$(RAGBRAIN) ingest "$(URL)"
else
	@echo "Usage: make ingest PATH=./mybook.pdf"
	@echo "       make ingest URL=https://example.substack.com/p/article"
	@exit 1
endif

query: ## Query the knowledge base (usage: make query Q="your question")
ifndef Q
	@echo "Usage: make query Q=\"your question\""
	@exit 1
endif
	$(RAGBRAIN) query "$(Q)"

digest: ## Generate and print today's digest
	$(RAGBRAIN) digest

serve: ## Start the Telegram bot (requires RAGBRAIN_TELEGRAM_BOT_TOKEN in .env)
	$(RAGBRAIN) serve

# ---- RSS pipeline ------------------------------------------

fetch-articles: ## Fetch and summarize articles from configured RSS feeds
	$(RAGBRAIN) fetch-articles

# ---- Development -------------------------------------------

test: ## Run tests
	pytest tests/ -v --cov=ragbrain --cov-report=term-missing

lint: ## Run linter
	ruff check src/ tests/

format: ## Format code
	ruff format src/ tests/

type-check: ## Run type checker
	mypy src/ragbrain/

clean: ## Remove build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage htmlcov/ .mypy_cache/
	@echo "\033[32m✓ Cleaned\033[0m"
