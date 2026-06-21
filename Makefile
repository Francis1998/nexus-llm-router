.PHONY: help install dev-install test lint type-check clean docker-build docker-run

# nexus-llm-router Makefile

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS=":.*?##"}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

dev-install:  ## Install with dev extras
	pip install -e ".[dev]"

test:  ## Run test suite
	pytest tests/ -v

lint:  ## Lint with ruff
	ruff check src/

type-check:  ## Type check with mypy
	mypy src/

clean:  ## Remove build artifacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage dist build *.egg-info

docker-build:  ## Build Docker image
	docker build -t nexus-llm-router:latest .

docker-run:  ## Run Docker container
	docker run --env-file .env -p 8000:8000 nexus-llm-router:latest
