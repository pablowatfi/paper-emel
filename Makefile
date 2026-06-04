.PHONY: install fmt lint test test-int eval ingest-local serve ui docker-up docker-down clean \
	research-chunking research-embedding research-retrieval research-reranker research-all \
	bootstrap-prod

install:
	uv sync --all-extras
	uv run pre-commit install

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy --strict src/

test:
	uv run pytest tests/unit/ -v --cov=src --cov-report=term

test-int:
	docker compose -f infra/docker-compose.yml up -d qdrant
	uv run pytest tests/integration/ -v
	docker compose -f infra/docker-compose.yml down

eval:
	uv run python -m arxiv_rag.eval \
		--golden eval/golden_set_v1.jsonl \
		--output eval/results/latest.json

ingest-local:
	uv run python -m arxiv_rag.ingestion.pipeline --days 7

# Phase R — Research & Ablations
research-chunking:
	uv run python research/harnesses/chunking_harness.py --output research/results/R01_chunking

research-embedding:
	uv run python research/harnesses/embedding_harness.py --output research/results/R02_embedding

research-retrieval:
	uv run python research/harnesses/retrieval_harness.py --output research/results/R03_retrieval

research-reranker:
	uv run python research/harnesses/reranker_harness.py --output research/results/R04_reranker

research-all: research-chunking research-embedding research-retrieval research-reranker

bootstrap-prod:
	uv run python -m arxiv_rag.ingestion.bootstrap

serve:
	uv run uvicorn arxiv_rag.api.app:app --reload --port 8000

ui:
	uv run python -m arxiv_rag.ui.gradio_app

docker-up:
	docker compose -f infra/docker-compose.yml up -d

docker-down:
	docker compose -f infra/docker-compose.yml down

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
