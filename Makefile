.PHONY: build run test lint format clean

# ── Docker ───────────────────────────────────────────────────────────────────
build:
	docker compose build

run:
	docker compose up -d

stop:
	docker compose down

logs:
	docker compose logs -f api

# ── Development ──────────────────────────────────────────────────────────────
dev:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

install:
	pip install -r requirements.txt
	pip install pytest pytest-asyncio anyio httpx ruff

# ── Quality ──────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

lint:
	ruff check .

format:
	ruff format .

check: lint test

# ── Cleanup ──────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
