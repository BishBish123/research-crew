.DEFAULT_GOAL := help
SHELL := /bin/bash
UV ?= uv

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Install dev + load deps
	$(UV) sync --extra dev --extra load

.PHONY: fmt
fmt: ## Format with ruff
	$(UV) run ruff format src tests

.PHONY: lint
lint: ## Lint with ruff
	$(UV) run ruff check src tests

.PHONY: typecheck
typecheck: ## mypy --strict
	$(UV) run mypy src

.PHONY: check
check: lint typecheck

.PHONY: test
test: ## Unit tests
	$(UV) run pytest -m "not integration"

.PHONY: test-integration
test-integration: ## Integration (needs Redis)
	@$(UV) run pytest -m integration; status=$$?; \
		if [ $$status -eq 5 ]; then \
			echo "no integration tests collected — skipping"; \
			exit 0; \
		fi; \
		exit $$status

.PHONY: test-all
test-all: ## All tests
	$(UV) run pytest

.PHONY: api
api: ## Run the FastAPI service locally (honours REDIS_PORT to derive REDIS_URL)
	@# Keep REDIS_PORT and REDIS_URL coherent: if the user pointed Redis
	@# at a non-default host port via REDIS_PORT (e.g. to dodge a 6379
	@# conflict with a host-installed redis-server), derive REDIS_URL
	@# from it automatically. An explicit REDIS_URL still wins so the
	@# override path is unchanged for advanced users.
	REDIS_URL=$${REDIS_URL:-redis://localhost:$${REDIS_PORT:-6379}/0} \
		$(UV) run research-api

.PHONY: up
up: ## Start Redis container (override port: REDIS_PORT=6390 make up)
	docker compose down --remove-orphans 2>/dev/null || true
	docker compose up -d --wait

.PHONY: down
down: ## Stop containers
	docker compose down -v

.PHONY: load
load: ## Run a Locust load test against the local API
	$(UV) run locust -f load/locustfile.py --host http://localhost:8000

.PHONY: clean
clean: ## Wipe caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist
	find . -name __pycache__ -type d -exec rm -rf {} +
