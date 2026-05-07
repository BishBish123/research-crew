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

.PHONY: format-check
format-check: ## Verify ruff format would not rewrite anything (CI gate)
	$(UV) run ruff format --check src tests

.PHONY: check
check: lint format-check typecheck ## Lint + format-check + typecheck

.PHONY: test
test: ## Unit tests
	$(UV) run pytest -m "not integration"

.PHONY: test-integration
test-integration: ## Placeholder for future @pytest.mark.integration tests requiring external services
	@$(UV) run pytest -m integration; status=$$?; \
		if [ $$status -eq 5 ]; then \
			echo "no integration tests collected — placeholder target until real ones land" >&2; \
			exit 0; \
		fi; \
		exit $$status

.PHONY: test-integration-mock
test-integration-mock: ## Run real-mode adapter integration tests against local mock servers (no cloud services needed)
	$(UV) run pytest tests/test_real_adapter_integration.py -v

.PHONY: test-integration-all
test-integration-all: test-integration-mock test-integration ## Run all integration test suites

.PHONY: test-all
test-all: ## All tests
	$(UV) run pytest

.PHONY: test-streams
test-streams: ## Run Redis Streams store tests with RESEARCH_CREW_STORE=streams (uses fakeredis)
	RESEARCH_CREW_STORE=streams $(UV) run pytest tests/test_redis_streams_store.py -v

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
	@docker compose up -d --wait || (echo ""; echo "[hint] Port $${REDIS_PORT:-6379} may be in use. Try: REDIS_PORT=$$(( $${REDIS_PORT:-6379} + 1 )) make up"; echo ""; exit 1)

.PHONY: down
down: ## Stop containers
	docker compose down -v

.PHONY: pg-up
pg-up: ## Start Postgres container for run-history backend (override port: PG_PORT=5433 make pg-up)
	docker compose -f docker-compose.postgres.yml down --remove-orphans 2>/dev/null || true
	@docker compose -f docker-compose.postgres.yml up -d --wait || (echo ""; echo "[hint] Port $${PG_PORT:-5432} may be in use. Try: PG_PORT=5433 make pg-up"; echo ""; exit 1)

.PHONY: pg-down
pg-down: ## Stop Postgres container
	docker compose -f docker-compose.postgres.yml down -v

.PHONY: load
load: ## Run a Locust load test against the local API (interactive UI)
	$(UV) run locust -f load/locustfile.py --host http://localhost:8000

.PHONY: load-real
load-real: ## Run headless load test (memory store, no Redis needed) and update docs/load-test-results.md
	bash scripts/run_load_test.sh --users 10 --duration 30s

.PHONY: eval
eval: ## Run the golden eval set and regenerate evals/REPORT.md (latency cells fixed → no git churn)
	$(UV) run python -m evals.harness --deterministic

.PHONY: eval-timings
eval-timings: ## Like eval but with real wall-clock latency numbers (output may differ each run)
	$(UV) run python -m evals.harness

.PHONY: chaos
chaos: ## Run chaos harness and regenerate docs/CHAOS.md (deterministic — byte-stable across runs)
	$(UV) run python -m research_crew.chaos --scenarios all --runs 20 --deterministic --out docs/CHAOS.md

.PHONY: inngest-dev
inngest-dev: ## Start the Inngest dev server (requires npx). The research-crew API will then route through Inngest.
	@if ! command -v npx > /dev/null 2>&1; then \
		echo ""; \
		echo "  npx is not on PATH. Install Node.js (https://nodejs.org) and re-run."; \
		echo ""; \
		echo "  Once npx is available:"; \
		echo "    make inngest-dev"; \
		echo ""; \
		echo "  Then in another terminal:"; \
		echo "    export INNGEST_DEV_SERVER_URL=http://localhost:8288"; \
		echo "    uv run research --use-inngest \"your question\""; \
		echo ""; \
		exit 0; \
	fi
	@echo ""
	@echo "  Starting Inngest dev server at http://localhost:8288"
	@echo "  In another terminal, run:"
	@echo "    export INNGEST_DEV_SERVER_URL=http://localhost:8288"
	@echo "    uv run research --use-inngest \"your question\""
	@echo ""
	npx inngest-cli@latest dev

.PHONY: docker-build
docker-build: ## Build the production Docker image (tag: research-crew:latest)
	docker build --tag research-crew:latest .

.PHONY: docker-run
docker-run: ## Run the production Docker image locally (memory store, port 8000)
	docker run --rm \
		--name research-crew-local \
		--publish 8000:8000 \
		--env RESEARCH_CREW_STORE=memory \
		--env RESEARCH_DEV_MODE=1 \
		research-crew:latest

.PHONY: deploy
deploy: ## Deploy to Fly.io via scripts/deploy.sh (requires fly CLI + auth)
	bash scripts/deploy.sh

.PHONY: openapi
openapi: ## Regenerate docs/openapi.json + docs/postman_collection.json from the live app schema
	$(UV) run python scripts/export_openapi.py

.PHONY: clean
clean: ## Wipe caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist
	find . -name __pycache__ -type d -exec rm -rf {} +
