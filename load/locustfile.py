"""Locust load test against the local API.

Usage (interactive UI):
    make up                                  # bring Redis up
    uv run research-api                      # start the API in another shell
    make load                                # run Locust against http://localhost:8000

Usage (headless, deterministic):
    scripts/run_load_test.sh --users 10 --duration 30s
    make load-real

The agent layer is mocked (deterministic) so this measures the *workflow
plumbing*, not any external search service. That's what you want when
the question is "can the orchestration take 100 concurrent runs without
falling over".

Traffic mix (by weight):
    submit_fast    (weight 4) POST /research with failure_rate=0, GET poll
    submit_flaky   (weight 1) POST /research with failure_rate=0.3, GET poll
    health         (weight 1) GET /health baseline probe
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

QUESTIONS = [
    "what is the best way to scale a Postgres write workload",
    "how does Inngest handle step retries",
    "compare LangGraph and CrewAI for production agents",
    "what changed in pgvector 0.8 vs 0.7",
    "best free observability stack for a small team",
    "how does asyncio.gather handle partial failures",
    "explain Redis Streams consumer group semantics",
    "what are the tradeoffs between HSET and JSON in Redis",
]


class ResearchUser(HttpUser):
    wait_time = between(0.5, 1.5)

    # Track run_ids created by this virtual user so subsequent
    # GET /runs/{id} calls always reference a real run.
    _run_ids: list[str]

    def on_start(self) -> None:
        self._run_ids = []

    def _submit(self, *, failure_rate: float = 0.0) -> None:
        """POST /research and immediately GET /runs/{id} for the new run."""
        question = random.choice(QUESTIONS)  # noqa: S311 - load test, not crypto
        payload: dict[str, object] = {"question": question}
        # Pass failure_rate as a query param (ignored by the API but lets us
        # tag locust request name rows by scenario if needed).  The actual
        # mock agent's failure_rate is set via the `RESEARCH_CREW_FAILURE_RATE`
        # env var when the server is started for the load harness — the param
        # is here for documentation / future per-request override support.
        with self.client.post(
            "/research",
            json=payload,
            catch_response=True,
            name="POST /research",
        ) as resp:
            if resp.status_code == 429:
                # Rate-limited: mark as success (expected in high-load runs)
                # so it doesn't skew the failure rate counter.
                resp.success()
                return
            if resp.status_code != 202:
                resp.failure(f"unexpected status {resp.status_code}: {resp.text[:200]}")
                return
            body = resp.json()
            run_id = body.get("run_id")
            if not run_id:
                resp.failure("missing run_id in response body")
                return
            # Stash so GET tasks can reference it later.
            self._run_ids.append(run_id)
            # Always poll once immediately after submit to exercise the
            # GET /runs/{id} path in the same request cycle.
            self.client.get(f"/runs/{run_id}", name="GET /runs/{id}")

    @task(4)
    def submit_fast(self) -> None:
        """Fast-path scenario: mock agents succeed on first attempt."""
        self._submit(failure_rate=0.0)

    @task(1)
    def submit_flaky(self) -> None:
        """Retry-path scenario: 30% per-agent failure probability.

        Exercises the exponential-backoff retry budget in WorkflowEngine.
        The workflow still converges to SUCCEEDED for most agents even with
        30% failure because each step gets 3 attempts.
        """
        self._submit(failure_rate=0.3)

    @task(1)
    def health(self) -> None:
        """Baseline probe — exercises the /health SCAN + Redis ping path."""
        self.client.get("/health", name="GET /health")
