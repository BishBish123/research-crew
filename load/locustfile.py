"""Locust load test against the local API.

Usage:
    make up                                  # bring Redis up
    uv run research-api                      # start the API in another shell
    make load                                # run Locust against http://localhost:8000

The agent layer is mocked (deterministic) so this measures the *workflow
plumbing*, not any external search service. That's what you want when
the question is "can the orchestration take 100 concurrent runs without
falling over".
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
]


class ResearchUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(3)
    def submit_research(self) -> None:
        question = random.choice(QUESTIONS)  # noqa: S311 - load test, not crypto
        with self.client.post(
            "/research", json={"question": question}, catch_response=True
        ) as resp:
            if resp.status_code != 202:
                resp.failure(f"unexpected status {resp.status_code}")
                return
            run_id = resp.json().get("run_id")
            if not run_id:
                resp.failure("missing run_id")
                return
            self.client.get(f"/runs/{run_id}", name="/runs/{id}")

    @task(1)
    def health(self) -> None:
        self.client.get("/health")
