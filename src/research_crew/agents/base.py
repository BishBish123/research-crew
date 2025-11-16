"""`Agent` protocol + a deterministic `MockAgent` for offline use.

Every agent has a single async `search(question)` method. Real adapters
(Tavily, Brave, Semantic Scholar, GitHub Code Search, Wikipedia) drop in
by implementing the same protocol.

The mock implementation is intentional: real search APIs are paid /
rate-limited / hard to make deterministic. The portfolio version's
*architecture* is what matters — fan-out, retries, idempotency,
deduplication. Real-search adapters live behind the same Protocol so
swapping them is one line.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from research_crew.models import AgentName, AgentResult, Citation, StepStatus


@runtime_checkable
class Agent(Protocol):
    name: AgentName

    async def search(self, question: str) -> AgentResult: ...


@dataclass
class MockAgent:
    """Deterministic stub that returns 1-3 fake citations per question.

    Citations are seeded from a blake2b digest so the same question
    always produces the same results — useful for the eval harness.
    Optional `latency_ms` simulates real-search jitter; `failure_rate`
    drives the retry path so the workflow's fault-tolerance is testable.
    """

    name: AgentName
    latency_ms: float = 50.0
    failure_rate: float = 0.0
    _attempt_counter: int = 0

    async def search(self, question: str) -> AgentResult:
        await asyncio.sleep(self.latency_ms / 1000.0)
        self._attempt_counter += 1
        # Deterministic, but failure_rate decides the binary outcome.
        if self.failure_rate > 0:
            digest = hashlib.blake2b(
                f"{self.name}|{question}|{self._attempt_counter}".encode(),
                digest_size=4,
            ).digest()
            roll = int.from_bytes(digest, "big") / 2**32
            if roll < self.failure_rate:
                return AgentResult(
                    agent=self.name,
                    status=StepStatus.FAILED,
                    summary="",
                    error=f"simulated {self.name} failure (roll={roll:.3f})",
                    elapsed_ms=self.latency_ms,
                )
        return AgentResult(
            agent=self.name,
            status=StepStatus.SUCCEEDED,
            summary=_mock_summary(self.name, question),
            citations=_mock_citations(self.name, question),
            elapsed_ms=self.latency_ms,
            attempts=1,
        )


def _mock_summary(name: AgentName, question: str) -> str:
    return f"[{name}] summary for: {question}"


def _mock_citations(name: AgentName, question: str) -> list[Citation]:
    base = hashlib.blake2b(f"{name}|{question}".encode(), digest_size=4).hexdigest()
    return [
        Citation(
            title=f"{name} result {i + 1}: {question[:40]}",
            url=f"https://example.com/{name}/{base}/{i}",
            snippet=f"Mock snippet from {name} ({base}-{i}).",
        )
        for i in range(2)
    ]


def default_agents(*, latency_ms: float = 50.0, failure_rate: float = 0.0) -> list[Agent]:
    """One MockAgent per AgentName, configured uniformly."""
    return [MockAgent(name=n, latency_ms=latency_ms, failure_rate=failure_rate) for n in AgentName]
