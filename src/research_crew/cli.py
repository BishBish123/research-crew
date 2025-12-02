"""`research` CLI: run a single research job locally without booting the API."""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import UTC, datetime

import redis.asyncio as aioredis
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from research_crew.agents import default_agents
from research_crew.models import RunStatus, StepStatus
from research_crew.store import InMemoryRunStore, RedisRunStore, RunStore
from research_crew.synthesizer import StitchSynthesizer
from research_crew.workflow import WorkflowEngine

app = typer.Typer(
    name="research", help="research-crew CLI", add_completion=False, no_args_is_help=True
)
console = Console()


def _store_factory(use_redis: bool) -> RunStore:
    if not use_redis:
        return InMemoryRunStore()
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    return RedisRunStore(aioredis.from_url(url, decode_responses=True))


@app.command()
def run(
    question: str = typer.Argument(..., help="Research question"),
    use_redis: bool = typer.Option(False, "--redis", help="Persist to Redis instead of memory"),
    failure_rate: float = typer.Option(
        0.0, help="Inject 0..1 failure rate per agent attempt to test the retry path"
    ),
) -> None:
    """Run one research job and print the report."""

    async def _go() -> None:
        store = _store_factory(use_redis)
        run_id = uuid.uuid4().hex
        run_status = RunStatus(run_id=run_id, question=question, state=StepStatus.RUNNING)
        await store.put_run(run_status)
        agents = default_agents(failure_rate=failure_rate)
        engine = WorkflowEngine(
            run_id=run_id,
            record_step=store.append_step,
            cache_get=store.cache_get,
            cache_put=store.cache_put,
        )
        results = await engine.run_parallel(agents, question)
        report = await StitchSynthesizer().synthesize(run_id, question, results)
        run_status.state = (
            StepStatus.FAILED
            if all(r.status is StepStatus.FAILED for r in results)
            else StepStatus.SUCCEEDED
        )
        run_status.finished_at = datetime.now(UTC)
        run_status.report = report
        await store.put_run(run_status)
        steps = await store.list_steps(run_id)

        # Render
        console.print(Markdown(report.summary))
        t = Table(title="Steps")
        t.add_column("Agent")
        t.add_column("Status")
        t.add_column("Attempts")
        t.add_column("Error")
        for s in steps:
            t.add_row(s.agent.value, s.status.value, str(s.attempts), s.error or "")
        console.print(t)
        console.print(
            f"[green]done[/] run_id={run_id}  "
            f"agents={len(results)}  citations={len(report.citations)}"
        )

    asyncio.run(_go())
