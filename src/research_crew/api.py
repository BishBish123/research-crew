"""FastAPI service: POST /research, GET /runs/{id}, health probe."""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import UTC, datetime

import redis.asyncio as aioredis
import typer
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from research_crew.agents import default_agents
from research_crew.models import (
    ResearchReport,
    ResearchRequest,
    RunStatus,
    StepStatus,
)
from research_crew.store import RedisRunStore, RunStore
from research_crew.synthesizer import StitchSynthesizer
from research_crew.workflow import WorkflowEngine

app = FastAPI(title="research-crew", version="0.1.0")


def _redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://localhost:6379/0")


@app.on_event("startup")
async def _startup() -> None:
    client = aioredis.from_url(_redis_url(), decode_responses=True)
    app.state.redis = client
    app.state.store = RedisRunStore(client)


@app.on_event("shutdown")
async def _shutdown() -> None:
    await app.state.redis.close()


@app.get("/health")
async def health() -> dict[str, str]:
    pong = await app.state.redis.ping()
    return {"status": "ok", "redis": "up" if pong else "down"}


@app.post("/research", status_code=202)
async def submit_research(
    payload: ResearchRequest, background: BackgroundTasks, request: Request
) -> dict[str, str]:
    """Enqueue a research run. Returns immediately with the run_id."""
    run_id = uuid.uuid4().hex
    store: RunStore = request.app.state.store
    run = RunStatus(run_id=run_id, question=payload.question, state=StepStatus.RUNNING)
    await store.put_run(run)
    background.add_task(_execute_run, store, run_id, payload)
    return {"run_id": run_id, "status_url": f"/runs/{run_id}"}


@app.get("/runs/{run_id}")
async def get_run(run_id: str, request: Request) -> RunStatus:
    store: RunStore = request.app.state.store
    run = await store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    run.steps = await store.list_steps(run_id)
    return run


# ---------------------------------------------------------------------------
# Background execution
# ---------------------------------------------------------------------------


async def _execute_run(store: RunStore, run_id: str, payload: ResearchRequest) -> None:
    agents = default_agents()
    if payload.agents:
        wanted = set(payload.agents)
        agents = [a for a in agents if a.name in wanted]
    engine = WorkflowEngine(
        run_id=run_id,
        record_step=store.append_step,
        cache_get=store.cache_get,
        cache_put=store.cache_put,
    )
    results = await engine.run_parallel(agents, payload.question)
    report: ResearchReport = await StitchSynthesizer().synthesize(run_id, payload.question, results)
    finished = await store.get_run(run_id)
    if finished is None:
        return
    finished.state = (
        StepStatus.FAILED
        if all(r.status is StepStatus.FAILED for r in results)
        else StepStatus.SUCCEEDED
    )
    finished.finished_at = datetime.now(UTC)
    finished.report = report
    await store.put_run(finished)


# ---------------------------------------------------------------------------
# CLI entry: `research-api`
# ---------------------------------------------------------------------------


_cli = typer.Typer(name="research-api", add_completion=False)


@_cli.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="uvicorn auto-reload"),
) -> None:
    """Run the FastAPI service."""
    uvicorn.run(
        "research_crew.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


def main() -> None:  # pragma: no cover
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    _cli()


if __name__ == "__main__":  # pragma: no cover
    main()
