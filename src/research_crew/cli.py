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

from research_crew.agents import BraveAgent, ExaAgent, MockAgent, TavilyAgent, default_agents
from research_crew.agents.base import Agent
from research_crew.models import AgentName, RunStatus, StepStatus
from research_crew.store import InMemoryRunStore, RedisRunStore, RunStore
from research_crew.synthesizer import StitchSynthesizer
from research_crew.workflow import WorkflowEngine
from research_crew.workflow_inngest import make_workflow

app = typer.Typer(
    name="research", help="research-crew CLI", add_completion=False, no_args_is_help=True
)
console = Console()

# Valid agent selector tokens (case-insensitive).
# "mockN" (mock1, mock2, …) resolves to a MockAgent with the given slot name.
_REAL_ADAPTERS: dict[str, type[TavilyAgent] | type[BraveAgent] | type[ExaAgent]] = {
    "tavily": TavilyAgent,
    "brave": BraveAgent,
    "exa": ExaAgent,
}

# Fallback slot names for mock agents when named explicitly via --agents.
# They resolve to AgentName enum members in order.
_MOCK_SLOT_NAMES: list[AgentName] = list(AgentName)


def _build_agents(agents_flag: str | None, failure_rate: float) -> list[Agent]:
    """Resolve ``--agents`` CSV string to a concrete agent list.

    Tokens (case-insensitive):
    - ``tavily`` → ``TavilyAgent`` (real when TAVILY_API_KEY set, mock otherwise)
    - ``brave``  → ``BraveAgent``  (real when BRAVE_API_KEY set, mock otherwise)
    - ``exa``    → ``ExaAgent``    (real when EXA_API_KEY set, mock otherwise)
    - ``mock1``…``mock5`` → ``MockAgent`` with the corresponding ``AgentName``
    - absent / ``None`` → the default five-MockAgent crew

    Duplicate tokens are permitted (they resolve to distinct instances).
    Unknown tokens raise ``typer.BadParameter``.
    """
    if agents_flag is None:
        return default_agents(failure_rate=failure_rate)

    tokens = [t.strip().lower() for t in agents_flag.split(",") if t.strip()]
    if not tokens:
        return default_agents(failure_rate=failure_rate)

    result: list[Agent] = []
    mock_slot_idx = 0
    for token in tokens:
        if token in _REAL_ADAPTERS:
            result.append(_REAL_ADAPTERS[token]())
        elif token.startswith("mock"):
            # mock1, mock2, … or just "mock" — each consumes a slot name in order.
            slot_name = _MOCK_SLOT_NAMES[mock_slot_idx % len(_MOCK_SLOT_NAMES)]
            mock_slot_idx += 1
            result.append(MockAgent(name=slot_name, failure_rate=failure_rate))
        else:
            raise typer.BadParameter(
                f"Unknown agent token '{token}'. "
                "Valid choices: tavily, brave, exa, mock1, mock2, mock3, mock4, mock5"
            )
    return result


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
    agents_flag: str | None = typer.Option(
        None,
        "--agents",
        help=(
            "Comma-separated list of agent tokens: tavily, brave, exa, mock1..mock5. "
            "Real adapters activate when the corresponding API key env var is set; "
            "fall back to mock otherwise. Default: all-mock five-agent crew."
        ),
    ),
    use_inngest: bool = typer.Option(
        False,
        "--use-inngest",
        help=(
            "Route the workflow through the Inngest dev server. "
            "Requires the 'inngest' package (pip install 'research-crew[real]') "
            "and the dev server running at INNGEST_DEV_SERVER_URL "
            "(default: http://localhost:8288 — start with: npx inngest-cli@latest dev)."
        ),
    ),
) -> None:
    """Run one research job and print the report."""

    async def _go() -> None:
        store = _store_factory(use_redis)
        run_id = uuid.uuid4().hex
        run_status = RunStatus(run_id=run_id, question=question, state=StepStatus.RUNNING)
        await store.put_run(run_status)
        agents = _build_agents(agents_flag, failure_rate)
        if use_inngest:
            try:
                engine = make_workflow(
                    run_id=run_id,
                    use_inngest=True,
                    record_step=store.append_step,
                )
            except ImportError as exc:
                raise typer.BadParameter(str(exc), param_hint="--use-inngest") from exc
        else:
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


@app.command()
def watch(
    question: str = typer.Argument(..., help="Research question to stream live"),
    api_url: str = typer.Option(
        "http://localhost:8000",
        "--api-url",
        help="Base URL of the research-crew API.",
    ),
    token: str | None = typer.Option(
        None,
        "--token",
        envvar="RESEARCH_API_TOKEN",
        help="Bearer token for authentication.",
    ),
) -> None:
    """Submit a research run and stream step events live via WebSocket.

    Connects to the running research-crew API, submits the question via
    POST /research, then opens a WebSocket to WS /runs/{run_id}/stream
    and prints each step event as it arrives.  Exits when the run
    reaches a terminal state.

    Requires a research-crew API to be running (``make api``).
    """
    import httpx  # noqa: PLC0415 — deferred import keeps startup fast

    async def _go() -> None:
        headers: dict[str, str] = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        async with httpx.AsyncClient(base_url=api_url, headers=headers) as http:
            resp = await http.post("/research", json={"question": question})
            if resp.status_code not in (200, 202):
                console.print(f"[red]submit failed[/] HTTP {resp.status_code}: {resp.text}")
                raise typer.Exit(1)
            run_id: str = resp.json()["run_id"]
            console.print(f"[bold]run_id:[/] {run_id}")
            console.print("[dim]streaming step events …[/]")

        # Connect to the WebSocket endpoint and stream events.
        from research_crew.client import stream_run  # noqa: PLC0415

        step_count = 0
        async for msg in stream_run(api_url=api_url, run_id=run_id, token=token):
            msg_type = msg.get("type")
            if msg_type == "snapshot":
                state = msg.get("state", "unknown")
                console.print(f"[cyan]snapshot[/]  state={state}")
            elif msg_type == "step":
                step_count += 1
                agent = msg.get("agent", "?")
                status = msg.get("status", "?")
                error = msg.get("error") or ""
                line = f"[green]step[/]      agent={agent}  status={status}"
                if error:
                    line += f"  error={error}"
                console.print(line)
            elif msg_type == "heartbeat":
                console.print("[dim]♥ heartbeat[/]")
            elif msg_type == "done":
                console.print(f"[bold green]done[/]  {step_count} step(s) received")
                break

    asyncio.run(_go())
