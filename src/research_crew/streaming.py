"""Per-run asyncio.Queue registry for WebSocket step-event fan-out.

Each active run can have zero or more WebSocket subscribers. When a step
event fires (via the wrapped ``record_step`` callback wired in the API),
the event is broadcast to every subscriber's queue.

Lifecycle
---------
1. ``RunQueueRegistry.create(run_id)`` — called at POST /research time.
   Returns the shared ``asyncio.Queue`` for the run. Each WS subscriber
   gets its OWN queue via ``subscribe``; ``create`` just initialises the
   per-run slot in the registry.
2. ``RunQueueRegistry.subscribe(run_id)`` — called per WS connection.
   Returns a fresh ``asyncio.Queue`` that will receive every subsequent
   ``publish`` call.
3. ``RunQueueRegistry.publish(run_id, event)`` — called by the wrapped
   ``record_step`` callback. Puts ``event`` onto every subscriber queue.
   Silently no-ops if the run has no subscribers or has already been
   torn down.
4. ``RunQueueRegistry.teardown(run_id)`` — called when the run reaches
   terminal state. Sends a ``None`` sentinel to every subscriber queue
   so WS handlers know to close the connection.

Concurrency
-----------
All mutations happen on the FastAPI event loop (single-threaded asyncio)
so dict mutations are not interleaved. No explicit lock is needed.

Future multi-instance support
-----------------------------
This implementation is intentionally in-process: a single API instance
owns the queues and the WS connections. For horizontal scaling, replace
``publish`` with a Redis pub/sub PUBLISH call and have subscribers run
``SUBSCRIBE`` on a per-run channel. The WebSocket handler loop is
unchanged — swap the queue.get() for an asyncio iterator over the Redis
channel messages.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress

import structlog

from research_crew.models import StepRecord

_log = structlog.get_logger(__name__)

# Sentinel pushed onto subscriber queues to signal run termination.
_TERMINAL: None = None


class RunQueueRegistry:
    """In-process broadcast registry: one run → many subscriber queues."""

    def __init__(self) -> None:
        # Maps run_id → list of subscriber queues.
        self._subscribers: dict[str, list[asyncio.Queue[StepRecord | None]]] = {}
        # Set of run_ids that have reached a terminal state; used to give
        # late-joining WS clients an immediate close signal.
        self._terminal: set[str] = set()

    def create(self, run_id: str) -> None:
        """Initialise the subscriber slot for a new run.

        Safe to call even if the slot already exists (idempotent).
        """
        if run_id not in self._subscribers:
            self._subscribers[run_id] = []

    def subscribe(self, run_id: str) -> asyncio.Queue[StepRecord | None]:
        """Register a new subscriber for ``run_id``.

        Returns a fresh queue that will receive subsequent step events.
        If the run is already terminal the queue receives ``None``
        immediately so the WS handler exits without blocking.
        """
        q: asyncio.Queue[StepRecord | None] = asyncio.Queue()
        if run_id in self._terminal:
            q.put_nowait(_TERMINAL)
            return q
        self._subscribers.setdefault(run_id, []).append(q)
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue[StepRecord | None]) -> None:
        """Remove ``q`` from the subscriber list for ``run_id``."""
        subs = self._subscribers.get(run_id)
        if subs is not None:
            with suppress(ValueError):
                subs.remove(q)

    def publish(self, run_id: str, step: StepRecord) -> None:
        """Broadcast ``step`` to all current subscribers of ``run_id``."""
        for q in list(self._subscribers.get(run_id, [])):
            q.put_nowait(step)

    def teardown(self, run_id: str) -> None:
        """Signal all subscribers that the run is done and clean up.

        Sends the ``None`` sentinel to every subscriber queue, then
        removes the run from the registry. Marks the run as terminal so
        late-joining subscribers immediately receive the sentinel.
        """
        self._terminal.add(run_id)
        for q in list(self._subscribers.pop(run_id, [])):
            q.put_nowait(_TERMINAL)
        _log.debug("streaming.teardown", run_id=run_id)


# Module-level singleton — the API module imports this instance.
registry: RunQueueRegistry = RunQueueRegistry()
