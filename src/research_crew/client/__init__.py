"""Async WebSocket client helpers for the research-crew streaming API.

Usage
-----
::

    from research_crew.client import stream_run

    async for event in stream_run("http://localhost:8000", run_id, token):
        if event["type"] == "step":
            print(event["agent"], event["status"])
        elif event["type"] == "done":
            break

The ``stream_run`` async generator connects to
``WS /runs/{run_id}/stream``, yields each JSON-decoded message dict as
it arrives, and closes the connection cleanly on ``done`` or any server-
initiated close.

``WebSocketClient`` is the lower-level class backing ``stream_run``.
Instantiate it directly if you need finer-grained control over the
connection lifetime (e.g. reconnect logic).

Note: requires ``websockets>=13`` (already a transitive dep of FastAPI's
WebSocket support).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse, urlunparse

import websockets.asyncio.client as ws_client

from research_crew.models import StepRecord


class WebSocketClient:
    """Thin async wrapper around ``websockets`` for the streaming endpoint.

    Parameters
    ----------
    api_url:
        HTTP(S) base URL of the research-crew API, e.g.
        ``"http://localhost:8000"``. The scheme is automatically
        converted to ``ws://`` / ``wss://``.
    token:
        Bearer token for authentication.  Sent as the ``?token=``
        query parameter so this works from browsers too.
    """

    def __init__(self, api_url: str, token: str | None = None) -> None:
        parsed = urlparse(api_url)
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        self._base = urlunparse(parsed._replace(scheme=ws_scheme))
        self._token = token

    def _ws_url(self, run_id: str) -> str:
        url = f"{self._base}/runs/{run_id}/stream"
        if self._token:
            url += f"?token={self._token}"
        return url

    async def stream(self, run_id: str) -> AsyncIterator[dict[str, Any]]:
        """Yield JSON message dicts from WS /runs/{run_id}/stream.

        Closes the connection and stops iteration on a ``"done"`` message
        or any server-initiated close.
        """
        url = self._ws_url(run_id)
        async with ws_client.connect(url) as conn:
            async for raw in conn:
                msg: dict[str, Any] = json.loads(raw)
                yield msg
                if msg.get("type") == "done":
                    break


async def stream_run(
    api_url: str,
    run_id: str,
    token: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Async generator that yields step-event dicts for ``run_id``.

    Connects to ``WS /runs/{run_id}/stream`` and yields each message dict
    (snapshot, step, heartbeat, done) as it arrives.

    Parameters
    ----------
    api_url:
        HTTP(S) base URL of the research-crew API.
    run_id:
        The run ID to stream events for.
    token:
        Bearer token; passed as ``?token=`` query parameter.

    Yields
    ------
    dict
        JSON-decoded message. Keys include ``"type"`` (``"snapshot"``,
        ``"step"``, ``"heartbeat"``, ``"done"``) and the fields of the
        relevant Pydantic model.
    """
    client = WebSocketClient(api_url=api_url, token=token)
    async for msg in client.stream(run_id):
        yield msg


__all__ = ["StepRecord", "WebSocketClient", "stream_run"]
