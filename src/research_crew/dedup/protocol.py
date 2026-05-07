"""Protocol definition for semantic deduplication.

``SemanticDedup`` is an intentional Protocol so callers depend only on
the interface.  The two concrete implementations are:

* :class:`~research_crew.dedup.null.NullDedup`  — no-op default (zero deps)
* :class:`~research_crew.dedup.pgvector.PgVectorSemanticDedup` — ANN search
  via pgvector (requires asyncpg + a live Postgres instance).
"""

from __future__ import annotations

from typing import Protocol


class SemanticDedup(Protocol):
    """Protocol satisfied by both NullDedup and PgVectorSemanticDedup.

    All methods are coroutines so callers ``await`` them uniformly;
    NullDedup's implementations return immediately.
    """

    async def is_duplicate(
        self,
        text: str,
        threshold: float = 0.85,
    ) -> tuple[bool, str | None]:
        """Return ``(True, similar_run_id)`` when *text* is semantically
        similar to a previously seen chunk, else ``(False, None)``.

        Parameters
        ----------
        text:
            The content to test — typically a citation snippet or summary.
        threshold:
            Cosine-similarity threshold in [0, 1].  Content whose ANN
            top-1 similarity is ≥ this value is considered a duplicate.
        """
        ...

    async def add_seen(self, text: str, run_id: str) -> None:
        """Record *text* as belonging to *run_id* for future dedup checks.

        Safe to call multiple times with the same text + run_id — the
        implementation may insert a new row each time or deduplicate
        internally; callers must not rely on either behaviour.
        """
        ...


__all__ = ["SemanticDedup"]
