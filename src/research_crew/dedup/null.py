"""NullDedup — no-op semantic deduplication.

Used as the default when ``RESEARCH_PG_DSN`` is not set.  Every call
returns immediately without side-effects so the synthesizer continues
to work exactly as before.
"""

from __future__ import annotations


class NullDedup:
    """No-op deduplicator.

    ``is_duplicate`` always returns ``(False, None)`` so every citation
    passes through and the synthesizer's URL-only dedup is the sole guard.

    ``add_seen`` is a no-op — nothing is stored, nothing is allocated.
    """

    async def is_duplicate(
        self,
        text: str,
        threshold: float = 0.85,
    ) -> tuple[bool, str | None]:
        """Always reports "not a duplicate"."""
        return (False, None)

    async def add_seen(self, text: str, run_id: str) -> None:
        """No-op — NullDedup never accumulates state."""


__all__ = ["NullDedup"]
