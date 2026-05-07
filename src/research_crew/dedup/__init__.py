"""Semantic deduplication package.

Cross-run content deduplication via pgvector ANN search.  The default
path is :class:`NullDedup` (no-op, zero dependencies).  Set
``RESEARCH_PG_DSN`` to activate :class:`PgVectorSemanticDedup`.

Quick-start
-----------
::

    from research_crew.dedup import make_semantic_dedup

    dedup = make_semantic_dedup()  # NullDedup unless RESEARCH_PG_DSN set
    await dedup.setup()  # no-op for NullDedup
    is_dup, prior = await dedup.is_duplicate(text)
    if not is_dup:
        await dedup.add_seen(text, run_id)

"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from research_crew.dedup.null import NullDedup
from research_crew.dedup.protocol import SemanticDedup

if TYPE_CHECKING:
    from research_crew.dedup.pgvector import PgVectorSemanticDedup

__all__ = [
    "NullDedup",
    "SemanticDedup",
    "make_semantic_dedup",
]


def make_semantic_dedup() -> NullDedup | PgVectorSemanticDedup:
    """Factory: return a :class:`PgVectorSemanticDedup` when
    ``RESEARCH_PG_DSN`` is set, otherwise a :class:`NullDedup`.

    The returned object satisfies the :class:`SemanticDedup` Protocol in
    both cases so callers need no conditional logic.

    Usage::

        dedup = make_semantic_dedup()
        await dedup.setup()
        is_dup, run_id = await dedup.is_duplicate(text)
        if not is_dup:
            await dedup.add_seen(text, current_run_id)
    """
    dsn = os.environ.get("RESEARCH_PG_DSN", "")
    if dsn:
        from research_crew.dedup.pgvector import PgVectorSemanticDedup

        return PgVectorSemanticDedup(dsn=dsn)
    return NullDedup()
