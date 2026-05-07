"""PgVector-backed semantic deduplication.

Schema
------
::

    CREATE TABLE IF NOT EXISTS seen_content (
        id         SERIAL PRIMARY KEY,
        run_id     TEXT        NOT NULL,
        text_hash  TEXT        NOT NULL,
        embedding  vector(384),
        created_at TIMESTAMPTZ DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS seen_content_embedding_idx
        ON seen_content USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);

Activation
----------
Constructed by :func:`~research_crew.dedup.make_semantic_dedup` when
``RESEARCH_PG_DSN`` is set.  Never imported by code that doesn't set
that variable, so ``asyncpg`` stays a soft dependency.

Embedding strategy
------------------
1. Attempt ``sentence_transformers`` (``all-MiniLM-L6-v2``, 384 dims).
2. If the SDK is absent, fall back to a deterministic hash-based fake
   encoder (same pattern as the codex-atlas FakeEncoder): each dimension
   is a signed value derived from the SHA-256 digest of the text, then
   normalised to unit length.  This is NOT semantically meaningful but
   produces stable, unique vectors and lets the test suite run without
   downloading any models.
"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    pass  # asyncpg type stubs only

_log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL_SEEN_CONTENT = """
CREATE TABLE IF NOT EXISTS seen_content (
    id         SERIAL PRIMARY KEY,
    run_id     TEXT        NOT NULL,
    text_hash  TEXT        NOT NULL,
    embedding  vector(384),
    created_at TIMESTAMPTZ DEFAULT now()
);
"""

_DDL_SEEN_CONTENT_IDX = """
CREATE INDEX IF NOT EXISTS seen_content_embedding_idx
    ON seen_content USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

_EMBEDDING_DIM = 384


def _fake_embed(text: str) -> list[float]:
    """Deterministic hash-based embedding (no ML model required).

    Produces a unit-length vector of 384 floats derived from the SHA-256
    digest of *text*.  Not semantically meaningful — suitable only for
    testing the plumbing without downloading models.
    """
    digest = hashlib.sha256(text.encode()).digest()
    # Repeat the 32-byte digest until we have ≥ 384 values.
    repeated = (digest * ((_EMBEDDING_DIM // len(digest)) + 1))[:_EMBEDDING_DIM]
    # Map bytes to signed floats in [-1, 1].
    raw = [(b - 128) / 128.0 for b in repeated]
    # Normalise to unit length.
    norm = math.sqrt(sum(v * v for v in raw)) or 1.0
    return [v / norm for v in raw]


def _load_sentence_transformer() -> Any:
    """Try to import sentence-transformers; return the model or None."""
    try:
        from sentence_transformers import (
            SentenceTransformer,  # type: ignore[import-untyped,unused-ignore]
        )

        model = SentenceTransformer("all-MiniLM-L6-v2")
        _log.debug("semantic_dedup.encoder", backend="sentence_transformers")
        return model
    except ImportError:
        _log.debug(
            "semantic_dedup.encoder",
            backend="fake_hash",
            hint="pip install sentence-transformers to enable real embeddings",
        )
        return None


def _require_asyncpg() -> Any:
    try:
        import asyncpg

        return asyncpg
    except ImportError as exc:
        raise ImportError(
            "asyncpg is required for PgVectorSemanticDedup. Install it with: pip install asyncpg"
        ) from exc


# ---------------------------------------------------------------------------
# PgVectorSemanticDedup
# ---------------------------------------------------------------------------


class PgVectorSemanticDedup:
    """Semantic deduplication backed by pgvector ANN search.

    Accepts either an already-open *pool* (for tests) or a *dsn* string
    (for production).  When a DSN is provided the pool is created lazily
    on the first call to :meth:`setup`.

    Call ``await dedup.setup()`` once at application startup to apply
    the DDL idempotently.
    """

    def __init__(self, pool: Any = None, *, dsn: str | None = None) -> None:
        if pool is None and dsn is None:
            raise ValueError("PgVectorSemanticDedup requires either a pool or a dsn")
        self._pool = pool
        self._dsn = dsn
        self._model: Any = _load_sentence_transformer()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Apply the schema idempotently.

        Creates the pool from :attr:`_dsn` if not already open.
        """
        if self._pool is None and self._dsn is not None:
            asyncpg = _require_asyncpg()
            self._pool = await asyncpg.create_pool(self._dsn)
        async with self._pool.acquire() as conn:
            # Enable pgvector extension (no-op if already enabled).
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            except Exception as exc:
                _log.warning(
                    "semantic_dedup.extension_warning",
                    error=str(exc),
                    hint="pgvector extension may not be installed",
                )
            await conn.execute(_DDL_SEEN_CONTENT)
            await conn.execute(_DDL_SEEN_CONTENT_IDX)
        _log.info("semantic_dedup.setup_complete")

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        if self._model is not None:
            embedding: list[float] = self._model.encode(text).tolist()
            return embedding
        return _fake_embed(text)

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    async def is_duplicate(
        self,
        text: str,
        threshold: float = 0.85,
    ) -> tuple[bool, str | None]:
        """Return ``(True, run_id)`` when similar content was seen before."""
        vec = self._embed(text)
        vec_str = "[" + ",".join(f"{v:.8f}" for v in vec) + "]"
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT run_id, 1 - (embedding <=> $1::vector) AS similarity
                FROM seen_content
                ORDER BY embedding <=> $1::vector
                LIMIT 1
                """,
                vec_str,
            )
        if row is None:
            return (False, None)
        similarity: float = float(row["similarity"])
        if similarity >= threshold:
            _log.debug(
                "semantic_dedup.duplicate_found",
                similarity=round(similarity, 4),
                threshold=threshold,
                prior_run_id=row["run_id"],
            )
            return (True, row["run_id"])
        return (False, None)

    async def add_seen(self, text: str, run_id: str) -> None:
        """Embed *text* and insert a row into ``seen_content``."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        vec = self._embed(text)
        vec_str = "[" + ",".join(f"{v:.8f}" for v in vec) + "]"
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO seen_content (run_id, text_hash, embedding)
                VALUES ($1, $2, $3::vector)
                """,
                run_id,
                text_hash,
                vec_str,
            )
        _log.debug("semantic_dedup.add_seen", run_id=run_id, text_hash=text_hash[:12])


__all__ = ["PgVectorSemanticDedup"]
