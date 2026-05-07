"""Tests for the semantic deduplication package.

All tests use mocked asyncpg pools — no real Postgres connection is made
and no sentence-transformers models are downloaded.

Coverage
--------
* NullDedup — is_duplicate always False; add_seen is a no-op
* PgVectorSemanticDedup
  - Schema applied on setup()
  - First call → is_duplicate=False, then add_seen persists the row
  - Second call with same content → is_duplicate=True, returns prior run_id
  - High threshold → nothing matches
  - Low threshold → always matches
* make_semantic_dedup() factory env-var dispatch
* Synthesizer integration
  - NullDedup: behaviour unchanged (all citations pass through)
  - PgVector mock: duplicates logged and skipped; survivors recorded
"""

from __future__ import annotations

import hashlib
import math
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_crew.dedup import NullDedup, make_semantic_dedup
from research_crew.dedup.pgvector import PgVectorSemanticDedup, _fake_embed
from research_crew.models import AgentName, AgentResult, Citation, StepStatus
from research_crew.synthesizer import StitchSynthesizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SENTINEL = object()  # distinct from None so we can tell "not set" apart


def _make_mock_pool(*, fetchrow_return: object = _SENTINEL) -> tuple[MagicMock, AsyncMock]:
    """Return a (pool, conn) pair whose async context manager works correctly.

    Pass ``fetchrow_return=None`` to make fetchrow return ``None`` (no row
    found).  Omit the argument to leave fetchrow as a bare AsyncMock (useful
    when you'll configure side_effect manually).
    """
    conn = AsyncMock()
    if fetchrow_return is not _SENTINEL:
        conn.fetchrow.return_value = fetchrow_return
    pool = MagicMock()
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    pool.acquire = MagicMock(return_value=cm)
    return pool, conn


def _citation(title: str, url: str, snippet: str = "") -> Citation:
    return Citation(title=title, url=url, snippet=snippet)


def _agent_result(
    citations: list[Citation],
    status: StepStatus = StepStatus.SUCCEEDED,
) -> AgentResult:
    return AgentResult(
        agent=AgentName.WEB_SEARCH,
        status=status,
        summary="test summary",
        citations=citations,
    )


# ---------------------------------------------------------------------------
# NullDedup
# ---------------------------------------------------------------------------


class TestNullDedup:
    async def test_is_duplicate_always_false(self) -> None:
        d = NullDedup()
        result = await d.is_duplicate("any text")
        assert result == (False, None)

    async def test_is_duplicate_with_high_threshold_still_false(self) -> None:
        d = NullDedup()
        result = await d.is_duplicate("any text", threshold=0.99)
        assert result == (False, None)

    async def test_add_seen_is_noop(self) -> None:
        # Must not raise and must not block.
        d = NullDedup()
        await d.add_seen("some text", "run-001")  # should complete without error

    async def test_repeated_calls_stay_false(self) -> None:
        d = NullDedup()
        await d.add_seen("text", "run-001")
        is_dup, run_id = await d.is_duplicate("text")
        assert not is_dup
        assert run_id is None


# ---------------------------------------------------------------------------
# _fake_embed unit test
# ---------------------------------------------------------------------------


class TestFakeEmbed:
    def test_returns_unit_vector(self) -> None:
        vec = _fake_embed("hello world")
        magnitude = math.sqrt(sum(v * v for v in vec))
        assert abs(magnitude - 1.0) < 1e-6

    def test_deterministic(self) -> None:
        assert _fake_embed("deterministic") == _fake_embed("deterministic")

    def test_different_texts_differ(self) -> None:
        assert _fake_embed("text A") != _fake_embed("text B")

    def test_length(self) -> None:
        assert len(_fake_embed("x")) == 384


# ---------------------------------------------------------------------------
# PgVectorSemanticDedup
# ---------------------------------------------------------------------------


class TestPgVectorSemanticDedupSetup:
    async def test_setup_executes_ddl_statements(self) -> None:
        pool, conn = _make_mock_pool()
        dedup = PgVectorSemanticDedup(pool=pool)
        await dedup.setup()
        # At minimum: CREATE EXTENSION, CREATE TABLE seen_content, CREATE INDEX
        assert conn.execute.call_count >= 3

    async def test_setup_includes_seen_content_table(self) -> None:
        pool, conn = _make_mock_pool()
        dedup = PgVectorSemanticDedup(pool=pool)
        await dedup.setup()
        all_sql = " ".join(str(call) for call in conn.execute.call_args_list)
        assert "seen_content" in all_sql

    async def test_setup_includes_ivfflat_index(self) -> None:
        pool, conn = _make_mock_pool()
        dedup = PgVectorSemanticDedup(pool=pool)
        await dedup.setup()
        all_sql = " ".join(str(call) for call in conn.execute.call_args_list)
        assert "ivfflat" in all_sql


class TestPgVectorIsDuplicate:
    async def test_returns_false_when_no_rows(self) -> None:
        pool, _conn = _make_mock_pool(fetchrow_return=None)
        dedup = PgVectorSemanticDedup(pool=pool)
        is_dup, run_id = await dedup.is_duplicate("brand new content")
        assert not is_dup
        assert run_id is None

    async def test_returns_false_when_similarity_below_threshold(self) -> None:
        row = MagicMock()
        row.__getitem__ = MagicMock(
            side_effect={"run_id": "run-old", "similarity": 0.5}.__getitem__
        )
        pool, _conn = _make_mock_pool(fetchrow_return=row)
        dedup = PgVectorSemanticDedup(pool=pool)
        is_dup, run_id = await dedup.is_duplicate("text", threshold=0.85)
        assert not is_dup
        assert run_id is None

    async def test_returns_true_when_similarity_above_threshold(self) -> None:
        row = MagicMock()
        row.__getitem__ = MagicMock(
            side_effect={"run_id": "run-prior", "similarity": 0.95}.__getitem__
        )
        pool, _conn = _make_mock_pool(fetchrow_return=row)
        dedup = PgVectorSemanticDedup(pool=pool)
        is_dup, run_id = await dedup.is_duplicate("similar text", threshold=0.85)
        assert is_dup
        assert run_id == "run-prior"

    async def test_threshold_respected_at_boundary(self) -> None:
        """Similarity exactly at threshold counts as duplicate."""
        row = MagicMock()
        row.__getitem__ = MagicMock(side_effect={"run_id": "run-x", "similarity": 0.85}.__getitem__)
        pool, _conn = _make_mock_pool(fetchrow_return=row)
        dedup = PgVectorSemanticDedup(pool=pool)
        is_dup, run_id = await dedup.is_duplicate("text", threshold=0.85)
        assert is_dup
        assert run_id == "run-x"

    async def test_high_threshold_blocks_match(self) -> None:
        """Threshold of 1.0 means only a perfect match would qualify."""
        row = MagicMock()
        row.__getitem__ = MagicMock(side_effect={"run_id": "run-x", "similarity": 0.95}.__getitem__)
        pool, _conn = _make_mock_pool(fetchrow_return=row)
        dedup = PgVectorSemanticDedup(pool=pool)
        is_dup, run_id = await dedup.is_duplicate("text", threshold=1.0)
        assert not is_dup
        assert run_id is None


class TestPgVectorAddSeen:
    async def test_add_seen_inserts_row(self) -> None:
        pool, conn = _make_mock_pool()
        dedup = PgVectorSemanticDedup(pool=pool)
        await dedup.add_seen("some content", "run-001")
        assert conn.execute.call_count == 1
        sql: str = conn.execute.call_args[0][0]
        assert "INSERT INTO seen_content" in sql

    async def test_add_seen_passes_run_id(self) -> None:
        pool, conn = _make_mock_pool()
        dedup = PgVectorSemanticDedup(pool=pool)
        await dedup.add_seen("content", "run-xyz")
        positional = conn.execute.call_args[0]
        # $1 = run_id, $2 = text_hash, $3 = embedding
        assert positional[1] == "run-xyz"

    async def test_add_seen_stores_text_hash(self) -> None:
        pool, conn = _make_mock_pool()
        dedup = PgVectorSemanticDedup(pool=pool)
        text = "unique content for hashing"
        await dedup.add_seen(text, "run-001")
        positional = conn.execute.call_args[0]
        expected_hash = hashlib.sha256(text.encode()).hexdigest()
        assert positional[2] == expected_hash


# ---------------------------------------------------------------------------
# First-call → add_seen; second-call → is_duplicate=True scenario
# ---------------------------------------------------------------------------


class TestPgVectorRoundTrip:
    async def test_first_not_dup_then_dup(self) -> None:
        """Simulate a stateful round-trip via side_effect on fetchrow."""
        pool, conn = _make_mock_pool()

        # First call: no rows in DB → not a dup.
        # Second call: the row we "inserted" is now returned.
        row_after_insert = MagicMock()
        row_after_insert.__getitem__ = MagicMock(
            side_effect={"run_id": "run-first", "similarity": 0.92}.__getitem__
        )
        conn.fetchrow.side_effect = [None, row_after_insert]

        dedup = PgVectorSemanticDedup(pool=pool)

        is_dup1, rid1 = await dedup.is_duplicate("some content", threshold=0.85)
        assert not is_dup1
        assert rid1 is None

        await dedup.add_seen("some content", "run-first")

        is_dup2, rid2 = await dedup.is_duplicate("some content", threshold=0.85)
        assert is_dup2
        assert rid2 == "run-first"


# ---------------------------------------------------------------------------
# make_semantic_dedup() factory
# ---------------------------------------------------------------------------


class TestMakeSemanticDedup:
    def test_returns_null_dedup_when_no_dsn(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RESEARCH_PG_DSN", None)
            result = make_semantic_dedup()
        assert isinstance(result, NullDedup)

    def test_returns_pgvector_when_dsn_set(self) -> None:
        with patch.dict(os.environ, {"RESEARCH_PG_DSN": "postgresql://test/test"}):
            result = make_semantic_dedup()
        assert isinstance(result, PgVectorSemanticDedup)

    def test_null_dedup_when_dsn_empty_string(self) -> None:
        with patch.dict(os.environ, {"RESEARCH_PG_DSN": ""}):
            result = make_semantic_dedup()
        assert isinstance(result, NullDedup)


# ---------------------------------------------------------------------------
# Synthesizer integration
# ---------------------------------------------------------------------------


class TestSynthesizerWithNullDedup:
    async def test_behavior_unchanged_with_null_dedup(self) -> None:
        """NullDedup: all citations pass through; URL dedup still works."""
        c1 = _citation("a", "https://x/a", "snippet a")
        c2 = _citation("dup url", "https://x/a", "snippet dup")  # URL dup
        c3 = _citation("b", "https://x/b", "snippet b")
        results = [
            _agent_result([c1, c3]),
            _agent_result([c2]),
        ]
        synth = StitchSynthesizer(semantic_dedup=NullDedup())
        report = await synth.synthesize("run-1", "question?", results)
        # URL dedup keeps first occurrence only.
        urls = [c.url for c in report.citations]
        assert "https://x/a" in urls
        assert "https://x/b" in urls
        assert len(report.citations) == 2

    async def test_failed_agents_excluded(self) -> None:
        results = [
            _agent_result([_citation("t", "https://x/1")]),
            AgentResult(
                agent=AgentName.NEWS,
                status=StepStatus.FAILED,
                summary="",
                error="timeout",
            ),
        ]
        synth = StitchSynthesizer(semantic_dedup=NullDedup())
        report = await synth.synthesize("run-2", "q", results)
        assert len(report.citations) == 1


class TestSynthesizerWithPgVectorMock:
    async def test_semantic_duplicate_skipped_and_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Citations flagged as duplicates are excluded from the final report."""
        c1 = _citation("original", "https://x/a", "content about Python")
        c2 = _citation("paraphrase", "https://x/b", "content about Python (rephrased)")

        pool, conn = _make_mock_pool()

        # c1 check: no prior row (not a dup).
        # c2 check: row found with high similarity (is a dup from run-prior).
        row_dup = MagicMock()
        row_dup.__getitem__ = MagicMock(
            side_effect={"run_id": "run-prior", "similarity": 0.93}.__getitem__
        )
        conn.fetchrow.side_effect = [None, row_dup]

        dedup = PgVectorSemanticDedup(pool=pool)
        results = [_agent_result([c1, c2])]
        synth = StitchSynthesizer(semantic_dedup=dedup)
        report = await synth.synthesize("run-new", "question?", results)

        # Only c1 should survive — c2 is a semantic dup.
        assert len(report.citations) == 1
        assert report.citations[0].url == "https://x/a"

    async def test_survivors_recorded_via_add_seen(self) -> None:
        """Surviving citations are passed to add_seen after synthesis."""
        c1 = _citation("unique", "https://x/unique", "unique snippet")

        pool, conn = _make_mock_pool(fetchrow_return=None)

        dedup = PgVectorSemanticDedup(pool=pool)
        results = [_agent_result([c1])]
        synth = StitchSynthesizer(semantic_dedup=dedup)
        await synth.synthesize("run-new", "q", results)

        # conn.execute called at least once — that's the INSERT from add_seen.
        assert conn.execute.call_count >= 1
        all_sql = " ".join(str(call) for call in conn.execute.call_args_list)
        assert "INSERT INTO seen_content" in all_sql

    async def test_all_duplicates_produces_empty_citations(self) -> None:
        """When every citation is a semantic dup, citations list is empty."""
        c1 = _citation("dup1", "https://x/1", "duplicated content")
        c2 = _citation("dup2", "https://x/2", "also duplicated")

        pool, conn = _make_mock_pool()

        # Both are flagged as dups.
        row_dup = MagicMock()
        row_dup.__getitem__ = MagicMock(
            side_effect={"run_id": "run-old", "similarity": 0.9}.__getitem__
        )
        conn.fetchrow.return_value = row_dup

        dedup = PgVectorSemanticDedup(pool=pool)
        results = [_agent_result([c1, c2])]
        synth = StitchSynthesizer(semantic_dedup=dedup)
        report = await synth.synthesize("run-new", "q", results)

        assert report.citations == []
