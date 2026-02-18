"""Property-based tests for the workflow's idempotency-key derivation.

The dedup key is a stability contract: the same `(run_id, agent, question)`
triple *must* hash to the same key forever, and any single component
change *must* produce a different key. Hypothesis is much better than
hand-rolled fixtures at finding the corner cases (empty strings, unicode,
huge inputs).
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from research_crew.agents import MockAgent
from research_crew.models import AgentName
from research_crew.workflow import WorkflowEngine

# Reasonable bounds: real run_ids are uuids, real questions are user input.
_run_ids = st.text(min_size=1, max_size=64)
_questions = st.text(min_size=1, max_size=512)
_agent_names = st.sampled_from(list(AgentName))


def _engine(run_id: str) -> WorkflowEngine:
    return WorkflowEngine(run_id=run_id)


@settings(max_examples=200)
@given(run_id=_run_ids, agent=_agent_names, question=_questions)
def test_key_is_deterministic(run_id: str, agent: AgentName, question: str) -> None:
    a = MockAgent(name=agent)
    e1 = _engine(run_id)
    e2 = _engine(run_id)
    assert e1._dedup_key(a, question) == e2._dedup_key(a, question)


@settings(max_examples=200)
@given(
    run_id=_run_ids,
    agent=_agent_names,
    q1=_questions,
    q2=_questions,
)
def test_different_questions_give_different_keys(
    run_id: str, agent: AgentName, q1: str, q2: str
) -> None:
    if q1 == q2:
        return  # vacuous
    a = MockAgent(name=agent)
    e = _engine(run_id)
    assert e._dedup_key(a, q1) != e._dedup_key(a, q2)


@settings(max_examples=200)
@given(r1=_run_ids, r2=_run_ids, agent=_agent_names, question=_questions)
def test_different_run_ids_give_different_keys(
    r1: str, r2: str, agent: AgentName, question: str
) -> None:
    if r1 == r2:
        return
    a = MockAgent(name=agent)
    assert _engine(r1)._dedup_key(a, question) != _engine(r2)._dedup_key(a, question)


@settings(max_examples=200)
@given(run_id=_run_ids, a1=_agent_names, a2=_agent_names, question=_questions)
def test_different_agents_give_different_keys(
    run_id: str, a1: AgentName, a2: AgentName, question: str
) -> None:
    if a1 == a2:
        return
    e = _engine(run_id)
    k1 = e._dedup_key(MockAgent(name=a1), question)
    k2 = e._dedup_key(MockAgent(name=a2), question)
    assert k1 != k2


@settings(max_examples=100)
@given(run_id=_run_ids, agent=_agent_names, question=_questions)
def test_key_format_is_step_prefixed(run_id: str, agent: AgentName, question: str) -> None:
    """All keys must use the `step:` namespace so the cache + step
    keyspaces never collide in Redis."""
    key = _engine(run_id)._dedup_key(MockAgent(name=agent), question)
    assert key.startswith("step:")
    # 24-char hex digest from blake2b(digest_size=12) ⇒ total length 5+24.
    assert len(key) == 5 + 24
