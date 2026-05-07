"""Eval harness for the research-crew pipeline.

Exports the golden question set and the harness runner so both can be
imported by tests and by ``python -m evals.harness``.
"""

from evals import golden, harness

__all__ = ["golden", "harness"]
