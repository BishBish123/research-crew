"""Test-suite-wide fixtures and Hypothesis configuration.

Hypothesis defaults to a filesystem-backed example database under
`.hypothesis/examples`. In sandboxed CI environments where the test
runner cannot write to the workspace (warnings-as-errors plus a
read-only mount) this trips the database-creation step before any
property even runs. Register and load a profile that disables the
database so the property tests in `test_dedup_property.py` are
portable across local dev and a no-disk-write sandbox.
"""

from __future__ import annotations

from hypothesis import HealthCheck, settings

# `database=None` makes Hypothesis hold its example shrink history in
# memory only — the per-test `max_examples` budget is unaffected, so
# `@settings(max_examples=200)` still draws 200 fresh examples.
settings.register_profile(
    "ci-no-db",
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("ci-no-db")
