"""Hand-curated golden eval set for the research-crew pipeline.

Each question covers one of the following categories:
  - factual:     single-entity lookup
  - comparative: "X vs Y" analysis
  - list:        enumeration / survey questions
  - trend:       "what changed in X between year-range" questions
  - oos:         out-of-scope / failure-likely questions (MockAgent expected to score 0.0)
  - refusal:     ill-defined questions where the correct answer is a clarifying refusal

The ``expected_url_substrings`` and ``expected_keyphrases`` fields define
what a *real* answer pipeline would need to satisfy.  With MockAgent the
citation URLs are always ``example.com`` and summaries are templated, so
scores reflect the mock floor, not true retrieval quality.  That is
intentional — the harness is honest.

``expected_keyphrases`` are matched case-insensitively against the full
synthesized report markdown, so phrases that appear in the question itself
(which MockAgent echoes back in its summary) will score 1.0, while phrases
that would only appear in a real synthesized answer score 0.0.  The mix is
deliberate: it makes the keyphrase metric non-trivially interpretable even
at the mock-pipeline floor.

Set size: 20 questions (brief mandates ≥ 20).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GoldenQuestion:
    """One entry in the golden eval set."""

    qid: str
    question: str
    category: str  # "factual" | "comparative" | "list"
    # Substrings any cited URL must contain to count as a good citation.
    # With MockAgent all URLs are example.com, so citation_correctness = 0.0.
    expected_url_substrings: list[str] = field(default_factory=list)
    # Phrases (case-insensitive) the synthesized answer should contain.
    expected_keyphrases: list[str] = field(default_factory=list)


#: The canonical golden set.  Add questions here; the harness picks them up automatically.
GOLDEN_SET: list[GoldenQuestion] = [
    # --- factual -----------------------------------------------------------
    GoldenQuestion(
        qid="factual-001",
        question="What is Python's Global Interpreter Lock and why does it matter?",
        category="factual",
        expected_url_substrings=["python.org", "docs.python", "realpython"],
        expected_keyphrases=[
            "python",
            "global interpreter lock",
            "threading",
        ],
    ),
    GoldenQuestion(
        qid="factual-002",
        question="How does Redis handle persistence with RDB and AOF?",
        category="factual",
        expected_url_substrings=["redis.io", "redis.com"],
        expected_keyphrases=[
            "redis",
            "persistence",
            "rdb",
        ],
    ),
    GoldenQuestion(
        qid="factual-003",
        question="What is exponential backoff and when should it be used?",
        category="factual",
        expected_url_substrings=["aws.amazon.com", "cloud.google.com", "martinfowler.com"],
        expected_keyphrases=[
            "exponential backoff",
            "retry",
            "jitter",
        ],
    ),
    # --- comparative -------------------------------------------------------
    GoldenQuestion(
        qid="comp-001",
        question="asyncio vs threading in Python: when to choose each?",
        category="comparative",
        expected_url_substrings=["python.org", "realpython", "docs.python"],
        expected_keyphrases=[
            "asyncio",
            "threading",
            "python",
        ],
    ),
    GoldenQuestion(
        qid="comp-002",
        question="Redis Streams vs Kafka: trade-offs for event-driven microservices",
        category="comparative",
        expected_url_substrings=["redis.io", "kafka.apache.org", "confluent.io"],
        expected_keyphrases=[
            "redis",
            "kafka",
            "streams",
        ],
    ),
    GoldenQuestion(
        qid="comp-003",
        question="FastAPI vs Flask: differences in async support and performance",
        category="comparative",
        expected_url_substrings=["fastapi.tiangolo.com", "flask.palletsprojects.com"],
        expected_keyphrases=[
            "fastapi",
            "flask",
            "async",
        ],
    ),
    # --- list --------------------------------------------------------------
    GoldenQuestion(
        qid="list-001",
        question="List the key design patterns used in distributed workflow engines",
        category="list",
        expected_url_substrings=["martinfowler.com", "aws.amazon.com", "temporal.io"],
        expected_keyphrases=[
            "workflow",
            "idempotency",
            "retry",
        ],
    ),
    GoldenQuestion(
        qid="list-002",
        question="What are the main observability pillars: metrics, logs, traces?",
        category="list",
        expected_url_substrings=["opentelemetry.io", "datadoghq.com", "grafana.com"],
        expected_keyphrases=[
            "metrics",
            "logs",
            "traces",
        ],
    ),
    GoldenQuestion(
        qid="list-003",
        question="List best practices for API rate limiting and throttling",
        category="list",
        expected_url_substrings=["cloud.google.com", "stripe.com", "aws.amazon.com"],
        expected_keyphrases=[
            "rate limiting",
            "throttling",
            "429",
        ],
    ),
    # --- factual (additional) -----------------------------------------------
    GoldenQuestion(
        qid="factual-004",
        question="What is consistent hashing and how does it reduce cache invalidation?",
        category="factual",
        expected_url_substrings=["martinfowler.com", "aws.amazon.com", "highscalability.com"],
        expected_keyphrases=[
            "consistent hashing",
            "virtual nodes",
            "cache invalidation",
        ],
    ),
    GoldenQuestion(
        qid="factual-005",
        question="How does the asyncio event loop schedule coroutines in CPython?",
        category="factual",
        expected_url_substrings=["docs.python.org", "python.org", "realpython.com"],
        expected_keyphrases=[
            "event loop",
            "asyncio",
            "selector",
        ],
    ),
    # --- comparative (additional) -------------------------------------------
    GoldenQuestion(
        qid="comp-004",
        question="PostgreSQL vs MySQL: differences in MVCC implementation and locking",
        category="comparative",
        expected_url_substrings=["postgresql.org", "dev.mysql.com", "percona.com"],
        expected_keyphrases=[
            "postgresql",
            "mysql",
            "mvcc",
        ],
    ),
    GoldenQuestion(
        qid="comp-005",
        question="gRPC vs REST: when to choose each for inter-service communication",
        category="comparative",
        expected_url_substrings=["grpc.io", "martinfowler.com", "cloud.google.com"],
        expected_keyphrases=[
            "grpc",
            "rest",
            "protobuf",
        ],
    ),
    # --- list (additional) --------------------------------------------------
    GoldenQuestion(
        qid="list-004",
        question="List the CAP theorem properties and how distributed databases trade them off",
        category="list",
        expected_url_substrings=["martinfowler.com", "cockroachlabs.com", "aws.amazon.com"],
        expected_keyphrases=[
            "cap theorem",
            "consistency",
            "partition tolerance",
        ],
    ),
    GoldenQuestion(
        qid="list-005",
        question="What are the SOLID principles in object-oriented design?",
        category="list",
        expected_url_substrings=["martinfowler.com", "refactoring.guru", "clean-code"],
        expected_keyphrases=[
            "solid",
            "single responsibility",
            "open/closed",
        ],
    ),
    # --- trend / recency ----------------------------------------------------
    GoldenQuestion(
        qid="trend-001",
        question="What changed in Python's asyncio between 2024 and 2026?",
        category="trend",
        expected_url_substrings=["docs.python.org", "python.org", "peps.python.org"],
        expected_keyphrases=[
            "asyncio",
            "python",
            "task group",
        ],
    ),
    GoldenQuestion(
        qid="trend-002",
        question="How has Kubernetes networking evolved between 2024 and 2026?",
        category="trend",
        expected_url_substrings=["kubernetes.io", "cncf.io", "cilium.io"],
        expected_keyphrases=[
            "kubernetes",
            "networking",
            "gateway api",
        ],
    ),
    # --- out-of-scope / failure-likely (MockAgent expected to score 0.0) ----
    GoldenQuestion(
        qid="oos-001",
        question="What is the current price of Bitcoin in USD right now?",
        category="oos",
        expected_url_substrings=["coinmarketcap.com", "coingecko.com", "finance.yahoo.com"],
        expected_keyphrases=[
            "bitcoin",
            "usd",
            "real-time price",
        ],
    ),
    GoldenQuestion(
        qid="oos-002",
        question="Summarise the full text of RFC 9110 (HTTP Semantics, 2022)",
        category="oos",
        expected_url_substrings=["rfc-editor.org", "ietf.org", "httpwg.org"],
        expected_keyphrases=[
            "rfc 9110",
            "http semantics",
            "request method",
        ],
    ),
    # --- refusal (ill-defined; correct answer is a clarifying refusal) ------
    GoldenQuestion(
        qid="refusal-001",
        question="What is the best programming language?",
        category="refusal",
        expected_url_substrings=["stackoverflow.com", "tiobe.com", "github.com"],
        expected_keyphrases=[
            "depends",
            "use case",
            "trade-offs",
        ],
    ),
]
