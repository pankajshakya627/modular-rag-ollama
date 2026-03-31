"""Observability module for Modular RAG.

Provides:
- structlog-based structured JSON logging (console-friendly in dev, JSON in prod)
- Prometheus metrics: query latency, error count, retrieval/reranking counts
- @trace_span decorator for lightweight request-scoped timing
"""
import functools
import logging
import os
import time
from contextvars import ContextVar
from typing import Any, Callable, Dict, Optional, TypeVar

import structlog

# ---------------------------------------------------------------------------
# Structured Logging Setup
# ---------------------------------------------------------------------------

_ENVIRONMENT = os.getenv("RAG_ENVIRONMENT", "development").lower()
_IS_PRODUCTION = _ENVIRONMENT in ("staging", "production")


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structlog for the environment.

    - Development: human-readable colored console output
    - Staging/Production: JSON output for log aggregators (Datadog, Loki, etc.)
    """
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if _IS_PRODUCTION:
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper(), logging.INFO),
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger by name."""
    return structlog.get_logger(name)


# ---------------------------------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------------------------------

_metrics_available = False
_query_duration_histogram = None
_query_errors_counter = None
_retrieval_count_counter = None
_reranking_count_counter = None
_index_doc_counter = None

try:
    from prometheus_client import Counter, Histogram, REGISTRY

    _query_duration_histogram = Histogram(
        "rag_query_duration_seconds",
        "End-to-end RAG query latency in seconds",
        buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
    )
    _query_errors_counter = Counter(
        "rag_query_errors_total",
        "Total number of RAG query errors",
        ["stage"],
    )
    _retrieval_count_counter = Counter(
        "rag_retrieval_docs_total",
        "Total number of documents retrieved",
        ["method"],
    )
    _reranking_count_counter = Counter(
        "rag_reranking_docs_total",
        "Total number of documents passed to reranker",
    )
    _index_doc_counter = Counter(
        "rag_indexed_documents_total",
        "Total number of documents indexed",
    )
    _metrics_available = True
except Exception:
    pass  # prometheus_client not installed — metrics are no-ops


def record_query_duration(duration_seconds: float) -> None:
    """Record query end-to-end latency."""
    if _metrics_available and _query_duration_histogram:
        _query_duration_histogram.observe(duration_seconds)


def record_query_error(stage: str) -> None:
    """Increment error counter for a pipeline stage."""
    if _metrics_available and _query_errors_counter:
        _query_errors_counter.labels(stage=stage).inc()


def record_retrieval(method: str, count: int) -> None:
    """Record number of docs retrieved by a given method."""
    if _metrics_available and _retrieval_count_counter:
        _retrieval_count_counter.labels(method=method).inc(count)


def record_reranking(count: int) -> None:
    """Record number of docs sent to the reranker."""
    if _metrics_available and _reranking_count_counter:
        _reranking_count_counter.inc(count)


def record_indexed_documents(count: int) -> None:
    """Record number of documents successfully indexed."""
    if _metrics_available and _index_doc_counter:
        _index_doc_counter.inc(count)


def get_metrics_content() -> bytes:
    """Generate Prometheus metrics exposition text for /metrics endpoint."""
    if not _metrics_available:
        return b"# prometheus_client not installed\n"
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return generate_latest()


PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# ---------------------------------------------------------------------------
# Request-ID Context Variable
# ---------------------------------------------------------------------------

_request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(request_id: str) -> None:
    """Set the current request ID in context."""
    _request_id_var.set(request_id)
    structlog.contextvars.bind_contextvars(request_id=request_id)


def get_request_id() -> str:
    """Get the current request ID."""
    return _request_id_var.get()


# ---------------------------------------------------------------------------
# @trace_span Decorator
# ---------------------------------------------------------------------------

F = TypeVar("F", bound=Callable[..., Any])


def trace_span(name: Optional[str] = None) -> Callable[[F], F]:
    """Decorator that logs entry/exit timing for a function.

    Usage:
        @trace_span("dense_retrieval")
        def _dense_retrieval(self, state):
            ...
    """
    def decorator(func: F) -> F:
        span_name = name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log = get_logger(__name__)
            start = time.perf_counter()
            log.debug("span_start", span=span_name, request_id=get_request_id())
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                log.debug(
                    "span_end",
                    span=span_name,
                    duration_ms=round(elapsed * 1000, 2),
                    request_id=get_request_id(),
                )
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - start
                log.error(
                    "span_error",
                    span=span_name,
                    duration_ms=round(elapsed * 1000, 2),
                    error=str(exc),
                    request_id=get_request_id(),
                )
                raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            log = get_logger(__name__)
            start = time.perf_counter()
            log.debug("span_start", span=span_name, request_id=get_request_id())
            try:
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                log.debug(
                    "span_end",
                    span=span_name,
                    duration_ms=round(elapsed * 1000, 2),
                    request_id=get_request_id(),
                )
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - start
                log.error(
                    "span_error",
                    span=span_name,
                    duration_ms=round(elapsed * 1000, 2),
                    error=str(exc),
                    request_id=get_request_id(),
                )
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return wrapper  # type: ignore[return-value]

    return decorator
