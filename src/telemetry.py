"""Small OpenTelemetry helpers for FraudGuard spans."""
from __future__ import annotations

import functools
import logging
import os
from contextlib import contextmanager
from typing import Any, Callable

log = logging.getLogger(__name__)
_TRACER = None


def get_tracer():
    """Return an SDK-backed tracer, preserving a provider configured by the host."""
    global _TRACER
    if _TRACER is not None:
        return _TRACER
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        from opentelemetry.trace import ProxyTracerProvider

        provider = trace.get_tracer_provider()
        if isinstance(provider, ProxyTracerProvider):
            provider = TracerProvider(resource=Resource.create({
                "service.name": os.getenv("OTEL_SERVICE_NAME", "fraudguard-copilot")
            }))
            endpoint = (os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
                        or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))
            if endpoint:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            elif os.getenv("OTEL_TRACES_EXPORTER", "").lower() == "console":
                provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            trace.set_tracer_provider(provider)
        _TRACER = trace.get_tracer("fraudguard.copilot")
        return _TRACER
    except ImportError:
        _TRACER = None
        return None


@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None):
    tracer = get_tracer()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as current:
        for key, value in (attributes or {}).items():
            if value is not None:
                current.set_attribute(key, value)
        try:
            yield current
        except Exception as exc:
            from opentelemetry.trace import Status, StatusCode
            # Do not copy exception messages into spans; they may contain SAR
            # fields or retrieved document excerpts.
            current.set_attribute("error.type", type(exc).__name__)
            current.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise


def traced(name: str | None = None):
    """Trace a sync function without adding transaction or document contents."""
    def decorate(function: Callable):
        span_name = name or function.__qualname__

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            with span(span_name) as current:
                result = function(*args, **kwargs)
                if current is not None and isinstance(result, (list, tuple)):
                    current.set_attribute("result.count", len(result))
                return result
        return wrapped
    return decorate
