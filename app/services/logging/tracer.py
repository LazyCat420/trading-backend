import logging
from functools import wraps
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

logger = logging.getLogger(__name__)

# Initialize basic Tracer
try:
    provider = TracerProvider()
    # For now, just export to console so we can see traces in the docker logs.
    # Later this can be switched to OTLP Exporter (Jaeger/Datadog)
    processor = BatchSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("trading-service.pipeline")
except Exception as e:
    logger.error("Failed to initialize OpenTelemetry: %s", e)
    tracer = trace.get_tracer(__name__)

def trace_span(span_name: str):
    """Decorator to trace a function execution time."""
    def decorator(func):
        import asyncio
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(span_name) as span:
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        span.record_exception(e)
                        raise
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(span_name) as span:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        span.record_exception(e)
                        raise
            return sync_wrapper
    return decorator
