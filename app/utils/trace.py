import contextvars
import uuid

# Global context variable for the trace ID
_trace_id_var = contextvars.ContextVar("trace_id", default=None)

def set_trace_id(trace_id: str | None = None) -> str:
    """Set the trace ID for the current context. Generates a new one if not provided."""
    if not trace_id:
        trace_id = f"trace-{uuid.uuid4().hex[:12]}"
    _trace_id_var.set(trace_id)
    return trace_id

def get_trace_id() -> str | None:
    """Get the current trace ID, or None if not set."""
    return _trace_id_var.get()
