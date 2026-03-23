__all__ = [
    "get_tracer",
    "TelemetryManager",
    "NoOpTelemetry",
    "is_otel_available",
]

"""
Optional OpenTelemetry integration for Water flows.

If opentelemetry is installed, provides automatic span creation for
flow and task execution. If not installed, all operations are no-ops.
"""
import contextvars
import logging
from typing import Any, Dict, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_tracer_var: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "_tracer_var", default=None
)
_otel_available_var: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_otel_available_var", default=False
)

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode

    _otel_available_var = contextvars.ContextVar(
        "_otel_available_var", default=True
    )
except ImportError:
    logger.debug("OpenTelemetry not installed; telemetry will be disabled")
    pass


def is_otel_enabled() -> bool:
    """Return whether OpenTelemetry is available in the current context."""
    return _otel_available_var.get()


def get_tracer(name: str = "water"):
    """Get or create an OpenTelemetry tracer. Returns None if OTel is not installed."""
    if not is_otel_enabled():
        return None
    tracer = _tracer_var.get()
    if tracer is None:
        tracer = trace.get_tracer(name)
        _tracer_var.set(tracer)
    return tracer


class TelemetryManager:
    """
    Manages telemetry for flow execution.

    Wraps OpenTelemetry tracing. If OTel is not installed, all methods
    are no-ops so there's zero overhead.
    """

    def __init__(self, enabled: bool = True, tracer_name: str = "water") -> None:
        self.enabled = enabled and is_otel_enabled()
        self._tracer_name = tracer_name
        self._tracer = get_tracer(tracer_name) if self.enabled else None

    @property
    def is_active(self) -> bool:
        return self.enabled and self._tracer is not None

    @contextmanager
    def flow_span(self, flow_id: str, **attributes):
        """Create a span for a flow execution."""
        if not self.is_active:
            yield None
            return

        with self._tracer.start_as_current_span(
            f"flow:{flow_id}",
            attributes={"water.flow_id": flow_id, **attributes},
        ) as span:
            yield span

    @contextmanager
    def task_span(self, task_id: str, flow_id: str, attempt: int = 1, **attributes):
        """Create a span for a task execution."""
        if not self.is_active:
            yield None
            return

        with self._tracer.start_as_current_span(
            f"task:{task_id}",
            attributes={
                "water.task_id": task_id,
                "water.flow_id": flow_id,
                "water.attempt": attempt,
                **attributes,
            },
        ) as span:
            yield span

    def record_error(self, span, error: Exception) -> None:
        """Record an error on a span."""
        if not self.is_active or span is None:
            return
        span.set_status(StatusCode.ERROR, str(error))
        span.record_exception(error)

    def set_success(self, span) -> None:
        """Mark a span as successful."""
        if not self.is_active or span is None:
            return
        span.set_status(StatusCode.OK)


class NoOpTelemetry(TelemetryManager):
    """A telemetry manager that does nothing. Used when telemetry is disabled."""

    def __init__(self):
        super().__init__(enabled=False)


# Convenience function to check if OTel is available
def is_otel_available() -> bool:
    return is_otel_enabled()
