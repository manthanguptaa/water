import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable

logger = logging.getLogger(__name__)

from water.core.types import SerializableMixin


@dataclass
class InstrumentationConfig:
    service_name: str = "water-service"
    endpoint: Optional[str] = None
    sample_rate: float = 1.0
    capture_input: bool = False
    capture_output: bool = False
    custom_attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class SpanRecord(SerializableMixin):
    """Internal span record for when OTel is not available."""
    name: str
    kind: str = "internal"  # internal, flow, task, llm
    start_time: float = 0.0
    end_time: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: Optional[str] = None
    children: List['SpanRecord'] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


class InstrumentationCollector:
    """Collects instrumentation data (used as fallback when OTel not available)."""
    def __init__(self):
        self.spans: List[SpanRecord] = []
        self._active_spans: Dict[str, SpanRecord] = {}

    def start_span(self, name: str, kind: str = "internal", attributes: Dict[str, Any] = None) -> SpanRecord:
        span = SpanRecord(name=name, kind=kind, start_time=time.time(), attributes=attributes or {})
        self._active_spans[name] = span
        return span

    def end_span(self, name: str, status: str = "ok", error: Optional[str] = None) -> Optional[SpanRecord]:
        span = self._active_spans.pop(name, None)
        if span:
            span.end_time = time.time()
            span.status = status
            span.error = error
            self.spans.append(span)
        return span

    def get_spans(self) -> List[SpanRecord]:
        return list(self.spans)

    def clear(self) -> None:
        self.spans.clear()
        self._active_spans.clear()


class AutoInstrumentor:
    """Provides auto-instrumentation for Water flows and tasks."""

    def __init__(self, config: Optional[InstrumentationConfig] = None):
        self.config = config or InstrumentationConfig()
        self._enabled = False
        self._collector = InstrumentationCollector()
        self._otel_available = False
        self._tracer = None
        self._meter = None

        # Try to set up OTel
        try:
            from opentelemetry import trace, metrics
            self._otel_available = True
        except ImportError:
            self._otel_available = False

    def enable(self) -> 'AutoInstrumentor':
        """Enable instrumentation."""
        self._enabled = True
        if self._otel_available and self.config.endpoint:
            self._setup_otel()
        return self

    def disable(self) -> 'AutoInstrumentor':
        self._enabled = False
        return self

    def _setup_otel(self) -> None:
        """Set up OpenTelemetry exporters (only if OTel is installed)."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.resources import Resource

            resource = Resource.create({"service.name": self.config.service_name})
            provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer(self.config.service_name)
        except ImportError:
            logger.warning("OpenTelemetry SDK packages not available for instrumentation setup")
            pass

    # Middleware interface
    async def before_task(self, task_id: str, data: Dict[str, Any], context: Any) -> Dict[str, Any]:
        if not self._enabled:
            return data
        attrs = {"task.id": task_id, "service.name": self.config.service_name}
        if self.config.capture_input:
            attrs["task.input"] = str(data)[:1000]
        self._collector.start_span(f"task:{task_id}", kind="task", attributes=attrs)
        return data

    async def after_task(self, task_id: str, data: Dict[str, Any], result: Dict[str, Any], context: Any) -> Dict[str, Any]:
        if not self._enabled:
            return result
        attrs = {}
        if self.config.capture_output:
            attrs["task.output"] = str(result)[:1000]
        span = self._collector.end_span(f"task:{task_id}", status="ok")
        if span:
            span.attributes.update(attrs)
        return result

    def get_collector(self) -> InstrumentationCollector:
        return self._collector

    @property
    def is_otel_available(self) -> bool:
        return self._otel_available


def auto_instrument(
    service_name: str = "water-service",
    endpoint: Optional[str] = None,
    sample_rate: float = 1.0,
    capture_input: bool = False,
    capture_output: bool = False,
) -> AutoInstrumentor:
    """One-line setup for auto-instrumentation."""
    config = InstrumentationConfig(
        service_name=service_name,
        endpoint=endpoint,
        sample_rate=sample_rate,
        capture_input=capture_input,
        capture_output=capture_output,
    )
    instrumentor = AutoInstrumentor(config=config)
    return instrumentor.enable()
