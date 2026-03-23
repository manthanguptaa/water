"""
Structured Logging & Log Correlation for Water.

Replaces ad-hoc print/logging with structured JSON logging that
automatically correlates logs across tasks in a flow execution.
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class LogFormat(str, Enum):
    """Output format for structured logs."""
    JSON = "json"
    TEXT = "text"


class LogExport(str, Enum):
    """Export destination for logs."""
    STDOUT = "stdout"
    FILE = "file"


@dataclass
class LogContext:
    """
    Carries correlation IDs through the execution chain.

    Automatically injected into every log message.
    """
    flow_id: str = ""
    execution_id: str = ""
    task_id: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        if self.flow_id:
            d["flow_id"] = self.flow_id
        if self.execution_id:
            d["execution_id"] = self.execution_id
        if self.task_id:
            d["task_id"] = self.task_id
        d.update(self.extra)
        return d

    def with_task(self, task_id: str) -> "LogContext":
        """Create a new context with updated task_id."""
        return LogContext(
            flow_id=self.flow_id,
            execution_id=self.execution_id,
            task_id=task_id,
            extra=dict(self.extra),
        )


class StructuredLogger:
    """
    JSON logger that auto-injects context fields.

    Every log line includes flow_id, execution_id, task_id, and timestamp.

    Args:
        level: Log level ("DEBUG", "INFO", "WARN", "ERROR").
        format: Output format ("json" or "text").
        export: Export destination ("stdout" or "file").
        file_path: Path to log file (when export="file").
        redact_fields: Fields to redact from log output.
        sample_rate: Log sampling rate (0.0 to 1.0). 1.0 = log everything.
    """

    def __init__(
        self,
        level: str = "INFO",
        format: str = "json",
        export: str = "stdout",
        file_path: Optional[str] = None,
        redact_fields: Optional[List[str]] = None,
        sample_rate: float = 1.0,
    ):
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.format = LogFormat(format)
        self.export = LogExport(export)
        self.file_path = file_path
        self.redact_fields = set(redact_fields or [])
        self.sample_rate = sample_rate
        self._context = LogContext()
        self._log_buffer: List[Dict[str, Any]] = []

        self._logger = logging.getLogger("water.structured")
        self._logger.setLevel(self.level)

        if not self._logger.handlers:
            if self.export == LogExport.FILE and self.file_path:
                handler = logging.FileHandler(self.file_path)
            else:
                handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(self.level)
            self._logger.addHandler(handler)

    @property
    def context(self) -> LogContext:
        """Current log context."""
        return self._context

    def set_context(
        self,
        flow_id: str = "",
        execution_id: str = "",
        task_id: str = "",
        **extra: Any,
    ) -> None:
        """Update the log context."""
        self._context = LogContext(
            flow_id=flow_id or self._context.flow_id,
            execution_id=execution_id or self._context.execution_id,
            task_id=task_id or self._context.task_id,
            extra={**self._context.extra, **extra},
        )

    def debug(self, msg: str, **kwargs: Any) -> None:
        """Log at DEBUG level."""
        self._log("DEBUG", msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        """Log at INFO level."""
        self._log("INFO", msg, **kwargs)

    def warn(self, msg: str, **kwargs: Any) -> None:
        """Log at WARN level."""
        self._log("WARN", msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        """Log at ERROR level."""
        self._log("ERROR", msg, **kwargs)

    def _log(self, level: str, msg: str, **kwargs: Any) -> None:
        """Internal log method."""
        # Sampling
        if self.sample_rate < 1.0:
            import random
            if random.random() > self.sample_rate:
                return

        log_level = getattr(logging, level.upper().replace("WARN", "WARNING"), logging.INFO)
        if log_level < self.level:
            return

        record = {
            "level": level,
            "msg": msg,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **self._context.to_dict(),
            **kwargs,
        }

        # Redact sensitive fields
        if self.redact_fields:
            record = self._redact(record)

        self._log_buffer.append(record)

        if self.format == LogFormat.JSON:
            self._logger.log(log_level, json.dumps(record, default=str))
        else:
            ctx_parts = []
            if self._context.flow_id:
                ctx_parts.append(f"flow={self._context.flow_id}")
            if self._context.execution_id:
                ctx_parts.append(f"exec={self._context.execution_id}")
            if self._context.task_id:
                ctx_parts.append(f"task={self._context.task_id}")
            ctx_str = " ".join(ctx_parts)
            extra_str = " ".join(f"{k}={v}" for k, v in kwargs.items())
            parts = [f"[{level}]", msg]
            if ctx_str:
                parts.append(f"({ctx_str})")
            if extra_str:
                parts.append(extra_str)
            self._logger.log(log_level, " ".join(parts))

    def _redact(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive fields from a log record."""
        result = {}
        for k, v in record.items():
            if k in self.redact_fields:
                result[k] = "***REDACTED***"
            elif isinstance(v, dict):
                result[k] = self._redact(v)
            else:
                result[k] = v
        return result

    def get_logs(self) -> List[Dict[str, Any]]:
        """Return the internal log buffer (useful for testing)."""
        return list(self._log_buffer)

    def clear(self) -> None:
        """Clear the log buffer."""
        self._log_buffer.clear()

    def close(self) -> None:
        """Close and remove all handlers from the logger."""
        for handler in self._logger.handlers[:]:
            try:
                handler.close()
            except Exception:
                logging.getLogger(__name__).warning(
                    "Failed to close log handler %s", handler, exc_info=True
                )
            finally:
                self._logger.removeHandler(handler)


class LogExporter:
    """
    Pluggable log export to stdout, file, or external services.

    For custom export destinations, subclass and override ``export()``.
    """

    def __init__(self, destination: str = "stdout", file_path: Optional[str] = None):
        self.destination = destination
        self.file_path = file_path

    def export(self, records: List[Dict[str, Any]]) -> None:
        """Export log records to the configured destination."""
        if self.destination == "file" and self.file_path:
            with open(self.file_path, "a") as f:
                for record in records:
                    f.write(json.dumps(record, default=str) + "\n")
        else:
            for record in records:
                print(json.dumps(record, default=str))
