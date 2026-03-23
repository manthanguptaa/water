__all__ = [
    "CircuitBreakerOpen",
    "CircuitBreaker",
]

"""
Circuit Breaker pattern for protecting external API calls.

Stops calls after consecutive failures and allows recovery after a timeout.
States:
  - CLOSED: Normal operation, calls pass through.
  - OPEN: Too many failures, calls are rejected immediately.
  - HALF_OPEN: Recovery timeout elapsed, one test call is allowed.
"""

import time
from typing import Optional


class CircuitBreakerOpen(Exception):
    """Raised when a call is attempted while the circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker that opens after consecutive failures and recovers after a timeout.

    Args:
        failure_threshold: Number of consecutive failures before the circuit opens.
        recovery_timeout: Seconds to wait before allowing a test call (half-open state).
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failure_count: int = 0
        self._opened_at: Optional[float] = None
        self._state: str = "closed"

    def record_success(self) -> None:
        """Reset failure count and close the circuit."""
        self._failure_count = 0
        self._state = "closed"
        self._opened_at = None

    def record_failure(self) -> None:
        """Increment failure count and open the circuit if threshold is reached."""
        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            self._state = "open"
            self._opened_at = time.monotonic()

    def can_execute(self) -> bool:
        """Return True if the circuit is closed or half-open (recovery timeout elapsed)."""
        if self._state == "closed":
            return True
        if self._state == "open" and self._opened_at is not None:
            if time.monotonic() - self._opened_at >= self.recovery_timeout:
                self._state = "half_open"
                return True
        if self._state == "half_open":
            return True
        return False

    @property
    def state(self) -> str:
        """Return the current state: 'closed', 'open', or 'half_open'."""
        # Check for transition to half_open on read
        if self._state == "open" and self._opened_at is not None:
            if time.monotonic() - self._opened_at >= self.recovery_timeout:
                self._state = "half_open"
        return self._state
