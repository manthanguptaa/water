"""Cron/scheduled flow execution for the Water framework."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ScheduledJob:
    """Represents a single scheduled flow execution."""

    job_id: str
    flow: Any
    input_data: Any
    cron_expr: Optional[str] = None
    interval_seconds: Optional[float] = None
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    enabled: bool = True


def _cron_matches(cron_expr: str, dt: datetime) -> bool:
    """Check whether a datetime matches a cron expression.

    Supports 5 fields: minute hour day_of_month month day_of_week.
    Each field may be:
      * ``*``        — matches any value
      * ``*/N``      — matches when value is divisible by N
      * ``N``        — matches a specific number
      * ``A,B,C``    — matches any of the listed values (may combine with */N)
    """
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        raise ValueError(f"Cron expression must have 5 fields, got {len(fields)}: {cron_expr!r}")

    # Values to compare against for each field
    values = [
        dt.minute,
        dt.hour,
        dt.day,
        dt.month,
        dt.isoweekday() % 7,  # 0=Sunday .. 6=Saturday (standard cron)
    ]

    for pattern, value in zip(fields, values):
        if not _field_matches(pattern, value):
            return False
    return True


def _field_matches(pattern: str, value: int) -> bool:
    """Return True if *value* satisfies a single cron field *pattern*."""
    for part in pattern.split(","):
        part = part.strip()
        if part == "*":
            return True
        if part.startswith("*/"):
            step = int(part[2:])
            if step > 0 and value % step == 0:
                return True
        else:
            try:
                if int(part) == value:
                    return True
            except ValueError:
                raise ValueError(f"Invalid cron field value: {part!r}")
    return False


def _next_cron_run(cron_expr: str, after: datetime) -> datetime:
    """Return the next datetime (minute-resolution) matching *cron_expr* after *after*."""
    # Start from the next whole minute
    candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
    # Search up to 366 days ahead to avoid infinite loops
    limit = after + timedelta(days=366)
    while candidate <= limit:
        if _cron_matches(cron_expr, candidate):
            return candidate
        candidate += timedelta(minutes=1)
    # Fallback — should not normally happen with valid expressions
    return after + timedelta(minutes=1)


class FlowScheduler:
    """Schedule Water flows to run on a cron expression or fixed interval."""

    def __init__(self) -> None:
        self._jobs: Dict[str, ScheduledJob] = {}
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    def schedule(
        self,
        flow: Any,
        input_data: Any,
        cron_expr: Optional[str] = None,
        interval_seconds: Optional[float] = None,
        job_id: Optional[str] = None,
    ) -> str:
        """Register a flow to run on a schedule.

        Either *cron_expr* or *interval_seconds* must be provided.
        Returns the job id.
        """
        if cron_expr is None and interval_seconds is None:
            raise ValueError("Either cron_expr or interval_seconds must be provided")

        if job_id is None:
            job_id = f"job_{uuid.uuid4().hex[:8]}"

        now = datetime.now()

        if interval_seconds is not None:
            next_run = now + timedelta(seconds=interval_seconds)
        else:
            assert cron_expr is not None
            next_run = _next_cron_run(cron_expr, now)

        job = ScheduledJob(
            job_id=job_id,
            flow=flow,
            input_data=input_data,
            cron_expr=cron_expr,
            interval_seconds=interval_seconds,
            last_run=None,
            next_run=next_run,
            enabled=True,
        )
        self._jobs[job_id] = job
        return job_id

    def unschedule(self, job_id: str) -> None:
        """Remove a scheduled job."""
        if job_id not in self._jobs:
            raise KeyError(f"No job with id {job_id!r}")
        del self._jobs[job_id]

    def list_jobs(self) -> List[dict]:
        """Return a summary of all registered jobs."""
        result: List[dict] = []
        for job in self._jobs.values():
            entry: dict = {
                "job_id": job.job_id,
                "flow_id": getattr(job.flow, "id", str(job.flow)),
                "next_run": job.next_run,
                "last_run": job.last_run,
                "enabled": job.enabled,
            }
            if job.cron_expr is not None:
                entry["cron_expr"] = job.cron_expr
            if job.interval_seconds is not None:
                entry["interval"] = job.interval_seconds
            result.append(entry)
        return result

    # ------------------------------------------------------------------
    # Scheduler loop
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background scheduler loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.ensure_future(self._loop())

    async def stop(self) -> None:
        """Stop the scheduler loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def tick(self) -> None:
        """Check all jobs and run any that are due.

        This is the core scheduling step — also useful for deterministic testing.
        """
        now = datetime.now()
        for job in list(self._jobs.values()):
            if not job.enabled:
                continue
            if job.next_run is not None and now >= job.next_run:
                await self._run_job(job, now)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        """Background loop that calls tick() every second."""
        try:
            while self._running:
                await self.tick()
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    async def _run_job(self, job: ScheduledJob, now: datetime) -> None:
        """Execute a single job and update its timing metadata."""
        try:
            flow = job.flow
            # Flows expose an async run() method
            await flow.run(job.input_data)
        except Exception:
            # In production you'd route to a DLQ or emit an event; here we
            # continue so the scheduler keeps running.
            logger.exception("Scheduled job '%s' failed", job.job_id)
            pass

        job.last_run = now

        # Compute the next run time
        if job.interval_seconds is not None:
            job.next_run = now + timedelta(seconds=job.interval_seconds)
        elif job.cron_expr is not None:
            job.next_run = _next_cron_run(job.cron_expr, now)
