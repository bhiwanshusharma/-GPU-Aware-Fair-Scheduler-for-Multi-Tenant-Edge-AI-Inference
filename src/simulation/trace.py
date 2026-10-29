"""Simulation execution trace recording and formatting."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, List, Any, Tuple


class TraceEventType(Enum):
    """Categorization of observable simulation events."""
    DISPATCH = auto()
    RESUME = auto()
    QUANTUM_EXPIRED = auto()
    PREEMPTION = auto()
    COMPLETION = auto()
    IDLE_START = auto()
    IDLE_END = auto()
    CONTEXT_SWITCH = auto()
    QUEUE_STATE = auto()


@dataclass
class TraceEvent:
    """Represents a single atomic event recorded in the execution trace."""
    timestamp_ms: float
    event_type: TraceEventType
    tenant_id: Optional[str] = None
    job_id: Optional[str] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class SimulationTrace:
    """Chronological execution trace container with querying and formatting utilities."""

    def __init__(self) -> None:
        self.events: List[TraceEvent] = []
        self.dispatch_events: List[TraceEvent] = []
        self.preemption_events: List[TraceEvent] = []
        self.completion_events: List[TraceEvent] = []
        self.idle_periods: List[Tuple[float, float, float]] = []  # (start_ms, end_ms, duration_ms)
        self.queue_snapshots: List[Tuple[float, Dict[str, int]]] = []  # (timestamp_ms, {tenant_id: size})

    def record_event(
        self,
        timestamp_ms: float,
        event_type: TraceEventType,
        tenant_id: Optional[str] = None,
        job_id: Optional[str] = None,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> TraceEvent:
        """Appends an event to the trace and categorizes it."""
        event = TraceEvent(
            timestamp_ms=float(timestamp_ms),
            event_type=event_type,
            tenant_id=tenant_id,
            job_id=job_id,
            message=message,
            details=details or {},
        )
        self.events.append(event)

        if event_type in (TraceEventType.DISPATCH, TraceEventType.RESUME):
            self.dispatch_events.append(event)
        elif event_type in (TraceEventType.PREEMPTION, TraceEventType.QUANTUM_EXPIRED):
            self.preemption_events.append(event)
        elif event_type == TraceEventType.COMPLETION:
            self.completion_events.append(event)

        return event

    def record_idle_period(self, start_ms: float, end_ms: float) -> None:
        """Records a contiguous period of GPU idleness."""
        duration = max(0.0, end_ms - start_ms)
        self.idle_periods.append((start_ms, end_ms, duration))

    def record_queue_snapshot(self, timestamp_ms: float, queue_sizes: Dict[str, int]) -> None:
        """Captures a snapshot of tenant queue lengths."""
        self.queue_snapshots.append((float(timestamp_ms), dict(queue_sizes)))

    def format_trace(self, include_idle: bool = True) -> str:
        """Renders the execution trace in human-readable chronological format.
        
        Example format:
            time=0.0
            tenant=A job=A1 dispatched
            time=10.0
            A1 quantum expired
            tenant=B job=B1 dispatched
            time=20.0
            B1 quantum expired
            tenant=A job=A1 resumed
        """
        lines: List[str] = []
        last_time: Optional[float] = None

        for ev in self.events:
            # Skip queue snapshots in plain text trace
            if ev.event_type == TraceEventType.QUEUE_STATE:
                continue
            if not include_idle and ev.event_type in (TraceEventType.IDLE_START, TraceEventType.IDLE_END):
                continue

            # Print timestamp header when time advances
            if last_time is None or ev.timestamp_ms != last_time:
                # Format integer if whole number, else 1 decimal
                t_str = f"{int(ev.timestamp_ms)}" if ev.timestamp_ms.is_integer() else f"{ev.timestamp_ms:.1f}"
                lines.append(f"time={t_str}")
                last_time = ev.timestamp_ms

            if ev.message:
                lines.append(ev.message)
            else:
                if ev.event_type == TraceEventType.DISPATCH:
                    lines.append(f"tenant={ev.tenant_id} job={ev.job_id} dispatched")
                elif ev.event_type == TraceEventType.RESUME:
                    lines.append(f"tenant={ev.tenant_id} job={ev.job_id} resumed")
                elif ev.event_type == TraceEventType.QUANTUM_EXPIRED:
                    lines.append(f"{ev.job_id} quantum expired")
                elif ev.event_type == TraceEventType.PREEMPTION:
                    lines.append(f"tenant={ev.tenant_id} job={ev.job_id} preempted")
                elif ev.event_type == TraceEventType.COMPLETION:
                    lines.append(f"tenant={ev.tenant_id} job={ev.job_id} completed")
                elif ev.event_type == TraceEventType.IDLE_START:
                    lines.append("GPU idle")

        return "\n".join(lines)

    @property
    def total_idle_time_ms(self) -> float:
        """Total sum of recorded idle periods."""
        return sum(dur for _, _, dur in self.idle_periods)
