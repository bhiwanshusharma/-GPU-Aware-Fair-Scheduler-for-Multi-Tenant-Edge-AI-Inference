"""Simulation event definitions and priority ordering for Discrete Event Simulation."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class SimEventType(Enum):
    """Types of discrete simulation events."""
    JOB_ARRIVAL = auto()              # New inference job arrived at tenant queue
    GPU_DISPATCH = auto()             # Scheduler selects and launches job on GPU
    CONTEXT_SWITCH_END = auto()       # Context switch overhead elapsed, compute begins
    SLICE_COMPLETION = auto()         # Job finishes its allocated quantum slice
    JOB_COMPLETION = auto()           # Job has finished all remaining execution time
    JOB_PREEMPTION = auto()           # Job quantum expired at scheduling boundary
    GPU_IDLE_CHECK = auto()           # Verification of idle status
    QUOTA_REFILL = auto()             # Periodic sliding window quota refresh
    AGING_TICK = auto()               # Periodic aging evaluation trigger


@dataclass(order=True)
class SimEvent:
    """Discrete Event container for the simulation priority queue (min-heap)."""
    timestamp_ms: float
    priority: int = 10                # Tie-breaker (lower number = higher priority)
    event_id: int = field(default=0, compare=True)
    event_type: SimEventType = field(compare=False, default=SimEventType.JOB_ARRIVAL)
    payload: Any = field(compare=False, default=None)

    def __post_init__(self) -> None:
        if self.timestamp_ms < 0.0:
            raise ValueError(f"timestamp_ms must be non-negative, got {self.timestamp_ms}")
