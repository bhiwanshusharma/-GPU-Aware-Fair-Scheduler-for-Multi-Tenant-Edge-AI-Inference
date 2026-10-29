"""Fundamental enumerations and data structures for the GPU-Aware Fair Scheduler."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, Tuple


import math
from src.core.exceptions import InvalidQuantumError, InvalidPredictorOutputError


class WorkloadClass(str, Enum):
    """Execution profile category predicted or profiled for an AI workload."""
    GPU_INTENSIVE = "GPU_INTENSIVE"   # Heavy matrix multiplication / tensor operations
    BALANCED = "BALANCED"             # Moderate compute and memory footprint
    MIXED = "MIXED"                   # Mixed compute, host and memory characteristics
    CPU_BOUND = "CPU_BOUND"           # High host pre/post-processing, minimal GPU time
    IO_BOUND = "IO_BOUND"             # Sensor streaming, payload transfer limited


class JobState(str, Enum):
    """Lifecycle states of an inference job within the scheduling pipeline."""
    QUEUED = "QUEUED"                 # Enqueued in tenant ingress queue
    PREDICTED = "PREDICTED"           # Profiled and predicted by AI predictor
    DISPATCHED = "DISPATCHED"         # Selected by scheduler, awaiting GPU execution
    RUNNING = "RUNNING"               # Actively computing on GPU
    PREEMPTED = "PREEMPTED"           # Temporarily suspended at quantum boundary
    COMPLETED = "COMPLETED"           # Execution finished, telemetry recorded
    DROPPED = "DROPPED"               # Rejected due to queue capacity overflow
    CANCELLED = "CANCELLED"           # Cancelled before or during execution


class PolicyType(str, Enum):
    """Supported scheduling algorithm policies."""
    FIFO = "fifo"
    ROUND_ROBIN = "round_robin"
    STRICT_PRIORITY = "strict_priority"
    PDWFQ = "pdwfq"


@dataclass(slots=True)
class SchedulingQuantum:
    """Configurable quantum slicing parameters."""
    base_quantum_ms: float = 25.0
    min_quantum_ms: float = 5.0
    max_quantum_ms: float = 100.0
    aging_alpha: float = 0.15
    starvation_threshold_ms: float = 150.0

    def __post_init__(self) -> None:
        if self.min_quantum_ms <= 0.0 or math.isnan(self.min_quantum_ms):
            raise InvalidQuantumError(f"min_quantum_ms must be positive, got {self.min_quantum_ms}")
        if self.base_quantum_ms <= 0.0 or math.isnan(self.base_quantum_ms):
            raise InvalidQuantumError(f"base_quantum_ms must be positive, got {self.base_quantum_ms}")
        if self.max_quantum_ms <= 0.0 or math.isnan(self.max_quantum_ms):
            raise InvalidQuantumError(f"max_quantum_ms must be positive, got {self.max_quantum_ms}")
        if self.min_quantum_ms > self.base_quantum_ms:
            raise InvalidQuantumError(
                f"min_quantum_ms ({self.min_quantum_ms}) cannot exceed base_quantum_ms ({self.base_quantum_ms})"
            )
        if self.base_quantum_ms > self.max_quantum_ms:
            raise InvalidQuantumError(
                f"base_quantum_ms ({self.base_quantum_ms}) cannot exceed max_quantum_ms ({self.max_quantum_ms})"
            )
        if self.aging_alpha < 0.0 or math.isnan(self.aging_alpha):
            raise ValueError(f"aging_alpha ({self.aging_alpha}) must be non-negative")
        if self.starvation_threshold_ms <= 0.0 or math.isnan(self.starvation_threshold_ms):
            raise ValueError(f"starvation_threshold_ms must be positive, got {self.starvation_threshold_ms}")


@dataclass(slots=True)
class PredictionOutput:
    """Estimated runtime characteristics produced by the AI Workload Predictor."""
    estimated_latency_ms: float
    estimated_memory_mb: float
    classification: WorkloadClass
    confidence_score: float = 1.0

    def __post_init__(self) -> None:
        if math.isnan(self.estimated_latency_ms) or math.isinf(self.estimated_latency_ms) or self.estimated_latency_ms < 0.0:
            raise InvalidPredictorOutputError(
                f"estimated_latency_ms must be a non-negative finite number, got {self.estimated_latency_ms}"
            )
        if math.isnan(self.estimated_memory_mb) or math.isinf(self.estimated_memory_mb) or self.estimated_memory_mb < 0.0:
            raise InvalidPredictorOutputError(
                f"estimated_memory_mb must be a non-negative finite number, got {self.estimated_memory_mb}"
            )
        if math.isnan(self.confidence_score) or not (0.0 <= self.confidence_score <= 1.0):
            raise InvalidPredictorOutputError(
                f"confidence_score must be in [0.0, 1.0], got {self.confidence_score}"
            )
        if not isinstance(self.classification, WorkloadClass):
            try:
                object.__setattr__(self, "classification", WorkloadClass(str(self.classification)))
            except ValueError:
                raise InvalidPredictorOutputError(
                    f"Unknown classification category: {self.classification}"
                )


@dataclass(slots=True)
class JobCompletionReport:
    """Telemetry report emitted upon completion of an inference job."""
    job_id: str
    tenant_id: str
    model_name: str
    arrival_time_ms: float
    start_time_ms: float
    completion_time_ms: float
    actual_gpu_time_ms: float
    waiting_time_ms: float
    turnaround_time_ms: float
    sla_deadline_ms: Optional[float]
    sla_violated: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
