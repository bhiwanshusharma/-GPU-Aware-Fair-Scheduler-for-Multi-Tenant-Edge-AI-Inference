"""GPU execution trace record tracking atomic kernel/job execution bursts."""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class GPUExecutionRecord:
    """Records telemetry from an executed inference burst or slice on the GPU."""
    record_id: str
    job_id: str
    tenant_id: str
    model_name: str
    start_time: float
    end_time: float
    allocated_quantum: float
    actual_execution_time: float
    remaining_gpu_time_after: float
    was_preempted: bool
    is_completed: bool
    vram_allocated_mb: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.record_id or not isinstance(self.record_id, str):
            raise ValueError(f"record_id must be a non-empty string, got '{self.record_id}'")
        if not self.job_id or not isinstance(self.job_id, str):
            raise ValueError(f"job_id must be a non-empty string, got '{self.job_id}'")
        if not self.tenant_id or not isinstance(self.tenant_id, str):
            raise ValueError(f"tenant_id must be a non-empty string, got '{self.tenant_id}'")
        if not self.model_name or not isinstance(self.model_name, str):
            raise ValueError(f"model_name must be a non-empty string, got '{self.model_name}'")

        if self.start_time < 0.0:
            raise ValueError(f"start_time must be non-negative, got {self.start_time}")
        if self.end_time < self.start_time:
            raise ValueError(
                f"end_time ({self.end_time}) cannot be earlier than start_time ({self.start_time})"
            )

        if self.allocated_quantum < 0.0:
            raise ValueError(f"allocated_quantum must be non-negative, got {self.allocated_quantum}")
        if self.actual_execution_time < 0.0:
            raise ValueError(
                f"actual_execution_time must be non-negative, got {self.actual_execution_time}"
            )
        if self.remaining_gpu_time_after < 0.0:
            raise ValueError(
                f"remaining_gpu_time_after must be non-negative, got {self.remaining_gpu_time_after}"
            )
        if self.vram_allocated_mb < 0.0:
            raise ValueError(
                f"vram_allocated_mb must be non-negative, got {self.vram_allocated_mb}"
            )

    @property
    def elapsed_wall_time(self) -> float:
        """Wall-clock or virtual clock duration of the execution slice."""
        return self.end_time - self.start_time
