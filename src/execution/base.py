"""Hardware Executor Interface for GPU Workload Dispatch.

ARCHITECTURAL PRINCIPLE:
========================
This system implements APPLICATION-LEVEL GPU WORKLOAD DISPATCH.
It does NOT claim that user-space software can arbitrarily perform
hardware-level GPU context preemption.

Distinction:
1. Hardware GPU Context Preemption:
   - Handled exclusively by GPU hardware schedulers and privileged kernel drivers
     (e.g., nvgpu / nvidia.ko) at compute instruction or thread-block boundaries.
   - User-space code cannot asynchronously pause an active GPU kernel running on an SM.

2. Software / Application-Level GPU Scheduling (Our Approach):
   - The scheduler acts as an application-level dispatch arbiter.
   - Workloads are partitioned and dispatched in BOUNDED EXECUTION UNITS
     (individual inference requests, micro-batches, batches, or chunked model layers).
   - The scheduler decides WHICH tenant gets GPU service next.
   - The executor handles HOW that selected bounded unit is launched and timed.
   - Slicing and preemption occur at these bounded dispatch boundaries.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional

from src.core.job import GPUJob, Job


class BoundedUnitType(str, Enum):
    """Granularity of bounded execution units dispatched to the GPU."""
    INFERENCE_REQUEST = "inference_request"  # Single input inference call
    MICRO_BATCH = "micro_batch"              # Small sub-batch chunk (e.g. batch_size=1 or 2)
    BATCH = "batch"                          # Standard inference batch
    KERNEL_CHUNK = "kernel_chunk"            # Partitioned layer/kernel chunk


@dataclass
class HardwareTelemetry:
    """Instantaneous hardware status reported by the execution backend."""
    hardware_available: bool
    backend_type: str  # "simulation" | "jetson_cuda" | "tensorrt"
    device_name: str = "Simulated GPU"
    gpu_utilization_pct: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    temperature_celsius: Optional[float] = None
    power_watts: Optional[float] = None
    cuda_driver_version: Optional[str] = None
    cuda_device_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result and performance telemetry returned after dispatching a bounded execution slice."""
    job_id: str
    tenant_id: str
    actual_gpu_execution_time_ms: float
    inference_latency_ms: float
    gpu_utilization_pct: float
    memory_used_bytes: int
    memory_total_bytes: int
    scheduling_overhead_ms: float
    is_completed: bool
    remaining_gpu_time_ms: float
    bounded_unit_type: BoundedUnitType
    metadata: Dict[str, Any] = field(default_factory=dict)


class HardwareExecutor(ABC):
    """Abstract interface defining the contract for both simulated and physical GPU execution backends."""

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the underlying physical or virtual backend runtime is functional."""
        pass

    @abstractmethod
    def dispatch_slice(
        self,
        job: GPUJob,
        quantum_slice_ms: float,
        bounded_unit: BoundedUnitType = BoundedUnitType.MICRO_BATCH,
    ) -> ExecutionResult:
        """Launches a bounded execution unit for the selected job on the target GPU runtime.
        
        Args:
            job: The GPUJob selected for service by the scheduler arbiter.
            quantum_slice_ms: Maximum time slice granted by the scheduler policy.
            bounded_unit: The execution granularity (request, micro-batch, chunk).
            
        Returns:
            ExecutionResult containing execution timing and hardware telemetry.
        """
        pass

    @abstractmethod
    def get_hardware_telemetry(self) -> HardwareTelemetry:
        """Queries instantaneous hardware metrics (temperature, VRAM, utilization)."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets executor state."""
        pass

    def execute(self, job: Job) -> float:
        """Legacy helper executing an entire job or slice, returning duration in ms."""
        res = self.dispatch_slice(job, quantum_slice_ms=job.remaining_gpu_time)
        return res.actual_gpu_execution_time_ms


# Backward compatibility alias
IGPUExecutor = HardwareExecutor
