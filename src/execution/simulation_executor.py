"""Simulation-Backed Hardware Executor.

Implements application-level GPU workload dispatch in discrete virtual simulation.
Always functional without requiring physical NVIDIA CUDA or Jetson hardware.
"""

import time
import logging
from typing import Dict, Any, Optional

from src.core.types import JobState
from src.core.job import GPUJob
from src.core.exceptions import (
    InvalidQuantumError,
    InvalidExecutionTimeError,
    JobCancelledError,
    GPUMemoryAllocationError,
)
from src.execution.base import (
    HardwareExecutor,
    ExecutionResult,
    HardwareTelemetry,
    BoundedUnitType,
)

logger = logging.getLogger(__name__)


class SimulationExecutor(HardwareExecutor):
    """Zero-dependency simulation backend executing bounded GPU slices."""

    def __init__(
        self,
        context_switch_overhead_ms: float = 0.5,
        total_memory_mb: float = 8192.0,
        device_name: str = "Simulated Edge GPU (Jetson Orin UMA Emulation)",
    ) -> None:
        self.context_switch_overhead_ms = float(context_switch_overhead_ms)
        self.total_memory_mb = float(total_memory_mb)
        self.device_name = device_name
        self._current_memory_allocated_mb: float = 0.0
        self._total_executed_ms: float = 0.0
        self._dispatches_count: int = 0

    def is_available(self) -> bool:
        """Simulation backend is always available."""
        return True

    def dispatch_slice(
        self,
        job: GPUJob,
        quantum_slice_ms: float,
        bounded_unit: BoundedUnitType = BoundedUnitType.MICRO_BATCH,
    ) -> ExecutionResult:
        """Launches a bounded execution unit for the selected job in simulation mode.
        
        Args:
            job: GPUJob to execute.
            quantum_slice_ms: Permitted time quantum.
            bounded_unit: Bounded unit granularity.
            
        Returns:
            ExecutionResult with simulated telemetry.
        """
        if quantum_slice_ms <= 0.0:
            logger.error(f"Invalid quantum slice: {quantum_slice_ms} ms")
            raise InvalidQuantumError(f"quantum_slice_ms must be positive, got {quantum_slice_ms}")

        if job.status == JobState.CANCELLED:
            logger.error(f"Cannot dispatch cancelled job '{job.job_id}'")
            raise JobCancelledError(f"Cannot dispatch cancelled job '{job.job_id}'.")

        if job.remaining_gpu_time <= 0.0:
            logger.error(f"Job '{job.job_id}' has non-positive remaining time: {job.remaining_gpu_time}")
            raise InvalidExecutionTimeError(f"remaining_gpu_time must be positive, got {job.remaining_gpu_time}")

        # Check VRAM limits
        requested_vram_mb = 350.0 if "nano" in job.model_name else (800.0 if "resnet" in job.model_name else 1400.0)
        requested_vram_mb *= (job.batch_size or 1)
        if requested_vram_mb > self.total_memory_mb:
            logger.error(f"VRAM allocation failure: requested {requested_vram_mb}MB exceeds total {self.total_memory_mb}MB")
            raise GPUMemoryAllocationError(
                f"Simulated VRAM allocation failed for job '{job.job_id}': "
                f"requested {requested_vram_mb:.1f} MB exceeds device memory limit {self.total_memory_mb:.1f} MB."
            )

        t_launch_start = time.perf_counter()

        # Bounded slice execution: min of remaining time and quantum slice
        planned_slice_duration = min(job.remaining_gpu_time, quantum_slice_ms)
        actual_gpu_time = planned_slice_duration

        # Update remaining time and completion state
        rem_before = job.remaining_gpu_time
        rem_after = max(0.0, rem_before - actual_gpu_time)
        job.remaining_gpu_time = rem_after
        job.accumulated_execution_time += actual_gpu_time
        is_completed = (rem_after <= 1e-6)

        self._current_memory_allocated_mb = min(self.total_memory_mb * 0.85, requested_vram_mb)

        self._total_executed_ms += actual_gpu_time
        self._dispatches_count += 1

        t_launch_end = time.perf_counter()
        scheduling_overhead = (t_launch_end - t_launch_start) * 1000.0 + self.context_switch_overhead_ms

        return ExecutionResult(
            job_id=job.job_id,
            tenant_id=job.tenant_id,
            actual_gpu_execution_time_ms=round(actual_gpu_time, 4),
            inference_latency_ms=round(actual_gpu_time + scheduling_overhead, 4),
            gpu_utilization_pct=95.0,
            memory_used_bytes=int(self._current_memory_allocated_mb * 1024 * 1024),
            memory_total_bytes=int(self.total_memory_mb * 1024 * 1024),
            scheduling_overhead_ms=round(scheduling_overhead, 4),
            is_completed=is_completed,
            remaining_gpu_time_ms=round(rem_after, 4),
            bounded_unit_type=bounded_unit,
            metadata={
                "backend": "simulation",
                "simulated_device": self.device_name,
                "dispatches_count": self._dispatches_count,
            },
        )

    def get_hardware_telemetry(self) -> HardwareTelemetry:
        """Returns synthetic edge GPU telemetry."""
        return HardwareTelemetry(
            hardware_available=True,
            backend_type="simulation",
            device_name=self.device_name,
            gpu_utilization_pct=92.5,
            memory_used_mb=round(self._current_memory_allocated_mb, 1),
            memory_total_mb=self.total_memory_mb,
            temperature_celsius=48.5,
            power_watts=12.4,
            cuda_driver_version="Simulation-DES",
            cuda_device_count=1,
            metadata={"simulated": True},
        )

    def reset(self) -> None:
        """Resets simulation metrics."""
        self._current_memory_allocated_mb = 0.0
        self._total_executed_ms = 0.0
        self._dispatches_count = 0
