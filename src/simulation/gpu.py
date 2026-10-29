"""Simulated edge GPU hardware model tracking execution, switching overhead, and utilization."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, Dict, Any
from src.core.job import GPUJob


class GPUState(Enum):
    """Operational states of the simulated edge GPU engine."""
    IDLE = auto()          # No active job dispatched; GPU awaiting requests
    SWITCHING = auto()     # Context/job switching overhead elapsing (DMA, setup, kernel queueing)
    BUSY = auto()          # Actively computing inference kernels on Tensor/CUDA cores


@dataclass
class SimulatedGPU:
    """Models a single shared edge GPU executing inference jobs under cooperative dispatch."""
    context_switch_overhead_ms: float = 0.5  # Job dispatch/context switch delay
    
    # Live execution state
    state: GPUState = GPUState.IDLE
    active_job: Optional[GPUJob] = None
    last_tenant_id: Optional[str] = None
    last_job_id: Optional[str] = None
    
    # Internal timers
    last_state_change_time: float = 0.0
    current_slice_start_time: Optional[float] = None
    current_slice_duration: float = 0.0
    
    # Cumulative hardware duty cycle statistics
    total_busy_time_ms: float = 0.0
    total_idle_time_ms: float = 0.0
    total_switching_time_ms: float = 0.0
    total_context_switches: int = 0
    total_slices_executed: int = 0

    def __post_init__(self) -> None:
        if self.context_switch_overhead_ms < 0.0:
            raise ValueError(
                f"context_switch_overhead_ms must be non-negative, got {self.context_switch_overhead_ms}"
            )

    @property
    def is_idle(self) -> bool:
        return self.state == GPUState.IDLE

    @property
    def is_busy(self) -> bool:
        return self.state in (GPUState.BUSY, GPUState.SWITCHING)

    def calculate_dispatch_slice(
        self,
        job: GPUJob,
        quantum_ms: float,
    ) -> Tuple[float, float]:
        """Calculates switching overhead and active compute duration for a dispatch.
        
        Returns:
            (switching_delay_ms, compute_duration_ms)
        """
        if quantum_ms <= 0.0:
            raise ValueError(f"quantum_ms must be positive, got {quantum_ms}")
        if job.remaining_gpu_time <= 0.0:
            raise ValueError(
                f"Job '{job.job_id}' has non-positive remaining_gpu_time: {job.remaining_gpu_time}"
            )

        # Context switch overhead applies when switching from a different job or from IDLE
        is_switching = (self.last_job_id is not None and self.last_job_id != job.job_id)
        switch_delay = self.context_switch_overhead_ms if is_switching else 0.0

        compute_duration = min(job.remaining_gpu_time, quantum_ms)
        return (switch_delay, compute_duration)

    def start_execution_slice(
        self,
        job: GPUJob,
        switch_delay_ms: float,
        compute_duration_ms: float,
        current_time_ms: float,
    ) -> None:
        """Transitions GPU into active execution of a job slice."""
        if self.state == GPUState.IDLE and current_time_ms > self.last_state_change_time:
            idle_delta = current_time_ms - self.last_state_change_time
            self.total_idle_time_ms += idle_delta

        if switch_delay_ms > 0.0:
            self.state = GPUState.SWITCHING
            self.total_context_switches += 1
            self.total_switching_time_ms += switch_delay_ms
        else:
            self.state = GPUState.BUSY

        self.active_job = job
        self.last_state_change_time = current_time_ms
        self.current_slice_start_time = current_time_ms + switch_delay_ms
        self.current_slice_duration = compute_duration_ms
        self.total_busy_time_ms += compute_duration_ms
        self.total_slices_executed += 1

    def complete_slice(self, current_time_ms: float) -> None:
        """Completes the active slice, recording state references."""
        if self.active_job is not None:
            self.last_job_id = self.active_job.job_id
            self.last_tenant_id = self.active_job.tenant_id

        self.state = GPUState.IDLE
        self.active_job = None
        self.current_slice_start_time = None
        self.current_slice_duration = 0.0
        self.last_state_change_time = current_time_ms

    def enter_idle(self, current_time_ms: float) -> None:
        """Transitions GPU to idle state."""
        self.state = GPUState.IDLE
        self.active_job = None
        self.last_state_change_time = current_time_ms

    def exit_idle(self, current_time_ms: float) -> float:
        """Exits idle state and returns the elapsed idle duration."""
        idle_duration = max(0.0, current_time_ms - self.last_state_change_time)
        if self.state == GPUState.IDLE:
            self.total_idle_time_ms += idle_duration
        self.last_state_change_time = current_time_ms
        return idle_duration

    def compute_utilization_pct(self, total_simulation_time_ms: float) -> float:
        """Computes true active GPU compute utilization percentage.
        
        U = (Total Active Compute Time / Total Simulation Time) * 100%
        """
        if total_simulation_time_ms <= 0.0:
            return 0.0
        pct = (self.total_busy_time_ms / total_simulation_time_ms) * 100.0
        return min(100.0, max(0.0, pct))

    def reset(self) -> None:
        """Resets GPU state and counters."""
        self.state = GPUState.IDLE
        self.active_job = None
        self.last_tenant_id = None
        self.last_job_id = None
        self.last_state_change_time = 0.0
        self.current_slice_start_time = None
        self.current_slice_duration = 0.0
        self.total_busy_time_ms = 0.0
        self.total_idle_time_ms = 0.0
        self.total_switching_time_ms = 0.0
        self.total_context_switches = 0
        self.total_slices_executed = 0
