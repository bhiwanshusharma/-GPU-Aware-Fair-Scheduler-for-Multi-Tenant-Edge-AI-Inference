"""Discrete Event GPU Simulation Engine modeling multi-tenant inference execution."""

import heapq
from typing import Dict, List, Optional, Tuple, Any
import logging

from src.core.types import JobState, PolicyType
from src.core.job import GPUJob
from src.core.tenant import Tenant
from src.core.queues import TenantQueueManager
from src.core.exceptions import (
    InvalidQuantumError,
    InvalidExecutionTimeError,
    DuplicateJobError,
    JobCancelledError,
)
from src.scheduler.base import BaseScheduler
from src.simulation.events import SimEvent, SimEventType
from src.simulation.gpu import SimulatedGPU, GPUState
from src.simulation.trace import SimulationTrace, TraceEventType

logger = logging.getLogger(__name__)


class GPUSimulator:
    """Event-driven simulator modeling a single shared GPU serving multiple AI tenants.
    
    Operates at application/job-dispatch boundary with configurable time quanta,
    context switch overheads, and non-blocking discrete event queue.
    """

    def __init__(
        self,
        tenants: Optional[Dict[str, Tenant]] = None,
        quantum_ms: float = 10.0,
        context_switch_overhead_ms: float = 0.0,
        policy: str = "round_robin",
        scheduler: Optional[BaseScheduler] = None,
        max_queue_snapshots: int = 5000,
    ) -> None:
        if quantum_ms <= 0.0:
            logger.error(f"Invalid quantum_ms: {quantum_ms}")
            raise InvalidQuantumError(f"quantum_ms must be strictly positive, got {quantum_ms}")
        if context_switch_overhead_ms < 0.0:
            raise ValueError(
                f"context_switch_overhead_ms must be non-negative, got {context_switch_overhead_ms}"
            )

        self.quantum_ms = float(quantum_ms)
        self.context_switch_overhead_ms = float(context_switch_overhead_ms)
        self.policy = policy.lower()
        self.scheduler = scheduler
        self.max_queue_snapshots = max_queue_snapshots

        # Hardware and Queues
        self.gpu = SimulatedGPU(context_switch_overhead_ms=self.context_switch_overhead_ms)
        if scheduler is not None:
            self.queue_manager = scheduler.queue_manager
            self.tenants = scheduler.tenants
            if hasattr(scheduler, "quantum") and scheduler.quantum:
                self.quantum_ms = scheduler.quantum.base_quantum_ms
            if tenants:
                for t in tenants.values():
                    self.register_tenant(t)
        else:
            self.queue_manager = TenantQueueManager()
            self.tenants = {}
            if tenants:
                for t in tenants.values():
                    self.register_tenant(t)

        # Simulation Event State
        self.current_time_ms: float = 0.0
        self._event_heap: List[SimEvent] = []
        self._event_counter: int = 0
        self._dispatch_pending: bool = False
        self.trace = SimulationTrace()
        
        # Round Robin cursor (-1 so first advance selects index 0)
        self._rr_tenant_index: int = -1
        self.completed_jobs: List[GPUJob] = []
        self._submitted_job_ids: set[str] = set()
        self._cancelled_job_ids: set[str] = set()

    def register_tenant(self, tenant: Tenant) -> None:
        """Registers a tenant and binds an isolated ingress queue."""
        self.tenants[tenant.tenant_id] = tenant
        self.queue_manager.register_tenant(tenant)

    def submit_job(self, job: GPUJob) -> None:
        """Schedules an inference job arrival event into the discrete simulation.
        
        Validates job execution time and registers tenant if necessary.
        """
        if job.status == JobState.CANCELLED:
            logger.error(f"Cannot submit cancelled job '{job.job_id}'")
            raise JobCancelledError(f"Cannot submit cancelled job '{job.job_id}'.")

        if job.job_id in self._submitted_job_ids:
            logger.error(f"Duplicate job rejection: '{job.job_id}' has already been submitted to simulation.")
            raise DuplicateJobError(
                f"Job with ID '{job.job_id}' has already been submitted to this simulation."
            )

        # Validate execution time
        if job.estimated_gpu_time <= 0.0 or job.remaining_gpu_time <= 0.0:
            logger.error(
                f"Invalid execution time for job '{job.job_id}': "
                f"estimated={job.estimated_gpu_time}, remaining={job.remaining_gpu_time}"
            )
            raise InvalidExecutionTimeError(
                f"Job '{job.job_id}' must have positive execution time, got "
                f"estimated={job.estimated_gpu_time}, remaining={job.remaining_gpu_time}"
            )

        self._submitted_job_ids.add(job.job_id)

        if job.tenant_id not in self.tenants:
            # Auto-register default tenant if not explicitly provided
            default_tenant = Tenant(
                tenant_id=job.tenant_id,
                priority=job.priority,
                weight=1.0,
                gpu_quota=100.0,
            )
            self.register_tenant(default_tenant)

        # JOB_ARRIVAL scheduled with priority 10 so arrivals at time t are processed before DISPATCH (priority 20)
        self._schedule_sim_event(
            timestamp_ms=job.arrival_time,
            event_type=SimEventType.JOB_ARRIVAL,
            priority=10,
            payload=job,
        )

    def cancel_job(self, job_id: str, reason: str = "User cancelled") -> bool:
        """Cancels a job by removing it from tenant queues or pending arrival events."""
        self._cancelled_job_ids.add(job_id)
        found = False

        # 1. Check active ingress queues
        for t_id in self.queue_manager.registered_tenant_ids:
            cancelled = self.queue_manager.cancel_job(t_id, job_id, reason, self.current_time_ms)
            if cancelled is not None:
                logger.info(f"Successfully cancelled job '{job_id}' from tenant '{t_id}' queue.")
                found = True

        # 2. Check pending arrival events in the simulation event heap
        for event in self._event_heap:
            if event.event_type == SimEventType.JOB_ARRIVAL and hasattr(event.payload, "job_id"):
                if event.payload.job_id == job_id:
                    event.payload.cancel(reason, self.current_time_ms)
                    found = True

        return found

    def _schedule_sim_event(
        self,
        timestamp_ms: float,
        event_type: SimEventType,
        priority: int = 10,
        payload: Any = None,
    ) -> None:
        """Pushes a discrete event onto the simulation priority min-heap."""
        self._event_counter += 1
        event = SimEvent(
            timestamp_ms=float(timestamp_ms),
            priority=priority,
            event_id=self._event_counter,
            event_type=event_type,
            payload=payload,
        )
        heapq.heappush(self._event_heap, event)

    def _request_dispatch(self) -> None:
        """Schedules a dispatch evaluation event at current_time_ms if not already scheduled."""
        if not self._dispatch_pending:
            self._dispatch_pending = True
            self._schedule_sim_event(
                timestamp_ms=self.current_time_ms,
                event_type=SimEventType.GPU_DISPATCH,
                priority=20,
            )

    def _select_next_job(self) -> Optional[GPUJob]:
        """Arbiter selecting the next eligible job from tenant queues."""
        if self.scheduler is not None:
            return self.scheduler.schedule(self.current_time_ms)

        active_tenant_ids = self.queue_manager.get_active_tenant_ids()
        if not active_tenant_ids:
            return None

        if self.policy == "priority":
            # Select from the tenant with highest effective priority
            active_tenants = [self.tenants[t_id] for t_id in active_tenant_ids]
            active_tenants.sort(key=lambda t: t.effective_priority, reverse=True)
            chosen_tenant = active_tenants[0]
            return self.queue_manager.dequeue(chosen_tenant.tenant_id)

        elif self.policy == "fifo":
            # Global FIFO: examine head of all queues, pick earliest arrival
            earliest_job: Optional[GPUJob] = None
            chosen_tenant_id: Optional[str] = None
            for t_id in active_tenant_ids:
                head = self.queue_manager.peek(t_id)
                if head is not None:
                    if earliest_job is None or head.arrival_time < earliest_job.arrival_time:
                        earliest_job = head
                        chosen_tenant_id = t_id
            if chosen_tenant_id:
                return self.queue_manager.dequeue(chosen_tenant_id)
            return None

        else:
            # Default: Round Robin across active tenant queues
            tenant_list = list(self.tenants.keys())
            n = len(tenant_list)
            for _ in range(n):
                self._rr_tenant_index = (self._rr_tenant_index + 1) % n
                candidate_id = tenant_list[self._rr_tenant_index]
                if not self.queue_manager.get_queue(candidate_id).is_empty:
                    return self.queue_manager.dequeue(candidate_id)

            # Fallback to any active queue
            return self.queue_manager.dequeue(active_tenant_ids[0])

    def _trigger_dispatch(self) -> None:
        """Attempts to select and dispatch an inference job onto the GPU."""
        if self.gpu.is_busy:
            # GPU is currently computing; cannot dispatch until active slice finishes
            return

        job = self._select_next_job()
        if job is None:
            # No eligible work: enter IDLE
            if not self.gpu.is_idle:
                self.gpu.enter_idle(self.current_time_ms)
                self.trace.record_event(
                    timestamp_ms=self.current_time_ms,
                    event_type=TraceEventType.IDLE_START,
                    message="GPU idle",
                )

            # If there are active queues whose tenants are throttled by quota,
            # schedule a wakeup event when the earliest quota window slides.
            active_ids = self.queue_manager.get_active_tenant_ids()
            if active_ids:
                wake_times = []
                for t_id in active_ids:
                    t = self.tenants.get(t_id)
                    if t and t.is_quota_exceeded(self.current_time_ms):
                        wake_times.append(t.window_start_time_ms + t.quota_window_ms)
                if wake_times:
                    next_wake = min(wake_times)
                    if next_wake > self.current_time_ms:
                        self._schedule_sim_event(
                            timestamp_ms=next_wake,
                            event_type=SimEventType.GPU_DISPATCH,
                            priority=20,
                        )
            return

        # We have a job to dispatch!
        if self.gpu.is_idle:
            idle_dur = self.gpu.exit_idle(self.current_time_ms)
            if idle_dur > 0:
                self.trace.record_idle_period(
                    start_ms=self.current_time_ms - idle_dur,
                    end_ms=self.current_time_ms,
                )
                self.trace.record_event(
                    timestamp_ms=self.current_time_ms,
                    event_type=TraceEventType.IDLE_END,
                    message="GPU resumed from idle",
                )

        if self.scheduler is not None and hasattr(self.scheduler, "get_allocated_quantum"):
            effective_quantum = self.scheduler.get_allocated_quantum(job)
        else:
            effective_quantum = self.quantum_ms

        switch_delay, compute_duration = self.gpu.calculate_dispatch_slice(job, effective_quantum)

        # Record dispatch or resume in trace
        is_first_start = (job.start_time is None)
        if is_first_start:
            job.mark_started(self.current_time_ms + switch_delay)
            self.trace.record_event(
                timestamp_ms=self.current_time_ms,
                event_type=TraceEventType.DISPATCH,
                tenant_id=job.tenant_id,
                job_id=job.job_id,
                message=f"tenant={job.tenant_id} job={job.job_id} dispatched",
                details={"allocated_quantum": compute_duration, "switch_delay": switch_delay},
            )
        else:
            if hasattr(job, "_last_preempt_time") and job._last_preempt_time is not None:
                job.accumulated_waiting_time += max(0.0, (self.current_time_ms + switch_delay) - job._last_preempt_time)
                job._last_preempt_time = None
            job.transition_to(JobState.RUNNING)
            self.trace.record_event(
                timestamp_ms=self.current_time_ms,
                event_type=TraceEventType.RESUME,
                tenant_id=job.tenant_id,
                job_id=job.job_id,
                message=f"tenant={job.tenant_id} job={job.job_id} resumed",
                details={"allocated_quantum": compute_duration, "switch_delay": switch_delay},
            )

        # Transition GPU to active execution
        self.gpu.start_execution_slice(
            job=job,
            switch_delay_ms=switch_delay,
            compute_duration_ms=compute_duration,
            current_time_ms=self.current_time_ms,
        )

        # Schedule the completion of this execution slice
        slice_end_time = self.current_time_ms + switch_delay + compute_duration
        self._schedule_sim_event(
            timestamp_ms=slice_end_time,
            event_type=SimEventType.SLICE_COMPLETION,
            priority=5,
            payload={
                "job": job,
                "compute_duration": compute_duration,
                "switch_delay": switch_delay,
            },
        )

    def _handle_job_arrival(self, job: GPUJob) -> None:
        """Processes a new job arrival event."""
        if job.job_id in self._cancelled_job_ids or job.status == JobState.CANCELLED:
            logger.info(f"Skipping arrival of cancelled job '{job.job_id}'")
            return

        # Enqueue job into tenant queue
        self.queue_manager.enqueue(job)
        tenant = self.tenants[job.tenant_id]
        if not tenant.has_job(job.job_id):
            tenant.add_job(job)

        # Record queue snapshot
        if len(self.trace.queue_snapshots) < self.max_queue_snapshots:
            queue_sizes = {t_id: q.size for t_id, q in self.queue_manager._queues.items()}
            self.trace.record_queue_snapshot(self.current_time_ms, queue_sizes)

        # If GPU is IDLE, trigger dispatch evaluation
        if self.gpu.is_idle:
            self._request_dispatch()

    def _handle_slice_completion(self, payload: Dict[str, Any]) -> None:
        """Processes the end of an allocated execution slice on the GPU."""
        job: GPUJob = payload["job"]
        compute_duration: float = payload["compute_duration"]

        # Deduct executed compute duration
        job.accumulated_execution_time += compute_duration
        job.remaining_gpu_time = max(0.0, job.remaining_gpu_time - compute_duration)

        # Record tenant usage
        tenant = self.tenants[job.tenant_id]
        tenant.record_gpu_execution(compute_duration, self.current_time_ms)

        # Check completion vs preemption
        if job.remaining_gpu_time <= 0.0:
            # Job fully completed!
            job.mark_completed(self.current_time_ms)
            tenant.record_job_completed(job)
            self.completed_jobs.append(job)

            self.trace.record_event(
                timestamp_ms=self.current_time_ms,
                event_type=TraceEventType.COMPLETION,
                tenant_id=job.tenant_id,
                job_id=job.job_id,
                message=f"tenant={job.tenant_id} job={job.job_id} completed",
                details={
                    "waiting_time": job.waiting_time_ms,
                    "turnaround_time": job.turnaround_time_ms,
                },
            )
            if self.scheduler is not None:
                self.scheduler.on_job_completed(job, compute_duration, self.current_time_ms)
        else:
            # Quantum expired at scheduling boundary -> preemption
            job.mark_preempted(self.current_time_ms)
            self.trace.record_event(
                timestamp_ms=self.current_time_ms,
                event_type=TraceEventType.QUANTUM_EXPIRED,
                tenant_id=job.tenant_id,
                job_id=job.job_id,
                message=f"{job.job_id} quantum expired",
                details={"remaining_gpu_time": job.remaining_gpu_time},
            )
            # Re-enqueue at the tail of the tenant queue
            self.queue_manager.enqueue(job)
            if self.scheduler is not None and hasattr(self.scheduler, "on_job_preempted"):
                self.scheduler.on_job_preempted(job, self.current_time_ms)

        # GPU finishes slice and returns to IDLE state
        self.gpu.complete_slice(self.current_time_ms)

        # Snapshot queue states
        if len(self.trace.queue_snapshots) < self.max_queue_snapshots:
            queue_sizes = {t_id: q.size for t_id, q in self.queue_manager._queues.items()}
            self.trace.record_queue_snapshot(self.current_time_ms, queue_sizes)

        # Request next dispatch immediately at current_time
        self._request_dispatch()

    def step(self) -> bool:
        """Processes the next discrete event in the simulation.
        
        Returns:
            True if an event was processed, False if event queue is empty.
        """
        if not self._event_heap:
            return False

        event = heapq.heappop(self._event_heap)
        self.current_time_ms = event.timestamp_ms

        if event.event_type == SimEventType.JOB_ARRIVAL:
            self._handle_job_arrival(event.payload)
        elif event.event_type == SimEventType.SLICE_COMPLETION:
            self._handle_slice_completion(event.payload)
        elif event.event_type == SimEventType.GPU_DISPATCH:
            self._dispatch_pending = False
            self._trigger_dispatch()

        return True

    def run(self, until_time_ms: Optional[float] = None) -> None:
        """Executes the event-driven simulation loop until completion or time limit."""
        while self._event_heap:
            next_event_time = self._event_heap[0].timestamp_ms
            if until_time_ms is not None and next_event_time > until_time_ms:
                self.current_time_ms = until_time_ms
                break
            self.step()

        # If simulation ended and GPU was idle, record final idle period
        if self.gpu.is_idle and self.current_time_ms > self.gpu.last_state_change_time:
            idle_dur = self.current_time_ms - self.gpu.last_state_change_time
            if idle_dur > 0:
                self.trace.record_idle_period(
                    start_ms=self.gpu.last_state_change_time,
                    end_ms=self.current_time_ms,
                )

    @property
    def gpu_utilization_pct(self) -> float:
        """Computes system GPU compute utilization percentage."""
        return self.gpu.compute_utilization_pct(self.current_time_ms)

    def get_queue_states(self) -> List[Tuple[float, Dict[str, int]]]:
        """Returns recorded queue size snapshots."""
        return list(self.trace.queue_snapshots)

    def get_idle_periods(self) -> List[Tuple[float, float, float]]:
        """Returns list of (start_ms, end_ms, duration_ms) idle periods."""
        return list(self.trace.idle_periods)

    def get_execution_trace_text(self, include_idle: bool = False) -> str:
        """Returns the formatted execution trace string."""
        return self.trace.format_trace(include_idle=include_idle)
