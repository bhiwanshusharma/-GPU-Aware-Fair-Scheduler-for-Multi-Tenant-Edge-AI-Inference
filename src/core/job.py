"""Inference job representation and lifecycle tracking for GPU-Aware Fair Scheduler."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, Union
import logging
from src.core.types import JobState, WorkloadClass, JobCompletionReport
from src.core.exceptions import InvalidExecutionTimeError, JobCancelledError

logger = logging.getLogger(__name__)


@dataclass
class GPUJob:
    """Represents an atomic AI inference request submitted by a tenant for GPU execution."""
    job_id: str
    tenant_id: str
    model_name: str
    arrival_time: float
    estimated_gpu_time: float
    remaining_gpu_time: float
    priority: int = 1
    deadline: Optional[float] = None
    workload_type: WorkloadClass = WorkloadClass.BALANCED
    batch_size: int = 1
    input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    status: JobState = JobState.QUEUED
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    accumulated_waiting_time: float = 0.0
    accumulated_execution_time: float = 0.0
    number_of_preemptions: int = 0
    predicted_memory_mb: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        job_id: str,
        tenant_id: str,
        model_name: str,
        *args: Any,
        arrival_time: Optional[float] = None,
        estimated_gpu_time: Optional[float] = None,
        remaining_gpu_time: Optional[float] = None,
        priority: int = 1,
        deadline: Optional[float] = None,
        workload_type: Optional[Union[WorkloadClass, str]] = None,
        batch_size: Optional[int] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        status: Optional[JobState] = None,
        start_time: Optional[float] = None,
        completion_time: Optional[float] = None,
        accumulated_waiting_time: float = 0.0,
        accumulated_execution_time: float = 0.0,
        number_of_preemptions: int = 0,
        predicted_memory_mb: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        # Backward-compatible parameter aliases
        arrival_time_ms: Optional[float] = None,
        deadline_ms: Optional[float] = None,
        predicted_execution_time_ms: Optional[float] = None,
        actual_gpu_time_ms: Optional[float] = None,
        state: Optional[JobState] = None,
        workload_class: Optional[Union[WorkloadClass, str]] = None,
        **kwargs: Any,
    ) -> None:
        # Validate job_id
        if not job_id or not isinstance(job_id, str) or not job_id.strip():
            raise ValueError(f"job_id must be a non-empty string, got '{job_id}'")
        self.job_id = job_id.strip()

        # Validate tenant_id
        if not tenant_id or not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError(f"tenant_id must be a non-empty string, got '{tenant_id}'")
        self.tenant_id = tenant_id.strip()

        # Validate model_name
        if not model_name or not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(f"model_name must be a non-empty string, got '{model_name}'")
        self.model_name = model_name.strip()

        # Handle positional args flexibility
        # Case A: Phase 1 legacy signature -> Job(job_id, tenant_id, model_name, input_shape, batch_size, arrival_time_ms)
        # Case B: Phase 2 signature -> GPUJob(job_id, tenant_id, model_name, arrival_time, estimated_gpu_time, remaining_gpu_time, priority)
        pos_arrival = arrival_time
        pos_est = estimated_gpu_time
        pos_rem = remaining_gpu_time
        pos_prio = priority
        pos_shape = input_shape
        pos_batch = batch_size

        if len(args) > 0:
            first_extra = args[0]
            if isinstance(first_extra, (tuple, list)):
                # Case A
                pos_shape = tuple(first_extra)
                if len(args) > 1:
                    pos_batch = int(args[1])
                if len(args) > 2:
                    pos_arrival = float(args[2])
            else:
                # Case B
                pos_arrival = float(first_extra)
                if len(args) > 1:
                    pos_est = float(args[1])
                if len(args) > 2:
                    pos_rem = float(args[2])
                if len(args) > 3:
                    pos_prio = int(args[3])

        # Resolve arrival_time
        arr = pos_arrival if pos_arrival is not None else arrival_time_ms
        if arr is None or arr < 0.0:
            raise ValueError(f"arrival_time must be non-negative, got {arr}")
        self.arrival_time = float(arr)

        # Resolve estimated_gpu_time
        est = pos_est if pos_est is not None else predicted_execution_time_ms
        if est is None:
            est = 0.0
        elif est < 0.0:
            logger.error(f"Job creation error: estimated_gpu_time must be non-negative, got {est}")
            raise InvalidExecutionTimeError(f"estimated_gpu_time must be non-negative, got {est}")
        self.estimated_gpu_time = float(est)

        # Resolve remaining_gpu_time
        if pos_rem is not None:
            if pos_rem < 0.0:
                logger.error(f"Job creation error: remaining_gpu_time must be non-negative, got {pos_rem}")
                raise InvalidExecutionTimeError(f"remaining_gpu_time must be non-negative, got {pos_rem}")
            self.remaining_gpu_time = float(pos_rem)
        else:
            self.remaining_gpu_time = float(self.estimated_gpu_time)

        # Validate priority
        prio = pos_prio
        if not isinstance(prio, int) or not (1 <= prio <= 10):
            raise ValueError(f"priority must be an integer between 1 and 10, got {prio}")
        self.priority = prio

        # Resolve deadline
        dl = deadline if deadline is not None else deadline_ms
        if dl is not None:
            if dl < self.arrival_time:
                raise ValueError(
                    f"deadline ({dl}) cannot be earlier than arrival_time ({self.arrival_time})"
                )
            self.deadline = float(dl)
        else:
            self.deadline = None

        # Resolve workload_type
        wt = workload_class if workload_class is not None else workload_type
        if wt is None:
            wt = WorkloadClass.BALANCED

        if isinstance(wt, str):
            try:
                self.workload_type = WorkloadClass(wt)
            except ValueError:
                raise ValueError(
                    f"Invalid workload_type '{wt}', must be one of {[w.value for w in WorkloadClass]}"
                )
        elif isinstance(wt, WorkloadClass):
            self.workload_type = wt
        else:
            raise TypeError(f"workload_type must be WorkloadClass or str, got {type(wt)}")

        # Validate batch_size
        b = pos_batch if pos_batch is not None else 1
        if not isinstance(b, int) or b <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {b}")
        self.batch_size = b

        # Input shape
        self.input_shape = tuple(pos_shape) if pos_shape is not None else (1, 3, 224, 224)

        # Status
        st = state if state is not None else status
        self.status = st if st is not None else JobState.QUEUED

        # Lifecycle timestamps & counters
        self.start_time = float(start_time) if start_time is not None else None
        self.completion_time = float(completion_time) if completion_time is not None else None
        
        if accumulated_waiting_time < 0.0:
            raise ValueError(f"accumulated_waiting_time must be non-negative, got {accumulated_waiting_time}")
        self.accumulated_waiting_time = float(accumulated_waiting_time)

        act = actual_gpu_time_ms if actual_gpu_time_ms is not None else accumulated_execution_time
        if act < 0.0:
            raise ValueError(f"accumulated_execution_time must be non-negative, got {act}")
        self.accumulated_execution_time = float(act)

        if number_of_preemptions < 0:
            raise ValueError(f"number_of_preemptions must be non-negative, got {number_of_preemptions}")
        self.number_of_preemptions = int(number_of_preemptions)

        if predicted_memory_mb < 0.0:
            raise ValueError(f"predicted_memory_mb must be non-negative, got {predicted_memory_mb}")
        self.predicted_memory_mb = float(predicted_memory_mb)

        self.metadata = dict(metadata) if metadata is not None else {}
        self._last_preempt_time: Optional[float] = None

    def transition_to(self, new_state: JobState) -> None:
        """Validates and executes lifecycle state transition."""
        if self.status == JobState.CANCELLED and new_state != JobState.CANCELLED:
            logger.error(f"Cannot transition cancelled job '{self.job_id}' to {new_state}")
            raise JobCancelledError(
                f"Job '{self.job_id}' is CANCELLED and cannot transition to {new_state}."
            )

        valid_transitions = {
            JobState.QUEUED: {JobState.PREDICTED, JobState.DISPATCHED, JobState.RUNNING, JobState.DROPPED, JobState.CANCELLED},
            JobState.PREDICTED: {JobState.DISPATCHED, JobState.RUNNING, JobState.DROPPED, JobState.CANCELLED},
            JobState.DISPATCHED: {JobState.RUNNING, JobState.DROPPED, JobState.CANCELLED},
            JobState.RUNNING: {JobState.COMPLETED, JobState.PREEMPTED, JobState.CANCELLED},
            JobState.PREEMPTED: {JobState.QUEUED, JobState.DISPATCHED, JobState.RUNNING, JobState.DROPPED, JobState.CANCELLED},
            JobState.COMPLETED: set(),
            JobState.DROPPED: set(),
            JobState.CANCELLED: set(),
        }

        if new_state not in valid_transitions[self.status]:
            raise ValueError(
                f"Invalid job state transition from {self.status} to {new_state} for job '{self.job_id}'"
            )
        self.status = new_state

    def cancel(self, reason: str = "User cancelled", current_time_ms: Optional[float] = None) -> None:
        """Cancels a pending or active job, preventing further scheduling or execution."""
        if self.status in (JobState.COMPLETED, JobState.DROPPED):
            logger.warning(f"Cannot cancel job '{self.job_id}' in terminal state {self.status}")
            return
        logger.info(f"Job '{self.job_id}' cancelled. Reason: {reason}")
        self.metadata["cancellation_reason"] = reason
        if current_time_ms is not None:
            self.metadata["cancellation_time_ms"] = current_time_ms
        self.transition_to(JobState.CANCELLED)

    def mark_started(
        self,
        start_time: Optional[float] = None,
        start_time_ms: Optional[float] = None,
    ) -> None:
        """Marks the job as actively running on the GPU."""
        if self.status == JobState.CANCELLED:
            raise JobCancelledError(f"Cannot start cancelled job '{self.job_id}'")
        st = start_time if start_time is not None else start_time_ms
        if st is None:
            raise ValueError("start_time must be provided")
        if st < self.arrival_time:
            raise ValueError(
                f"start_time ({st}) cannot precede arrival_time ({self.arrival_time})"
            )
        if self.start_time is None:
            self.start_time = float(st)
            self.accumulated_waiting_time = max(0.0, st - self.arrival_time)
        self._last_preempt_time = None
        self.transition_to(JobState.RUNNING)

    def mark_preempted(self, preempt_time: float) -> None:
        """Marks the job as preempted at a quantum boundary."""
        if self.status == JobState.CANCELLED:
            raise JobCancelledError(f"Cannot preempt cancelled job '{self.job_id}'")
        if self.status != JobState.RUNNING:
            raise RuntimeError(f"Cannot preempt job '{self.job_id}' in state {self.status}")
        self._last_preempt_time = float(preempt_time)
        self.number_of_preemptions += 1
        self.transition_to(JobState.PREEMPTED)

    def execute_slice(self, actual_gpu_time: float, current_time: float) -> bool:
        """Executes a quantum slice on the GPU and updates remaining time.
        
        Returns:
            True if job finished all work, False if remaining work requires preemption.
        """
        if self.status == JobState.CANCELLED:
            raise JobCancelledError(f"Cannot execute slice on cancelled job '{self.job_id}'")
        if actual_gpu_time < 0.0:
            raise InvalidExecutionTimeError(f"actual_gpu_time must be non-negative, got {actual_gpu_time}")
        if self.start_time is None:
            self.mark_started(current_time)
        elif self._last_preempt_time is not None:
            self.accumulated_waiting_time += max(0.0, current_time - self._last_preempt_time)
            self._last_preempt_time = None

        self.accumulated_execution_time += actual_gpu_time
        self.remaining_gpu_time = max(0.0, self.remaining_gpu_time - actual_gpu_time)

        if self.remaining_gpu_time <= 0.0:
            self.mark_completed(current_time + actual_gpu_time)
            return True
        else:
            self.mark_preempted(current_time + actual_gpu_time)
            return False

    def mark_completed(
        self,
        completion_time: Optional[float] = None,
        actual_gpu_time: Optional[float] = None,
        completion_time_ms: Optional[float] = None,
        actual_gpu_time_ms: Optional[float] = None,
    ) -> None:
        """Marks the job as completed and records measured execution duration."""
        if self.start_time is None:
            raise RuntimeError(f"Job '{self.job_id}' completed without start_time recorded")
        
        ct = completion_time if completion_time is not None else completion_time_ms
        if ct is None:
            raise ValueError("completion_time must be provided")
        if ct < self.start_time:
            raise ValueError(
                f"completion_time ({ct}) cannot precede start_time ({self.start_time})"
            )

        act = actual_gpu_time if actual_gpu_time is not None else actual_gpu_time_ms
        if act is not None:
            if act < 0.0:
                raise ValueError(f"actual_gpu_time ({act}) cannot be negative")
            self.accumulated_execution_time = float(act)

        self.completion_time = float(ct)
        self.remaining_gpu_time = 0.0
        self.accumulated_waiting_time = max(0.0, self.completion_time - self.arrival_time - self.accumulated_execution_time)
        self._last_preempt_time = None
        self.transition_to(JobState.COMPLETED)

    # ---------------- Backward Compatibility Properties ----------------

    @property
    def arrival_time_ms(self) -> float:
        return self.arrival_time

    @property
    def start_time_ms(self) -> Optional[float]:
        return self.start_time

    @property
    def completion_time_ms(self) -> Optional[float]:
        return self.completion_time

    @property
    def actual_gpu_time_ms(self) -> float:
        return self.accumulated_execution_time

    @actual_gpu_time_ms.setter
    def actual_gpu_time_ms(self, value: float) -> None:
        self.accumulated_execution_time = float(value)

    @property
    def predicted_execution_time_ms(self) -> float:
        return self.estimated_gpu_time

    @predicted_execution_time_ms.setter
    def predicted_execution_time_ms(self, value: float) -> None:
        self.estimated_gpu_time = float(value)

    @property
    def deadline_ms(self) -> Optional[float]:
        return self.deadline

    @property
    def state(self) -> JobState:
        return self.status

    @state.setter
    def state(self, new_state: JobState) -> None:
        self.status = new_state

    @property
    def workload_class(self) -> WorkloadClass:
        return self.workload_type

    @property
    def waiting_time_ms(self) -> float:
        """Total waiting time spent in ready queues without receiving GPU execution."""
        if self.completion_time is not None:
            return max(0.0, self.completion_time - self.arrival_time - self.accumulated_execution_time)
        if self.accumulated_waiting_time > 0.0:
            return self.accumulated_waiting_time
        if self.start_time is not None:
            return max(0.0, self.start_time - self.arrival_time)
        return 0.0

    @property
    def turnaround_time_ms(self) -> float:
        """Total turnaround duration from arrival to completion."""
        if self.completion_time is not None:
            return max(0.0, self.completion_time - self.arrival_time)
        return 0.0

    @property
    def is_sla_violated(self) -> bool:
        """True if completion occurred after the specified deadline."""
        if self.deadline is None or self.completion_time is None:
            return False
        return self.completion_time > self.deadline

    def to_completion_report(self) -> JobCompletionReport:
        """Generates a JobCompletionReport telemetry record."""
        if self.status != JobState.COMPLETED or self.start_time is None or self.completion_time is None:
            raise RuntimeError(f"Cannot generate completion report for non-completed job '{self.job_id}'")

        return JobCompletionReport(
            job_id=self.job_id,
            tenant_id=self.tenant_id,
            model_name=self.model_name,
            arrival_time_ms=self.arrival_time,
            start_time_ms=self.start_time,
            completion_time_ms=self.completion_time,
            actual_gpu_time_ms=self.accumulated_execution_time,
            waiting_time_ms=self.waiting_time_ms,
            turnaround_time_ms=self.turnaround_time_ms,
            sla_deadline_ms=self.deadline,
            sla_violated=self.is_sla_violated,
            metadata=dict(self.metadata),
        )


# Backward compatibility alias
Job = GPUJob
