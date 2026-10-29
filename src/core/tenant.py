"""Multi-tenant representation, quota management, and aging logic for GPU-Aware Fair Scheduler."""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Union
import math
import logging
from src.core.job import GPUJob
from src.core.exceptions import InvalidTenantConfigError, TenantQuotaExhaustedError

logger = logging.getLogger(__name__)


@dataclass
class WaitingStatistics:
    """Statistics tracking queuing latency for a tenant."""
    total_waiting_time_ms: float = 0.0
    jobs_count: int = 0
    mean_waiting_time_ms: float = 0.0
    max_waiting_time_ms: float = 0.0
    min_waiting_time_ms: float = 0.0

    def record_wait(self, wait_ms: float) -> None:
        """Records an observed job waiting duration."""
        if wait_ms < 0.0:
            raise ValueError(f"wait_ms cannot be negative, got {wait_ms}")
        self.total_waiting_time_ms += wait_ms
        if self.jobs_count == 0:
            self.min_waiting_time_ms = wait_ms
            self.max_waiting_time_ms = wait_ms
        else:
            self.min_waiting_time_ms = min(self.min_waiting_time_ms, wait_ms)
            self.max_waiting_time_ms = max(self.max_waiting_time_ms, wait_ms)
        self.jobs_count += 1
        self.mean_waiting_time_ms = self.total_waiting_time_ms / self.jobs_count

    def reset(self) -> None:
        """Resets all waiting statistics."""
        self.total_waiting_time_ms = 0.0
        self.jobs_count = 0
        self.mean_waiting_time_ms = 0.0
        self.max_waiting_time_ms = 0.0
        self.min_waiting_time_ms = 0.0


@dataclass
class Tenant:
    """Represents an independent client/tenant sharing the edge GPU."""
    tenant_id: str
    priority: int                     # Base priority tier (1 = lowest, 10 = highest)
    weight: float                     # Scheduling weight w_i > 0 for proportional sharing
    gpu_quota: float                  # Allowed GPU share percentage (0.0 to 100.0)
    max_concurrent_jobs: int = 10     # Max active/concurrent jobs permitted
    name: str = ""
    quota_window_ms: float = 1000.0   # Sliding window duration in ms
    sla_latency_target_ms: float = 100.0
    max_queue_capacity: int = 100
    workload_type: str = "GENERIC"

    # Job tracking
    jobs: List[GPUJob] = field(default_factory=list)

    # Accounting & telemetry
    accumulated_gpu_service: float = 0.0
    waiting_statistics: WaitingStatistics = field(default_factory=WaitingStatistics)

    # Dynamic Scheduling State
    effective_priority: float = field(init=False)
    deficit_counter_ms: float = 0.0
    accumulated_gpu_time_window_ms: float = 0.0
    window_start_time_ms: float = 0.0
    last_service_time_ms: float = 0.0
    
    # Cumulative Performance Statistics
    total_jobs_submitted: int = 0
    total_jobs_completed: int = 0
    total_jobs_dropped: int = 0
    total_sla_violations: int = 0
    total_execution_time_ms: float = 0.0

    def __init__(
        self,
        tenant_id: str,
        *args: Any,
        priority: Optional[int] = None,
        weight: Optional[float] = None,
        gpu_quota: Optional[float] = None,
        max_concurrent_jobs: int = 10,
        name: Optional[str] = None,
        quota_window_ms: float = 1000.0,
        sla_latency_target_ms: float = 100.0,
        max_queue_capacity: int = 100,
        workload_type: str = "GENERIC",
        jobs: Optional[List[GPUJob]] = None,
        accumulated_gpu_service: float = 0.0,
        waiting_statistics: Optional[WaitingStatistics] = None,
        gpu_quota_pct: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        # Validate tenant_id
        if not tenant_id or not isinstance(tenant_id, str) or not tenant_id.strip():
            logger.error(f"Tenant configuration error: invalid tenant_id '{tenant_id}'")
            raise InvalidTenantConfigError(f"tenant_id must be a non-empty string, got '{tenant_id}'")
        self.tenant_id = tenant_id.strip()

        # Handle positional args polymorphism:
        # Form 1: Tenant(tenant_id, name, weight, priority, gpu_quota_pct)
        # Form 2: Tenant(tenant_id, priority, weight, gpu_quota)
        resolved_name = name
        resolved_priority = priority
        resolved_weight = weight
        resolved_quota = gpu_quota if gpu_quota is not None else gpu_quota_pct
        resolved_max_concurrent = max_concurrent_jobs

        if len(args) > 0:
            first_arg = args[0]
            if isinstance(first_arg, str):
                # Form 1: first_arg is name
                resolved_name = first_arg
                if len(args) > 1:
                    resolved_weight = float(args[1])
                if len(args) > 2:
                    resolved_priority = int(args[2])
                if len(args) > 3:
                    resolved_quota = float(args[3])
                if len(args) > 4:
                    resolved_max_concurrent = int(args[4])
            else:
                # Form 2: first_arg is priority (int)
                resolved_priority = int(first_arg)
                if len(args) > 1:
                    resolved_weight = float(args[1])
                if len(args) > 2:
                    resolved_quota = float(args[2])
                if len(args) > 3:
                    resolved_max_concurrent = int(args[3])

        # Validate priority
        if resolved_priority is None or not isinstance(resolved_priority, int) or not (1 <= resolved_priority <= 10):
            logger.error(f"Tenant configuration error for '{self.tenant_id}': invalid priority {resolved_priority}")
            raise InvalidTenantConfigError(f"priority must be an integer in [1, 10], got {resolved_priority}")
        self.priority = resolved_priority

        # Validate weight
        if (
            resolved_weight is None
            or not isinstance(resolved_weight, (int, float))
            or math.isnan(float(resolved_weight))
            or math.isinf(float(resolved_weight))
            or resolved_weight <= 0.0
        ):
            logger.error(f"Tenant configuration error for '{self.tenant_id}': invalid weight {resolved_weight}")
            raise InvalidTenantConfigError(f"weight must be strictly positive and finite, got {resolved_weight}")
        self.weight = float(resolved_weight)

        # Validate gpu_quota
        if resolved_quota is None:
            resolved_quota = 100.0

        if (
            not isinstance(resolved_quota, (int, float))
            or math.isnan(float(resolved_quota))
            or math.isinf(float(resolved_quota))
            or not (0.0 < resolved_quota <= 100.0)
        ):
            logger.error(f"Tenant configuration error for '{self.tenant_id}': invalid gpu_quota {resolved_quota}")
            raise InvalidTenantConfigError(f"gpu_quota must be in (0.0, 100.0], got {resolved_quota}")
        self.gpu_quota = float(resolved_quota)

        # Name
        self.name = resolved_name.strip() if resolved_name and resolved_name.strip() else self.tenant_id

        # Validate max_concurrent_jobs
        if not isinstance(resolved_max_concurrent, int) or resolved_max_concurrent <= 0:
            raise InvalidTenantConfigError(f"max_concurrent_jobs must be positive, got {resolved_max_concurrent}")
        self.max_concurrent_jobs = resolved_max_concurrent

        # Window & SLA
        if quota_window_ms <= 0.0 or math.isnan(quota_window_ms):
            raise InvalidTenantConfigError(f"quota_window_ms must be positive, got {quota_window_ms}")
        self.quota_window_ms = float(quota_window_ms)

        if sla_latency_target_ms <= 0.0 or math.isnan(sla_latency_target_ms):
            raise InvalidTenantConfigError(f"sla_latency_target_ms must be positive, got {sla_latency_target_ms}")
        self.sla_latency_target_ms = float(sla_latency_target_ms)

        if max_queue_capacity <= 0:
            raise InvalidTenantConfigError(f"max_queue_capacity must be positive, got {max_queue_capacity}")
        self.max_queue_capacity = int(max_queue_capacity)

        self.workload_type = str(workload_type)

        # Collections & metrics
        self.jobs = list(jobs) if jobs is not None else []
        self._job_id_set: set[str] = {j.job_id for j in self.jobs}
        self.accumulated_gpu_service = float(accumulated_gpu_service)
        self.waiting_statistics = waiting_statistics if waiting_statistics is not None else WaitingStatistics()

        # Dynamic state
        self.effective_priority = float(self.priority)
        self.deficit_counter_ms = 0.0
        self.accumulated_gpu_time_window_ms = 0.0
        self.window_start_time_ms = 0.0
        self.last_service_time_ms = 0.0
        self.total_jobs_submitted = len(self.jobs)
        self.total_jobs_completed = 0
        self.total_jobs_dropped = 0
        self.total_sla_violations = 0
        self.total_execution_time_ms = 0.0

    def has_job(self, job_or_id: Union[str, GPUJob]) -> bool:
        """O(1) membership check for active tracked jobs."""
        jid = job_or_id.job_id if hasattr(job_or_id, "job_id") else str(job_or_id)
        return jid in self._job_id_set

    def add_job(self, job: GPUJob) -> None:
        """Attaches a new job to this tenant's tracked active jobs."""
        if job.tenant_id != self.tenant_id:
            raise ValueError(
                f"Job tenant '{job.tenant_id}' does not match tenant '{self.tenant_id}'"
            )
        self.jobs.append(job)
        self._job_id_set.add(job.job_id)
        self.total_jobs_submitted += 1

    def remove_job(self, job_id: str) -> Optional[GPUJob]:
        """Removes and returns an active job by its job_id."""
        if job_id not in self._job_id_set:
            return None
        for idx, j in enumerate(self.jobs):
            if j.job_id == job_id:
                self._job_id_set.discard(job_id)
                return self.jobs.pop(idx)
        return None

    @property
    def active_jobs_count(self) -> int:
        """Returns the count of active incomplete jobs."""
        return len(self.jobs)

    def can_admit_concurrent_job(self) -> bool:
        """True if tenant has not reached its maximum concurrent job limit."""
        return len(self.jobs) < self.max_concurrent_jobs

    def apply_aging(
        self,
        current_time_ms: float,
        alpha: float = 0.15,
        starvation_threshold_ms: float = 150.0,
    ) -> float:
        """Calculates and updates effective priority based on starvation wait time.
        
        P_eff(t) = P_base + alpha * max(0, (t - last_service) / starvation_threshold)
        """
        if current_time_ms < self.last_service_time_ms:
            time_since_service = 0.0
        else:
            time_since_service = current_time_ms - self.last_service_time_ms

        aging_boost = alpha * (time_since_service / max(1.0, starvation_threshold_ms))
        self.effective_priority = float(self.priority) + aging_boost
        return self.effective_priority

    def update_window(self, current_time_ms: float) -> None:
        """Slides or rolls over the quota measurement window if elapsed."""
        if current_time_ms - self.window_start_time_ms >= self.quota_window_ms - 1e-9:
            self.accumulated_gpu_time_window_ms = 0.0
            self.window_start_time_ms = current_time_ms

    def is_quota_exceeded(self, current_time_ms: float) -> bool:
        """Checks if tenant has exhausted its maximum allowable GPU time in current window."""
        self.update_window(current_time_ms)
        allowed_gpu_time_ms = (self.gpu_quota / 100.0) * self.quota_window_ms
        return self.accumulated_gpu_time_window_ms >= allowed_gpu_time_ms

    def enforce_quota_strict(self, current_time_ms: float) -> None:
        """Validates that tenant is within quota limits, raising TenantQuotaExhaustedError if exceeded."""
        if self.is_quota_exceeded(current_time_ms):
            allowed_gpu_time_ms = (self.gpu_quota / 100.0) * self.quota_window_ms
            logger.warning(
                f"Tenant quota exhaustion: tenant '{self.tenant_id}' consumed "
                f"{self.accumulated_gpu_time_window_ms:.1f}ms >= allowed {allowed_gpu_time_ms:.1f}ms "
                f"in window {self.window_start_time_ms:.1f}..{self.window_start_time_ms + self.quota_window_ms:.1f}ms"
            )
            raise TenantQuotaExhaustedError(
                f"Tenant '{self.tenant_id}' has exhausted its GPU quota ({self.gpu_quota}%) "
                f"with {self.accumulated_gpu_time_window_ms:.1f}ms consumed in window starting at {self.window_start_time_ms:.1f}ms."
            )

    def record_gpu_execution(self, gpu_time_ms: float, current_time_ms: float) -> None:
        """Records consumed GPU compute time against quota and updates last service timestamp."""
        if gpu_time_ms < 0.0:
            raise ValueError(f"gpu_time_ms must be non-negative, got {gpu_time_ms}")
        self.update_window(current_time_ms)
        self.accumulated_gpu_time_window_ms += gpu_time_ms
        self.accumulated_gpu_service += gpu_time_ms
        self.last_service_time_ms = current_time_ms

    def replenish_deficit(self, quantum_slice_ms: float) -> None:
        """Adds a quantum allocation to the Deficit Round Robin counter."""
        if quantum_slice_ms < 0.0:
            raise ValueError(f"quantum_slice_ms must be non-negative, got {quantum_slice_ms}")
        self.deficit_counter_ms += quantum_slice_ms

    def deduct_deficit(self, used_time_ms: float) -> None:
        """Deducts consumed execution time from the Deficit Round Robin counter."""
        if used_time_ms < 0.0:
            raise ValueError(f"used_time_ms must be non-negative, got {used_time_ms}")
        self.deficit_counter_ms -= used_time_ms

    def record_job_completed(self, job: GPUJob) -> None:
        """Updates cumulative tenant telemetry upon job completion."""
        self.total_jobs_completed += 1
        self.waiting_statistics.record_wait(job.waiting_time_ms)
        self.total_execution_time_ms += job.actual_gpu_time_ms
        if job.is_sla_violated:
            self.total_sla_violations += 1

        # Remove from active jobs list if present
        self.remove_job(job.job_id)

    def record_job_dropped(self) -> None:
        """Increments drop counter when an ingress job is rejected."""
        self.total_jobs_dropped += 1

    # ---------------- Backward Compatibility Properties ----------------

    @property
    def gpu_quota_pct(self) -> float:
        return self.gpu_quota

    @gpu_quota_pct.setter
    def gpu_quota_pct(self, val: float) -> None:
        self.gpu_quota = float(val)

    @property
    def accumulated_gpu_time_ms(self) -> float:
        return self.accumulated_gpu_service

    @accumulated_gpu_time_ms.setter
    def accumulated_gpu_time_ms(self, val: float) -> None:
        self.accumulated_gpu_service = float(val)

    @property
    def total_waiting_time_ms(self) -> float:
        return self.waiting_statistics.total_waiting_time_ms

    @property
    def mean_waiting_time_ms(self) -> float:
        return self.waiting_statistics.mean_waiting_time_ms

    @property
    def mean_execution_time_ms(self) -> float:
        if self.total_jobs_completed == 0:
            return 0.0
        return self.total_execution_time_ms / self.total_jobs_completed

    @property
    def sla_violation_rate(self) -> float:
        if self.total_jobs_completed == 0:
            return 0.0
        return self.total_sla_violations / self.total_jobs_completed

    def reset_dynamic_state(self) -> None:
        """Resets dynamic scheduling state back to initial values for fresh experiment runs."""
        self.effective_priority = float(self.priority)
        self.deficit_counter_ms = 0.0
        self.accumulated_gpu_time_window_ms = 0.0
        self.accumulated_gpu_service = 0.0
        self.window_start_time_ms = 0.0
        self.last_service_time_ms = 0.0
        self.jobs.clear()
        self.waiting_statistics.reset()
        self.total_jobs_submitted = 0
        self.total_jobs_completed = 0
        self.total_jobs_dropped = 0
        self.total_sla_violations = 0
        self.total_execution_time_ms = 0.0
