"""Weighted Round Robin (WRR) Baseline GPU Scheduling Policy."""

from typing import Optional, Dict, List
from src.core.types import SchedulingQuantum
from src.core.job import GPUJob
from src.core.tenant import Tenant
from src.core.queues import TenantQueueManager
from src.scheduler.base import BaseScheduler


class WeightedRoundRobinScheduler(BaseScheduler):
    """Weighted Round Robin (WRR) baseline scheduler.
    
    Allocates GPU execution time slices proportional to configured tenant weights (w_i).
    In each round, tenant i receives a continuous time budget Q_i = Q_base * w_i.
    When a tenant exhausts its round budget or has an empty queue, the scheduler
    cycles to the next active tenant.
    """

    def __init__(
        self,
        tenants: Optional[Dict[str, Tenant]] = None,
        queue_manager: Optional[TenantQueueManager] = None,
        quantum: Optional[SchedulingQuantum] = None,
    ) -> None:
        super().__init__(
            tenants=tenants,
            queue_manager=queue_manager,
            quantum=quantum,
            name="Weighted Round Robin",
        )
        self._cursor: int = -1
        self._current_tenant_id: Optional[str] = None
        self._current_round_budget_ms: float = 0.0

    def schedule(self, current_time_ms: float) -> Optional[GPUJob]:
        """Selects the next job, maintaining proportional round budgets per tenant."""
        tenant_ids = list(self.tenants.keys())
        if not tenant_ids:
            return None

        # Check if current tenant still has remaining round budget and queued jobs
        if (
            self._current_tenant_id is not None
            and self._current_round_budget_ms > 0.0
            and not self.queue_manager.get_queue(self._current_tenant_id).is_empty
        ):
            self.total_dispatches += 1
            return self.queue_manager.dequeue(self._current_tenant_id)

        # Otherwise, rotate to next tenant with work
        n = len(tenant_ids)
        for _ in range(n):
            self._cursor = (self._cursor + 1) % n
            candidate_id = tenant_ids[self._cursor]
            queue = self.queue_manager.get_queue(candidate_id)
            if not queue.is_empty:
                self._current_tenant_id = candidate_id
                t = self.tenants[candidate_id]
                self._current_round_budget_ms = self.quantum.base_quantum_ms * t.weight
                self.total_dispatches += 1
                return self.queue_manager.dequeue(candidate_id)

        # All queues are empty
        self._current_tenant_id = None
        self._current_round_budget_ms = 0.0
        return None

    def get_allocated_quantum(self, job: GPUJob) -> float:
        """Allocates a slice up to the tenant's remaining round budget."""
        if self._current_round_budget_ms > 0.0:
            return min(job.remaining_gpu_time, self._current_round_budget_ms)
        t = self.tenants.get(job.tenant_id)
        w = t.weight if t else 1.0
        return self.quantum.base_quantum_ms * w

    def on_job_completed(
        self,
        job: GPUJob,
        actual_gpu_time_ms: float,
        completion_time_ms: float,
    ) -> None:
        """Deducts consumed execution time from current tenant's round budget."""
        self._current_round_budget_ms = max(0.0, self._current_round_budget_ms - actual_gpu_time_ms)
        if self._current_round_budget_ms <= 0.0:
            self._current_tenant_id = None

    def on_job_preempted(self, job: GPUJob, current_time_ms: float) -> None:
        """When quantum expires at slice boundary, advance to next tenant in next schedule."""
        self._current_round_budget_ms = 0.0
        self._current_tenant_id = None

    def reset(self) -> None:
        """Resets round state and counters."""
        super().reset()
        self._cursor = -1
        self._current_tenant_id = None
        self._current_round_budget_ms = 0.0
