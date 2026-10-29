"""First-In-First-Out (FIFO) Baseline GPU Scheduling Policy."""

import math
from typing import Optional, Dict
from src.core.types import SchedulingQuantum
from src.core.job import GPUJob
from src.core.tenant import Tenant
from src.core.queues import TenantQueueManager
from src.scheduler.base import BaseScheduler


class FIFOScheduler(BaseScheduler):
    """First-In, First-Out (FIFO) baseline scheduler.
    
    Examines all tenant ingress queues and dispatches jobs in strict order
    of arrival timestamp. Runs jobs to completion without time-slicing.
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
            name="FIFO",
        )

    def schedule(self, current_time_ms: float) -> Optional[GPUJob]:
        """Selects the earliest arriving job across all active tenant queues."""
        active_tenant_ids = self.queue_manager.get_active_tenant_ids()
        if not active_tenant_ids:
            return None

        earliest_job: Optional[GPUJob] = None
        chosen_tenant_id: Optional[str] = None

        for t_id in active_tenant_ids:
            head = self.queue_manager.peek(t_id)
            if head is not None:
                if earliest_job is None:
                    earliest_job = head
                    chosen_tenant_id = t_id
                elif head.arrival_time < earliest_job.arrival_time - 1e-9:
                    earliest_job = head
                    chosen_tenant_id = t_id
                elif math.isclose(head.arrival_time, earliest_job.arrival_time, abs_tol=1e-9):
                    # Deterministic tie-breaker by (tenant_id, job_id)
                    cand_key = (head.tenant_id, head.job_id)
                    earliest_key = (earliest_job.tenant_id, earliest_job.job_id)
                    if cand_key < earliest_key:
                        earliest_job = head
                        chosen_tenant_id = t_id

        if chosen_tenant_id is not None:
            self.total_dispatches += 1
            return self.queue_manager.dequeue(chosen_tenant_id)

        return None

    def get_allocated_quantum(self, job: GPUJob) -> float:
        """In FIFO, jobs run non-preemptively to completion without quantum preemption."""
        return max(0.1, job.remaining_gpu_time)

    def on_job_completed(
        self,
        job: GPUJob,
        actual_gpu_time_ms: float,
        completion_time_ms: float,
    ) -> None:
        """Completion notification."""
        pass
