"""Strict Priority Baseline GPU Scheduling Policy."""

from typing import Optional, Dict, List
from src.core.types import SchedulingQuantum
from src.core.job import GPUJob
from src.core.tenant import Tenant
from src.core.queues import TenantQueueManager
from src.scheduler.base import BaseScheduler


class PriorityScheduler(BaseScheduler):
    """Strict Priority baseline scheduler.
    
    Always dispatches inference requests belonging to the tenant with the
    highest configured priority tier (10 = highest, 1 = lowest).
    Lower priority tenants can starve if higher priority tenants continuously
    submit work.
    """

    def __init__(
        self,
        tenants: Optional[Dict[str, Tenant]] = None,
        queue_manager: Optional[TenantQueueManager] = None,
        quantum: Optional[SchedulingQuantum] = None,
        preemptive_quantum: bool = True,
    ) -> None:
        super().__init__(
            tenants=tenants,
            queue_manager=queue_manager,
            quantum=quantum,
            name="Priority Scheduling",
        )
        self.preemptive_quantum = preemptive_quantum

    def schedule(self, current_time_ms: float) -> Optional[GPUJob]:
        """Selects the job belonging to the highest-priority active tenant."""
        active_tenant_ids = self.queue_manager.get_active_tenant_ids()
        if not active_tenant_ids:
            return None

        # Sort active tenants by base priority descending, tie-break by head arrival time
        def tenant_sort_key(t_id: str):
            t = self.tenants[t_id]
            head = self.queue_manager.peek(t_id)
            head_arr = head.arrival_time if head else float("inf")
            # Return tuple: (-priority, head_arrival_time)
            return (-t.priority, head_arr)

        sorted_tenant_ids = sorted(active_tenant_ids, key=tenant_sort_key)
        chosen_tenant_id = sorted_tenant_ids[0]

        self.total_dispatches += 1
        return self.queue_manager.dequeue(chosen_tenant_id)

    def get_allocated_quantum(self, job: GPUJob) -> float:
        """Determines time slice. If preemptive_quantum is True, allocates base quantum
        to allow higher-priority jobs arriving later to be dispatched at quantum boundary.
        """
        if self.preemptive_quantum:
            return self.quantum.base_quantum_ms
        return max(0.1, job.remaining_gpu_time)

    def on_job_completed(
        self,
        job: GPUJob,
        actual_gpu_time_ms: float,
        completion_time_ms: float,
    ) -> None:
        """Completion notification."""
        pass
