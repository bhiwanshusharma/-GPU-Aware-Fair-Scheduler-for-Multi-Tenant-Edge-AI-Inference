"""Standard Round Robin Baseline GPU Scheduling Policy."""

from typing import Optional, Dict
from src.core.types import SchedulingQuantum
from src.core.job import GPUJob
from src.core.tenant import Tenant
from src.core.queues import TenantQueueManager
from src.scheduler.base import BaseScheduler


class RoundRobinScheduler(BaseScheduler):
    """Standard Round Robin baseline scheduler.
    
    Rotates through active tenant queues in cyclic order, allocating a fixed
    base GPU time quantum to each job. If a job does not complete within the
    quantum, it is preempted at the scheduling boundary and re-queued.
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
            name="Round Robin",
        )
        self._cursor: int = -1

    def schedule(self, current_time_ms: float) -> Optional[GPUJob]:
        """Selects the next job using cyclic round-robin across tenant queues."""
        tenant_ids = list(self.tenants.keys())
        if not tenant_ids:
            return None

        n = len(tenant_ids)
        for _ in range(n):
            self._cursor = (self._cursor + 1) % n
            candidate_id = tenant_ids[self._cursor]
            queue = self.queue_manager.get_queue(candidate_id)
            if not queue.is_empty:
                self.total_dispatches += 1
                return self.queue_manager.dequeue(candidate_id)

        return None

    def get_allocated_quantum(self, job: GPUJob) -> float:
        """Standard Round Robin allocates a fixed base quantum for every tenant."""
        return self.quantum.base_quantum_ms

    def on_job_completed(
        self,
        job: GPUJob,
        actual_gpu_time_ms: float,
        completion_time_ms: float,
    ) -> None:
        """Completion notification."""
        pass

    def reset(self) -> None:
        """Resets cursor and superclass state."""
        super().reset()
        self._cursor = -1
