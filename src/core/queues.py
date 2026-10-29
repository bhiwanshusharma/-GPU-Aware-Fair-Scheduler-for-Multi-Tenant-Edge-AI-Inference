"""Per-tenant isolated job queues and queue manager with backpressure."""

from collections import deque
from typing import Dict, List, Optional
import logging

from src.core.types import JobState
from src.core.job import Job
from src.core.tenant import Tenant
from src.core.exceptions import (
    EmptyQueueError,
    DuplicateJobError,
    UnknownTenantError,
    JobCancelledError,
)

logger = logging.getLogger(__name__)


class QueueFullError(Exception):
    """Raised when an ingress job is enqueued to a full tenant queue under strict drop policy."""
    pass


class TenantJobQueue:
    """Isolated, capacity-bounded FIFO queue for a single tenant."""

    def __init__(self, tenant_id: str, max_capacity: int = 100):
        if max_capacity <= 0:
            raise ValueError(f"max_capacity must be positive, got {max_capacity}")
        self.tenant_id = tenant_id
        self.max_capacity = max_capacity
        self._queue: deque[Job] = deque()
        self._job_ids: set[str] = set()
        self._total_enqueued = 0
        self._total_dequeued = 0
        self._total_dropped = 0

    def enqueue(self, job: Job) -> bool:
        """Enqueues an inference job. Rejects if capacity is reached or ID is duplicate.
        
        Returns:
            True if job was admitted, False if dropped due to capacity overflow.
            
        Raises:
            ValueError: If job belongs to another tenant.
            DuplicateJobError: If job ID is already queued.
            JobCancelledError: If job is already cancelled.
        """
        if job.status == JobState.CANCELLED:
            logger.error(f"Cannot enqueue cancelled job '{job.job_id}' for tenant '{self.tenant_id}'")
            raise JobCancelledError(f"Cannot enqueue cancelled job '{job.job_id}'.")

        if job.tenant_id != self.tenant_id:
            logger.error(
                f"Cross-tenant pollution: job '{job.job_id}' (tenant='{job.tenant_id}') != queue '{self.tenant_id}'"
            )
            raise ValueError(
                f"Cross-tenant pollution error: Job tenant '{job.tenant_id}' does not match queue '{self.tenant_id}'"
            )

        if job.job_id in self._job_ids:
            logger.error(f"Duplicate job rejection: Job ID '{job.job_id}' is already present in queue '{self.tenant_id}'")
            raise DuplicateJobError(
                f"Duplicate job ID detected: '{job.job_id}' is already queued for tenant '{self.tenant_id}'."
            )

        if len(self._queue) >= self.max_capacity:
            self._total_dropped += 1
            job.transition_to(JobState.DROPPED)
            logger.warning(
                f"Queue overflow for tenant '{self.tenant_id}': Job '{job.job_id}' dropped (size={len(self._queue)})"
            )
            return False

        self._queue.append(job)
        self._job_ids.add(job.job_id)
        self._total_enqueued += 1
        return True

    def dequeue(self) -> Optional[Job]:
        """Pops and returns the job at the head of the queue, or None if empty."""
        if not self._queue:
            return None
        job = self._queue.popleft()
        self._job_ids.discard(job.job_id)
        self._total_dequeued += 1
        return job

    def dequeue_strict(self) -> Job:
        """Pops and returns the job at the head of the queue. Raises EmptyQueueError if empty."""
        job = self.dequeue()
        if job is None:
            logger.warning(f"Strict dequeue attempted on empty queue for tenant '{self.tenant_id}'")
            raise EmptyQueueError(f"Cannot dequeue: Queue for tenant '{self.tenant_id}' is empty.")
        return job

    def cancel_job(
        self,
        job_id: str,
        reason: str = "User cancelled",
        current_time_ms: Optional[float] = None,
    ) -> Optional[Job]:
        """Cancels and removes a pending job from the queue if present."""
        if job_id not in self._job_ids:
            return None

        for idx, job in enumerate(self._queue):
            if job.job_id == job_id:
                del self._queue[idx]
                self._job_ids.discard(job_id)
                job.cancel(reason, current_time_ms)
                logger.info(f"Cancelled queued job '{job_id}' from tenant '{self.tenant_id}' queue.")
                return job
        return None

    def peek(self) -> Optional[Job]:
        """Returns the job at the head without removing it."""
        if not self._queue:
            return None
        return self._queue[0]

    def clear(self) -> None:
        """Empties the queue."""
        self._queue.clear()
        self._job_ids.clear()

    @property
    def is_empty(self) -> bool:
        return len(self._queue) == 0

    @property
    def is_full(self) -> bool:
        return len(self._queue) >= self.max_capacity

    @property
    def size(self) -> int:
        return len(self._queue)

    @property
    def total_enqueued(self) -> int:
        return self._total_enqueued

    @property
    def total_dequeued(self) -> int:
        return self._total_dequeued

    @property
    def total_dropped(self) -> int:
        return self._total_dropped

    def __len__(self) -> int:
        return len(self._queue)


class TenantQueueManager:
    """Manages independent isolated job queues for all registered tenants."""

    def __init__(self) -> None:
        self._queues: Dict[str, TenantJobQueue] = {}
        self._active_tenant_ids: set[str] = set()

    def register_tenant(self, tenant: Tenant) -> None:
        """Registers a tenant and establishes an isolated ingress queue."""
        if tenant.tenant_id in self._queues:
            logger.debug(f"Re-registering tenant queue for '{tenant.tenant_id}'")
        self._queues[tenant.tenant_id] = TenantJobQueue(
            tenant_id=tenant.tenant_id,
            max_capacity=tenant.max_queue_capacity,
        )
        self._active_tenant_ids.discard(tenant.tenant_id)

    def enqueue(self, job: Job) -> bool:
        """Routes an incoming job to its dedicated tenant queue."""
        if job.tenant_id not in self._queues:
            logger.error(f"Unknown tenant error: '{job.tenant_id}' is not registered with queue manager.")
            raise UnknownTenantError(
                f"No registered queue found for tenant '{job.tenant_id}'. Registered tenants: {list(self._queues.keys())}"
            )
        admitted = self._queues[job.tenant_id].enqueue(job)
        if admitted:
            self._active_tenant_ids.add(job.tenant_id)
        return admitted

    def dequeue(self, tenant_id: str) -> Optional[Job]:
        """Dequeues from a specific tenant queue."""
        if tenant_id not in self._queues:
            logger.error(f"Unknown tenant error during dequeue: '{tenant_id}' is not registered.")
            raise UnknownTenantError(
                f"No registered queue found for tenant '{tenant_id}'. Registered tenants: {list(self._queues.keys())}"
            )
        job = self._queues[tenant_id].dequeue()
        if self._queues[tenant_id].is_empty:
            self._active_tenant_ids.discard(tenant_id)
        return job

    def dequeue_strict(self, tenant_id: str) -> Job:
        """Strictly dequeues from a specific tenant queue, raising EmptyQueueError if empty."""
        if tenant_id not in self._queues:
            raise UnknownTenantError(
                f"No registered queue found for tenant '{tenant_id}'. Registered tenants: {list(self._queues.keys())}"
            )
        job = self._queues[tenant_id].dequeue_strict()
        if self._queues[tenant_id].is_empty:
            self._active_tenant_ids.discard(tenant_id)
        return job

    def cancel_job(
        self,
        tenant_id: str,
        job_id: str,
        reason: str = "User cancelled",
        current_time_ms: Optional[float] = None,
    ) -> Optional[Job]:
        """Cancels a job in the specified tenant queue."""
        if tenant_id not in self._queues:
            raise UnknownTenantError(
                f"No registered queue found for tenant '{tenant_id}'. Registered tenants: {list(self._queues.keys())}"
            )
        job = self._queues[tenant_id].cancel_job(job_id, reason, current_time_ms)
        if self._queues[tenant_id].is_empty:
            self._active_tenant_ids.discard(tenant_id)
        return job

    def peek(self, tenant_id: str) -> Optional[Job]:
        """Peeks at the next job in a specific tenant queue."""
        if tenant_id not in self._queues:
            logger.error(f"Unknown tenant error during peek: '{tenant_id}' is not registered.")
            raise UnknownTenantError(
                f"No registered queue found for tenant '{tenant_id}'. Registered tenants: {list(self._queues.keys())}"
            )
        return self._queues[tenant_id].peek()

    def get_queue(self, tenant_id: str) -> TenantJobQueue:
        """Returns the queue for the specified tenant."""
        if tenant_id not in self._queues:
            logger.error(f"Unknown tenant error getting queue: '{tenant_id}' is not registered.")
            raise UnknownTenantError(
                f"No registered queue found for tenant '{tenant_id}'. Registered tenants: {list(self._queues.keys())}"
            )
        return self._queues[tenant_id]

    def get_active_tenant_ids(self) -> List[str]:
        """Returns list of tenant IDs that currently have at least one queued job."""
        return sorted(self._active_tenant_ids)

    @property
    def total_queued_jobs(self) -> int:
        """Total number of queued jobs across all tenant queues."""
        return sum(len(q) for q in self._queues.values())

    @property
    def registered_tenant_ids(self) -> List[str]:
        """List of all registered tenant IDs."""
        return list(self._queues.keys())

    def reset(self) -> None:
        """Clears all tenant queues."""
        for q in self._queues.values():
            q.clear()
        self._active_tenant_ids.clear()
