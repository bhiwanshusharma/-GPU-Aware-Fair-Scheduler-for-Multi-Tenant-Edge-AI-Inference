"""Abstract base class for GPU inference scheduling policies."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import logging

from src.core.types import SchedulingQuantum
from src.core.job import GPUJob, Job
from src.core.tenant import Tenant
from src.core.queues import TenantQueueManager

logger = logging.getLogger(__name__)


class BaseScheduler(ABC):
    """Abstract interface defining the scheduler contract for GPU inference dispatch."""

    def __init__(
        self,
        tenants: Optional[Dict[str, Tenant]] = None,
        queue_manager: Optional[TenantQueueManager] = None,
        quantum: Optional[SchedulingQuantum] = None,
        name: str = "BaseScheduler",
    ) -> None:
        self.tenants = tenants if tenants is not None else {}
        self.queue_manager = queue_manager if queue_manager is not None else TenantQueueManager()
        self.quantum = quantum or SchedulingQuantum()
        self.name = name
        self.total_dispatches = 0

        # Ensure registered tenants exist in queue_manager
        for t in self.tenants.values():
            if t.tenant_id not in self.queue_manager.registered_tenant_ids:
                self.queue_manager.register_tenant(t)

    @abstractmethod
    def schedule(self, current_time_ms: float) -> Optional[GPUJob]:
        """Selects the next eligible inference job to dispatch to the GPU.
        
        Args:
            current_time_ms: Current simulation timestamp in milliseconds.
            
        Returns:
            The selected GPUJob instance, or None if no eligible job is ready.
        """
        pass

    def get_allocated_quantum(self, job: GPUJob) -> float:
        """Determines the maximum time quantum (slice) allocated for this dispatch.
        
        Default implementation returns the configured base quantum.
        """
        return self.quantum.base_quantum_ms

    @abstractmethod
    def on_job_completed(
        self,
        job: GPUJob,
        actual_gpu_time_ms: float,
        completion_time_ms: float,
    ) -> None:
        """Lifecycle callback invoked when an inference job completes execution."""
        pass

    def on_job_preempted(self, job: GPUJob, current_time_ms: float) -> None:
        """Lifecycle callback invoked when an inference job is preempted at quantum boundary."""
        pass

    def register_tenant(self, tenant: Tenant) -> None:
        """Registers a new tenant and establishes its ingress queue."""
        self.tenants[tenant.tenant_id] = tenant
        self.queue_manager.register_tenant(tenant)

    def get_active_tenants(self) -> List[Tenant]:
        """Returns list of tenants that currently have jobs waiting in their ingress queue."""
        active_ids = self.queue_manager.get_active_tenant_ids()
        return [self.tenants[t_id] for t_id in active_ids if t_id in self.tenants]

    def reset(self) -> None:
        """Resets scheduler internal state and tenant dynamic counters."""
        self.total_dispatches = 0
        self.queue_manager.reset()
        for tenant in self.tenants.values():
            tenant.reset_dynamic_state()
