"""Scheduling decision container recording arbiter outputs at each dispatch tick."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from src.core.job import GPUJob


@dataclass
class SchedulingDecision:
    """Represents an atomic, fully explainable scheduling decision produced by the policy engine."""
    decision_id: str
    timestamp: float
    selected_job: Optional[GPUJob]
    selected_tenant_id: Optional[str]
    allocated_quantum_ms: float
    reason: str
    active_tenants_count: int = 0
    deficit_snapshot: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Explicit explainability fields
    predicted_workload_type: str = "BALANCED"
    predicted_gpu_behavior: Dict[str, Any] = field(default_factory=dict)
    current_tenant_gpu_service: float = 0.0
    tenant_weight: float = 1.0
    tenant_quota_usage: float = 0.0
    priority: float = 1.0
    waiting_time: float = 0.0
    scheduling_score: float = 0.0
    reason_for_selection: str = ""
    score_breakdown: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.decision_id or not isinstance(self.decision_id, str):
            raise ValueError(f"decision_id must be a non-empty string, got '{self.decision_id}'")

        if self.timestamp < 0.0:
            raise ValueError(f"timestamp must be non-negative, got {self.timestamp}")

        if self.allocated_quantum_ms < 0.0:
            raise ValueError(
                f"allocated_quantum_ms must be non-negative, got {self.allocated_quantum_ms}"
            )

        if not self.reason or not isinstance(self.reason, str):
            raise ValueError(f"reason must be a non-empty string, got '{self.reason}'")

        if self.active_tenants_count < 0:
            raise ValueError(
                f"active_tenants_count must be non-negative, got {self.active_tenants_count}"
            )

        if self.selected_job is not None:
            if not self.selected_tenant_id:
                self.selected_tenant_id = self.selected_job.tenant_id
            elif self.selected_tenant_id != self.selected_job.tenant_id:
                raise ValueError(
                    f"Tenant mismatch: selected_tenant_id '{self.selected_tenant_id}' != "
                    f"selected_job.tenant_id '{self.selected_job.tenant_id}'"
                )

        if not self.reason_for_selection:
            self.reason_for_selection = self.reason

    @property
    def is_idle(self) -> bool:
        """True if the scheduler decided to idle (no eligible work available)."""
        return self.selected_job is None

    @property
    def selected_tenant(self) -> Optional[str]:
        """Convenience property for selected tenant identifier."""
        return self.selected_tenant_id

    @property
    def selected_quantum(self) -> float:
        """Convenience property for allocated quantum in milliseconds."""
        return self.allocated_quantum_ms

    def explain(self) -> str:
        """Formats the scheduling decision into an explainable audit record."""
        if self.is_idle:
            return f"[{self.decision_id} @ {self.timestamp:.1f}ms] IDLE: No eligible jobs ready for dispatch."

        job_str = self.selected_job.job_id if self.selected_job else "N/A"
        lines = [
            f"=== Scheduling Decision: {self.decision_id} (t={self.timestamp:.2f}ms) ===",
            f"Selected Tenant           : {self.selected_tenant_id}",
            f"Selected Job              : {job_str}",
            f"Reason For Selection      : {self.reason_for_selection}",
            f"Composite Score           : {self.scheduling_score:.4f}",
            f"Allocated Quantum         : {self.allocated_quantum_ms:.2f} ms",
            "--- Workload & ML Prediction ---",
            f"Predicted Workload Type   : {self.predicted_workload_type}",
            f"Predicted GPU Behavior    : {self.predicted_gpu_behavior}",
            "--- Tenant State ---",
            f"Tenant Weight             : {self.tenant_weight:.2f}",
            f"Base / Eff Priority       : {self.priority:.2f}",
            f"Accumulated GPU Service   : {self.current_tenant_gpu_service:.2f} ms",
            f"Quota Window Usage        : {self.tenant_quota_usage:.2f} ms",
            f"Job Waiting Time          : {self.waiting_time:.2f} ms",
            "--- Score Component Breakdown ---",
        ]
        for term, val in self.score_breakdown.items():
            lines.append(f"  {term:<25}: {val:+.4f}")
        lines.append("=============================================================")
        return "\n".join(lines)
