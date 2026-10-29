"""Scheduler configuration data structure and validation."""

from dataclasses import dataclass
from typing import Optional, Union, Dict, Any
from src.core.types import PolicyType, SchedulingQuantum
from src.core.exceptions import InvalidQuantumError


@dataclass
class SchedulerConfiguration:
    """Strongly typed, validated configuration parameters for GPU scheduling policies."""
    policy: PolicyType = PolicyType.PDWFQ
    base_quantum_ms: float = 25.0
    min_quantum_ms: float = 5.0
    max_quantum_ms: float = 100.0
    aging_alpha: float = 0.15
    starvation_threshold_ms: float = 150.0
    quota_window_ms: float = 1000.0
    enable_quota_enforcement: bool = True
    enable_workload_multiplier: bool = True
    max_preemptions_per_job: int = 5

    def __init__(
        self,
        policy: Union[PolicyType, str] = PolicyType.PDWFQ,
        base_quantum_ms: float = 25.0,
        min_quantum_ms: float = 5.0,
        max_quantum_ms: float = 100.0,
        aging_alpha: float = 0.15,
        starvation_threshold_ms: float = 150.0,
        quota_window_ms: float = 1000.0,
        enable_quota_enforcement: bool = True,
        enable_workload_multiplier: bool = True,
        max_preemptions_per_job: int = 5,
    ) -> None:
        if isinstance(policy, str):
            try:
                self.policy = PolicyType(policy.lower())
            except ValueError:
                raise ValueError(
                    f"Invalid policy '{policy}', must be one of {[p.value for p in PolicyType]}"
                )
        elif isinstance(policy, PolicyType):
            self.policy = policy
        else:
            raise TypeError(f"policy must be PolicyType or str, got {type(policy)}")

        if base_quantum_ms <= 0.0:
            raise InvalidQuantumError(f"base_quantum_ms must be positive, got {base_quantum_ms}")
        self.base_quantum_ms = float(base_quantum_ms)

        if min_quantum_ms <= 0.0:
            raise InvalidQuantumError(f"min_quantum_ms must be positive, got {min_quantum_ms}")
        self.min_quantum_ms = float(min_quantum_ms)

        if max_quantum_ms < base_quantum_ms:
            raise InvalidQuantumError(
                f"max_quantum_ms ({max_quantum_ms}) cannot be less than base_quantum_ms ({base_quantum_ms})"
            )
        self.max_quantum_ms = float(max_quantum_ms)

        if self.min_quantum_ms > self.base_quantum_ms:
            raise InvalidQuantumError(
                f"min_quantum_ms ({self.min_quantum_ms}) cannot exceed base_quantum_ms ({self.base_quantum_ms})"
            )

        if aging_alpha < 0.0:
            raise ValueError(f"aging_alpha must be non-negative, got {aging_alpha}")
        self.aging_alpha = float(aging_alpha)

        if starvation_threshold_ms <= 0.0:
            raise ValueError(
                f"starvation_threshold_ms must be positive, got {starvation_threshold_ms}"
            )
        self.starvation_threshold_ms = float(starvation_threshold_ms)

        if quota_window_ms <= 0.0:
            raise ValueError(f"quota_window_ms must be positive, got {quota_window_ms}")
        self.quota_window_ms = float(quota_window_ms)

        self.enable_quota_enforcement = bool(enable_quota_enforcement)
        self.enable_workload_multiplier = bool(enable_workload_multiplier)

        if max_preemptions_per_job < 0:
            raise ValueError(
                f"max_preemptions_per_job must be non-negative, got {max_preemptions_per_job}"
            )
        self.max_preemptions_per_job = int(max_preemptions_per_job)

    def to_scheduling_quantum(self) -> SchedulingQuantum:
        """Converts to core SchedulingQuantum object."""
        return SchedulingQuantum(
            base_quantum_ms=self.base_quantum_ms,
            min_quantum_ms=self.min_quantum_ms,
            max_quantum_ms=self.max_quantum_ms,
            aging_alpha=self.aging_alpha,
            starvation_threshold_ms=self.starvation_threshold_ms,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchedulerConfiguration":
        """Instantiates a validated configuration from a raw dictionary."""
        return cls(
            policy=data.get("policy", PolicyType.PDWFQ),
            base_quantum_ms=float(data.get("base_quantum_ms", 25.0)),
            min_quantum_ms=float(data.get("min_quantum_ms", 5.0)),
            max_quantum_ms=float(data.get("max_quantum_ms", 100.0)),
            aging_alpha=float(data.get("aging_alpha", 0.15)),
            starvation_threshold_ms=float(data.get("starvation_threshold_ms", 150.0)),
            quota_window_ms=float(data.get("quota_window_ms", 1000.0)),
            enable_quota_enforcement=bool(data.get("enable_quota_enforcement", True)),
            enable_workload_multiplier=bool(data.get("enable_workload_multiplier", True)),
            max_preemptions_per_job=int(data.get("max_preemptions_per_job", 5)),
        )


# Convenient alias
SchedulerConfig = SchedulerConfiguration
