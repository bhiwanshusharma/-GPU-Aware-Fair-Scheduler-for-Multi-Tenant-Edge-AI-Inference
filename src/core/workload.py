"""Workload profile representation and job instantiation factory."""

from dataclasses import dataclass
from typing import Tuple, Optional, Union
from src.core.types import WorkloadClass
from src.core.job import GPUJob


@dataclass
class Workload:
    """Represents a profiled edge AI workload model family with execution characteristics."""
    model_name: str
    domain: str
    workload_type: WorkloadClass
    input_shape: Tuple[int, ...]
    default_batch_size: int = 1
    max_batch_size: int = 16
    estimated_flops_g: float = 1.0
    memory_footprint_mb: float = 10.0
    calibrated_base_latency_ms: float = 10.0
    latency_per_batch_item_ms: float = 2.0

    def __init__(
        self,
        model_name: str,
        domain: str,
        workload_type: Optional[Union[WorkloadClass, str]] = None,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        default_batch_size: int = 1,
        max_batch_size: int = 16,
        estimated_flops_g: float = 1.0,
        memory_footprint_mb: float = 10.0,
        calibrated_base_latency_ms: float = 10.0,
        latency_per_batch_item_ms: float = 2.0,
        category: Optional[Union[WorkloadClass, str]] = None,
    ) -> None:
        if not model_name or not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(f"model_name must be a non-empty string, got '{model_name}'")
        self.model_name = model_name.strip()

        if not domain or not isinstance(domain, str) or not domain.strip():
            raise ValueError(f"domain must be a non-empty string, got '{domain}'")
        self.domain = domain.strip()

        wt = workload_type if workload_type is not None else category
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

        if not input_shape or not isinstance(input_shape, (tuple, list)):
            raise ValueError(f"input_shape must be a non-empty tuple or list, got {input_shape}")
        self.input_shape = tuple(input_shape)

        if not isinstance(default_batch_size, int) or default_batch_size <= 0:
            raise ValueError(f"default_batch_size must be positive, got {default_batch_size}")
        self.default_batch_size = default_batch_size

        if not isinstance(max_batch_size, int) or max_batch_size < default_batch_size:
            raise ValueError(
                f"max_batch_size ({max_batch_size}) must be >= default_batch_size ({default_batch_size})"
            )
        self.max_batch_size = max_batch_size

        if estimated_flops_g <= 0.0:
            raise ValueError(f"estimated_flops_g must be positive, got {estimated_flops_g}")
        self.estimated_flops_g = float(estimated_flops_g)

        if memory_footprint_mb <= 0.0:
            raise ValueError(f"memory_footprint_mb must be positive, got {memory_footprint_mb}")
        self.memory_footprint_mb = float(memory_footprint_mb)

        if calibrated_base_latency_ms <= 0.0:
            raise ValueError(
                f"calibrated_base_latency_ms must be positive, got {calibrated_base_latency_ms}"
            )
        self.calibrated_base_latency_ms = float(calibrated_base_latency_ms)

        if latency_per_batch_item_ms < 0.0:
            raise ValueError(
                f"latency_per_batch_item_ms must be non-negative, got {latency_per_batch_item_ms}"
            )
        self.latency_per_batch_item_ms = float(latency_per_batch_item_ms)

    def estimate_execution_time(self, batch_size: int = 1) -> float:
        """Estimates total GPU execution duration for a given batch size."""
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be positive integer, got {batch_size}")
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"batch_size ({batch_size}) exceeds workload maximum ({self.max_batch_size})"
            )

        if batch_size == 1:
            return self.calibrated_base_latency_ms
        return self.calibrated_base_latency_ms + (batch_size - 1) * self.latency_per_batch_item_ms

    def estimate_memory_mb(self, batch_size: int = 1) -> float:
        """Estimates total memory allocation for a given batch size."""
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be positive integer, got {batch_size}")
        return self.memory_footprint_mb * (1.0 + 0.1 * (batch_size - 1))

    def create_job(
        self,
        job_id: str,
        tenant_id: str,
        arrival_time: float,
        batch_size: Optional[int] = None,
        priority: int = 1,
        deadline: Optional[float] = None,
    ) -> GPUJob:
        """Factory method creating a calibrated GPUJob based on this workload profile."""
        b = batch_size if batch_size is not None else self.default_batch_size
        est_gpu_time = self.estimate_execution_time(b)
        est_mem = self.estimate_memory_mb(b)

        return GPUJob(
            job_id=job_id,
            tenant_id=tenant_id,
            model_name=self.model_name,
            arrival_time=arrival_time,
            estimated_gpu_time=est_gpu_time,
            remaining_gpu_time=est_gpu_time,
            priority=priority,
            deadline=deadline,
            workload_type=self.workload_type,
            batch_size=b,
            input_shape=self.input_shape,
            predicted_memory_mb=est_mem,
            metadata={
                "domain": self.domain,
                "flops_g": self.estimated_flops_g,
            },
        )

    @property
    def category(self) -> str:
        """String category identifier for backward compatibility."""
        return self.workload_type.value


# Backward compatibility alias
WorkloadModelProfile = Workload
