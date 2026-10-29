"""Domain exception hierarchy for GPU-Aware Fair Scheduler.

This module defines specialized, strongly-typed exceptions corresponding to all
critical failure modes across scheduling, queuing, prediction, configuration,
and physical hardware execution.

All custom exceptions inherit from GPUSchedulerError as well as standard Python
built-in exceptions (ValueError, KeyError, RuntimeError, FileNotFoundError, etc.)
to maintain 100% backward compatibility while enabling fail-safe, auditable error handling.
"""

from typing import Optional, Any


class GPUSchedulerError(Exception):
    """Base exception for all scheduler, queuing, and GPU execution errors."""
    pass


# 1. Invalid Tenant Configuration
class InvalidTenantConfigError(ValueError, GPUSchedulerError):
    """Raised when tenant configuration contains invalid parameters (e.g. negative weight, bad quota)."""
    pass


# 2. Invalid GPU Quantum
class InvalidQuantumError(ValueError, GPUSchedulerError):
    """Raised when GPU quantum configuration is non-positive, ill-conditioned, or inverted."""
    pass


# 3 & 4. Zero and Negative GPU Execution Time
class InvalidExecutionTimeError(ValueError, GPUSchedulerError):
    """Raised when job GPU execution time is zero, negative, or invalid."""
    pass


# 5. Empty Queue Operations
class EmptyQueueError(IndexError, GPUSchedulerError):
    """Raised when attempting a strict pop/dequeue from an empty tenant queue."""
    pass


# 6. Tenant Quota Exhaustion
class TenantQuotaExhaustedError(RuntimeError, GPUSchedulerError):
    """Raised when a tenant has exhausted its allocated GPU quota window."""
    pass


# 7. Scheduler Starvation
class SchedulerStarvationError(RuntimeError, GPUSchedulerError):
    """Raised when an inference job's queue latency exceeds critical starvation thresholds."""
    pass


# 8. Predictor Unavailable
class PredictorUnavailableError(RuntimeError, GPUSchedulerError):
    """Raised when an AI workload predictor is required but unavailable or uninitialized."""
    pass


# 9. Invalid Predictor Output
class InvalidPredictorOutputError(ValueError, GPUSchedulerError):
    """Raised when a workload predictor produces NaN, inf, or out-of-range telemetry."""
    pass


# 10. Missing ML Model
class ModelNotFoundError(FileNotFoundError, GPUSchedulerError):
    """Raised when an ML model artifact file is missing, unreadable, or corrupted."""
    pass


# 11. Corrupted Experiment Configuration
class ConfigurationCorruptedError(ValueError, GPUSchedulerError):
    """Raised when a YAML configuration file is malformed, corrupted, or has invalid schema."""
    pass


# 12. CUDA Unavailable
class CUDAUnavailableError(RuntimeError, GPUSchedulerError):
    """Raised when physical CUDA execution is attempted on a host without functional CUDA hardware."""
    pass


# 13. TensorRT Unavailable
class TensorRTUnavailableError(RuntimeError, GPUSchedulerError):
    """Raised when TensorRT acceleration is requested but TensorRT libraries are not installed."""
    pass


# 14. GPU Memory Allocation Failure
class GPUMemoryAllocationError(MemoryError, GPUSchedulerError):
    """Raised when GPU VRAM allocation fails (e.g. CUDA Out of Memory or simulated VRAM overflow)."""
    pass


# 15. Hardware Executor Failure
class HardwareExecutionError(RuntimeError, GPUSchedulerError):
    """Raised when a physical hardware execution failure occurs during kernel dispatch."""
    pass


# 16. Job Cancellation
class JobCancelledError(RuntimeError, GPUSchedulerError):
    """Raised when an execution or scheduling operation is attempted on a cancelled job."""
    pass


# 17. Duplicate Job IDs
class DuplicateJobError(ValueError, GPUSchedulerError):
    """Raised when an ingress job ID duplicates an active or already registered job."""
    pass


# 18. Unknown Tenant IDs
class UnknownTenantError(KeyError, GPUSchedulerError):
    """Raised when a job or query references an unknown or unregistered tenant ID."""
    pass
