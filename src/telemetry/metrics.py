"""Mathematical definitions and metric computation for scheduler performance evaluation."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np


@dataclass
class TenantPerformanceMetrics:
    """Consolidated performance telemetry for a single tenant."""
    tenant_id: str
    weight: float
    priority: int
    submitted_jobs: int
    completed_jobs: int
    dropped_jobs: int
    allocated_gpu_time_ms: float
    mean_waiting_time_ms: float
    p95_waiting_time_ms: float
    max_waiting_time_ms: float
    mean_turnaround_time_ms: float
    p95_turnaround_time_ms: float
    mean_execution_time_ms: float
    sla_violation_count: int
    sla_violation_rate_pct: float


@dataclass
class BenchmarkMetrics:
    """Consolidated system-wide benchmark metrics."""
    policy_name: str
    total_simulation_time_ms: float
    total_jobs_submitted: int
    total_jobs_completed: int
    total_jobs_dropped: int
    throughput_req_per_sec: float
    overall_gpu_utilization_pct: float
    jains_fairness_index: float
    mean_system_waiting_time_ms: float
    p95_system_waiting_time_ms: float
    mean_system_turnaround_time_ms: float
    p95_system_turnaround_time_ms: float
    overall_sla_violation_rate_pct: float
    tenant_breakdown: Dict[str, TenantPerformanceMetrics] = field(default_factory=dict)


def compute_jains_fairness_index(
    allocated_gpu_times: Dict[str, float],
    nominal_weights: Optional[Dict[str, float]] = None,
) -> float:
    """Computes Jain's Fairness Index.
    
    If nominal_weights is None:
        J = (sum(x_i)^2) / (n * sum(x_i^2))
        where x_i represents the GPU service received by tenant i.
    If nominal_weights is provided:
        Computes weighted proportional fairness:
        x_i = allocated_time_i / (w_i * total_time)
    
    Returns:
        Float in [1/n, 1.0], where 1.0 denotes ideal fairness.
    """
    if nominal_weights is None:
        from src.telemetry.metrics_engine import compute_jains_fairness_unweighted
        return compute_jains_fairness_unweighted(allocated_gpu_times)

    active_tenants = [t for t in allocated_gpu_times if nominal_weights.get(t, 0.0) > 0.0]
    n = len(active_tenants)
    if n <= 1:
        return 1.0

    total_allocated = sum(allocated_gpu_times[t] for t in active_tenants)
    if total_allocated <= 0.0:
        return 1.0

    # Calculate normalized shares x_i = actual_share / nominal_weight
    x: List[float] = []
    for t in active_tenants:
        w_i = nominal_weights[t]
        actual_share = allocated_gpu_times[t] / total_allocated
        normalized_ratio = actual_share / w_i
        x.append(normalized_ratio)

    sum_x = sum(x)
    sum_x_sq = sum(val * val for val in x)

    if sum_x_sq <= 0.0:
        return 1.0

    jain_index = (sum_x * sum_x) / (n * sum_x_sq)
    # Bound between 1/n and 1.0 due to floating point precision
    return float(np.clip(jain_index, 1.0 / n, 1.0))


def compute_percentiles(values: List[float]) -> Dict[str, float]:
    """Computes mean, median, p90, p95, p99, and max from a series of measurements."""
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }

    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }
