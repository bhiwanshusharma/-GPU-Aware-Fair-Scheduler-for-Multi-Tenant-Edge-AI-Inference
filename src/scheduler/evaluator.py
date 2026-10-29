"""Unified benchmark runner and evaluation metrics extractor for scheduling policies."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import copy

from src.core.job import GPUJob
from src.core.tenant import Tenant
from src.scheduler.base import BaseScheduler
from src.telemetry.metrics import compute_jains_fairness_index, compute_percentiles


@dataclass
class SchedulerEvaluationResult:
    """Consolidated performance evaluation report for a scheduled experiment."""
    scheduler_name: str
    total_simulation_time_ms: float
    completed_jobs_count: int
    total_jobs_submitted: int
    execution_trace: str
    waiting_time_mean_ms: float
    waiting_time_max_ms: float
    response_time_mean_ms: float
    turnaround_time_mean_ms: float
    throughput_jobs_per_sec: float
    gpu_utilization_pct: float
    per_tenant_gpu_service_ms: Dict[str, float] = field(default_factory=dict)
    per_tenant_completed_jobs: Dict[str, int] = field(default_factory=dict)
    jains_fairness_index: float = 1.0
    completed_jobs: List[GPUJob] = field(default_factory=list)

    def summary_table(self) -> str:
        """Renders formatted ASCII summary table of the evaluation result."""
        lines = [
            f"=== Evaluation Summary: {self.scheduler_name} ===",
            f"Total Simulation Time : {self.total_simulation_time_ms:.1f} ms",
            f"Completed Jobs        : {self.completed_jobs_count} / {self.total_jobs_submitted}",
            f"Throughput            : {self.throughput_jobs_per_sec:.2f} jobs/sec",
            f"GPU Utilization       : {self.gpu_utilization_pct:.1f}%",
            f"Jain's Fairness Index : {self.jains_fairness_index:.4f}",
            f"Mean Waiting Time     : {self.waiting_time_mean_ms:.2f} ms (Max: {self.waiting_time_max_ms:.2f} ms)",
            f"Mean Response Time    : {self.response_time_mean_ms:.2f} ms",
            f"Mean Turnaround Time  : {self.turnaround_time_mean_ms:.2f} ms",
            "Per-Tenant GPU Service Allocation:",
        ]
        for t_id, service_ms in self.per_tenant_gpu_service_ms.items():
            count = self.per_tenant_completed_jobs.get(t_id, 0)
            lines.append(f"  - Tenant '{t_id:<18}': {service_ms:>7.1f} ms ({count} jobs completed)")
        return "\n".join(lines)


def run_scheduler_benchmark(
    scheduler: BaseScheduler,
    jobs: List[GPUJob],
    quantum_ms: float = 10.0,
    context_switch_overhead_ms: float = 0.0,
) -> SchedulerEvaluationResult:
    """Executes a workload against a scheduling algorithm using the discrete GPU simulator."""
    # Reset scheduler and queues for fresh experiment run
    scheduler.reset()

    # Create simulator bound to this scheduler
    from src.simulation.simulator import GPUSimulator
    sim = GPUSimulator(
        quantum_ms=quantum_ms,
        context_switch_overhead_ms=context_switch_overhead_ms,
        scheduler=scheduler,
    )

    # Submit deep copies of jobs so original inputs are not mutated across benchmark runs
    cloned_jobs = [copy.deepcopy(j) for j in jobs]
    for j in cloned_jobs:
        sim.submit_job(j)

    # Run discrete event simulation to completion
    sim.run()

    completed = sim.completed_jobs
    total_time_ms = max(0.1, sim.current_time_ms)

    # Calculate latency metrics
    wait_times = [j.waiting_time_ms for j in completed]
    response_times = [max(0.0, (j.start_time - j.arrival_time) if j.start_time is not None else 0.0) for j in completed]
    turnaround_times = [j.turnaround_time_ms for j in completed]

    wait_stats = compute_percentiles(wait_times)
    resp_stats = compute_percentiles(response_times)
    turn_stats = compute_percentiles(turnaround_times)

    # Calculate throughput
    throughput = (len(completed) / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0.0

    # Calculate per-tenant service
    per_tenant_service: Dict[str, float] = {t_id: 0.0 for t_id in scheduler.tenants.keys()}
    per_tenant_counts: Dict[str, int] = {t_id: 0 for t_id in scheduler.tenants.keys()}

    for j in completed:
        per_tenant_service[j.tenant_id] = per_tenant_service.get(j.tenant_id, 0.0) + j.actual_gpu_time_ms
        per_tenant_counts[j.tenant_id] = per_tenant_counts.get(j.tenant_id, 0) + 1

    # Calculate Jain's Fairness Index
    nominal_weights = {t_id: t.weight for t_id, t in scheduler.tenants.items()}
    jains_index = compute_jains_fairness_index(per_tenant_service, nominal_weights)

    return SchedulerEvaluationResult(
        scheduler_name=scheduler.name,
        total_simulation_time_ms=total_time_ms,
        completed_jobs_count=len(completed),
        total_jobs_submitted=len(jobs),
        execution_trace=sim.get_execution_trace_text(include_idle=False),
        waiting_time_mean_ms=wait_stats["mean"],
        waiting_time_max_ms=wait_stats["max"],
        response_time_mean_ms=resp_stats["mean"],
        turnaround_time_mean_ms=turn_stats["mean"],
        throughput_jobs_per_sec=throughput,
        gpu_utilization_pct=sim.gpu_utilization_pct,
        per_tenant_gpu_service_ms=per_tenant_service,
        per_tenant_completed_jobs=per_tenant_counts,
        jains_fairness_index=jains_index,
        completed_jobs=completed,
    )
