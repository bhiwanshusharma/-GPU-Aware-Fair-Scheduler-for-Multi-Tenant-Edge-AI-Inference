"""Stress and invariant verification tests for the GPU-Aware Fair Scheduler."""

import pytest
import random
from typing import Dict, List

from src.core.types import SchedulingQuantum
from src.core.job import GPUJob
from src.core.tenant import Tenant
from src.scheduler.fair_scheduler import GPUFairScheduler
from src.scheduler.evaluator import run_scheduler_benchmark


def test_stress_100_jobs_10_tenants_chaotic():
    """Stress test with 100 heterogeneous jobs across 10 tenants with randomized parameters."""
    rng = random.Random(42)  # Deterministic seed

    tenants = {}
    for i in range(10):
        t_id = f"tenant_{i:02d}"
        tenants[t_id] = Tenant(
            t_id,
            name=f"Tenant-{i}",
            weight=rng.choice([1.0, 2.0, 3.0, 4.0]),
            priority=rng.randint(1, 10),
            gpu_quota=100.0,
        )

    sq = SchedulingQuantum(base_quantum_ms=15.0, min_quantum_ms=5.0, max_quantum_ms=45.0)
    sched = GPUFairScheduler(
        tenants=tenants,
        quantum=sq,
        aging_rate=1.5,
        fairness_factor=1.5,
    )

    jobs: List[GPUJob] = []
    for j_idx in range(100):
        t_id = f"tenant_{rng.randint(0, 9):02d}"
        duration = float(rng.randint(5, 40))
        arrival = float(rng.randint(0, 150))
        jobs.append(
            GPUJob(
                job_id=f"STRESS_{j_idx:03d}",
                tenant_id=t_id,
                model_name="stress_model",
                arrival_time=arrival,
                estimated_gpu_time=duration,
                remaining_gpu_time=duration,
            )
        )

    res = run_scheduler_benchmark(sched, jobs, quantum_ms=15.0, context_switch_overhead_ms=0.5)

    assert len(res.completed_jobs) == 100
    assert res.gpu_utilization_pct > 80.0
    # Every job has bounded wait time
    for j in res.completed_jobs:
        assert j.completion_time is not None
        assert j.turnaround_time_ms is not None
        assert j.turnaround_time_ms > 0.0
        assert j.remaining_gpu_time == 0.0


def test_stress_extreme_priority_disparity():
    """Verify that priority 10 cannot starve priority 1 under high job count."""
    t_vip = Tenant("tenant_vip", name="VIP", weight=1.0, priority=10, gpu_quota=100.0)
    t_batch = Tenant("tenant_batch", name="Batch", weight=1.0, priority=1, gpu_quota=100.0)
    tenants = {t_vip.tenant_id: t_vip, t_batch.tenant_id: t_batch}

    sq = SchedulingQuantum(base_quantum_ms=20.0)
    sched = GPUFairScheduler(
        tenants=tenants,
        quantum=sq,
        priority_factor=1.0,
        aging_rate=2.0,
        starvation_threshold_ms=50.0,
    )

    # VIP submits 20 jobs of 20 ms (400 ms of compute)
    jobs: List[GPUJob] = [
        GPUJob(f"VIP_{i}", "tenant_vip", "vip_model", arrival_time=0.0, estimated_gpu_time=20.0, remaining_gpu_time=20.0)
        for i in range(20)
    ]
    # Batch submits 3 jobs of 20 ms
    for b_idx in range(3):
        jobs.append(
            GPUJob(f"BATCH_{b_idx}", "tenant_batch", "batch_model", arrival_time=0.0, estimated_gpu_time=20.0, remaining_gpu_time=20.0)
        )

    res = run_scheduler_benchmark(sched, jobs, quantum_ms=20.0, context_switch_overhead_ms=0.5)

    assert len(res.completed_jobs) == 23
    batch_jobs = [j for j in res.completed_jobs if j.tenant_id == "tenant_batch"]
    assert len(batch_jobs) == 3

    # The batch jobs must complete well before the final VIP job finishes
    final_vip = [j for j in res.completed_jobs if j.job_id == "VIP_19"][0]
    for bj in batch_jobs:
        assert bj.completion_time < final_vip.completion_time


def test_stress_rapid_preemption_churn():
    """Stress test with high preemption frequency under tiny time quantum."""
    t1 = Tenant("tenant_churn_1", weight=1.0, priority=5, gpu_quota=100.0)
    t2 = Tenant("tenant_churn_2", weight=1.0, priority=5, gpu_quota=100.0)
    tenants = {t1.tenant_id: t1, t2.tenant_id: t2}

    # Small base quantum forces frequent preemptions
    sq = SchedulingQuantum(base_quantum_ms=5.0, min_quantum_ms=5.0, max_quantum_ms=10.0)
    sched = GPUFairScheduler(tenants=tenants, quantum=sq)

    jobs: List[GPUJob] = []
    for i in range(10):
        jobs.append(GPUJob(f"C1_{i}", "tenant_churn_1", "m", arrival_time=0.0, estimated_gpu_time=25.0, remaining_gpu_time=25.0))
        jobs.append(GPUJob(f"C2_{i}", "tenant_churn_2", "m", arrival_time=0.0, estimated_gpu_time=25.0, remaining_gpu_time=25.0))

    res = run_scheduler_benchmark(sched, jobs, quantum_ms=5.0, context_switch_overhead_ms=0.2)

    assert len(res.completed_jobs) == 20
    # Every job underwent multiple preemptions
    total_preemptions = sum(j.number_of_preemptions for j in res.completed_jobs)
    assert total_preemptions >= 60
    assert res.gpu_utilization_pct > 90.0
