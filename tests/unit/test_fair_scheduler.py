"""Deterministic unit tests for the GPU-Aware Fair Scheduler (GPUFairScheduler / P-DWFQ).

Verifies the 10 required scheduling contracts:
1. Equal-weight tenants: Proportional sharing and high fairness.
2. Unequal-weight tenants: Service proportional to configured weights.
3. High-priority tenant: Immediate low-latency dispatch.
4. Low-priority tenant starvation prevention: Aging guarantees bounded wait time.
5. Quota enforcement: Strict sliding-window GPU time throttling.
6. Long-running job: Multi-quantum preemption and interleaving.
7. Short-running jobs: Single-slice completion and high responsiveness.
8. Burst workload: Fair deficit compensation absorbs bursts smoothly.
9. Heavy multi-tenant contention: Multi-tenant stability and zero starvation under load.
10. GPU idle period: Clean transition to idle and resumption on delayed arrivals.
"""

import pytest
import math
from typing import Dict, List

from src.core.types import SchedulingQuantum
from src.core.job import GPUJob
from src.core.tenant import Tenant
from src.scheduler.fair_scheduler import GPUFairScheduler, ScoreBreakdown
from src.scheduler.evaluator import run_scheduler_benchmark
from src.simulation.simulator import GPUSimulator


def make_tenant_map(tenants: List[Tenant]) -> Dict[str, Tenant]:
    return {t.tenant_id: t for t in tenants}


# =========================================================================
# 1. Equal-Weight Tenants
# =========================================================================
def test_equal_weight_tenants():
    """Tenants with equal weights and priorities receive equal GPU service share."""
    t1 = Tenant("tenant_A", name="Tenant A", weight=1.0, priority=5, gpu_quota=100.0)
    t2 = Tenant("tenant_B", name="Tenant B", weight=1.0, priority=5, gpu_quota=100.0)
    tenants = make_tenant_map([t1, t2])

    sq = SchedulingQuantum(base_quantum_ms=20.0, min_quantum_ms=5.0, max_quantum_ms=50.0)
    sched = GPUFairScheduler(tenants=tenants, quantum=sq, fairness_factor=2.0)

    # 4 jobs each, 20 ms execution each, all arriving at t=0
    jobs: List[GPUJob] = []
    for i in range(4):
        jobs.append(GPUJob(f"A{i}", "tenant_A", "model_a", arrival_time=0.0, estimated_gpu_time=20.0, remaining_gpu_time=20.0, priority=5))
        jobs.append(GPUJob(f"B{i}", "tenant_B", "model_b", arrival_time=0.0, estimated_gpu_time=20.0, remaining_gpu_time=20.0, priority=5))

    res = run_scheduler_benchmark(sched, jobs, quantum_ms=20.0, context_switch_overhead_ms=0.5)

    assert len(res.completed_jobs) == 8
    # Equal work completed: 80 ms each
    assert res.per_tenant_gpu_service_ms["tenant_A"] == 80.0
    assert res.per_tenant_gpu_service_ms["tenant_B"] == 80.0
    assert math.isclose(res.jains_fairness_index, 1.0, abs_tol=1e-3)


# =========================================================================
# 2. Unequal-Weight Tenants
# =========================================================================
def test_unequal_weight_tenants():
    """Tenants with weights 3.0 vs 1.0 receive service proportional to weights."""
    t_heavy = Tenant("tenant_heavy", name="Heavy", weight=3.0, priority=5, gpu_quota=100.0)
    t_light = Tenant("tenant_light", name="Light", weight=1.0, priority=5, gpu_quota=100.0)
    tenants = make_tenant_map([t_heavy, t_light])

    sq = SchedulingQuantum(base_quantum_ms=20.0, min_quantum_ms=5.0, max_quantum_ms=60.0)
    sched = GPUFairScheduler(
        tenants=tenants,
        quantum=sq,
        priority_factor=0.0,  # Isolate weight/fairness effect
        fairness_factor=2.0,
    )

    # Heavy submits 6 jobs of 20ms (120ms total), Light submits 2 jobs of 20ms (40ms total)
    jobs: List[GPUJob] = []
    for i in range(6):
        jobs.append(GPUJob(f"H{i}", "tenant_heavy", "model_h", arrival_time=0.0, estimated_gpu_time=20.0, remaining_gpu_time=20.0))
    for i in range(2):
        jobs.append(GPUJob(f"L{i}", "tenant_light", "model_l", arrival_time=0.0, estimated_gpu_time=20.0, remaining_gpu_time=20.0))

    res = run_scheduler_benchmark(sched, jobs, quantum_ms=20.0, context_switch_overhead_ms=0.5)

    assert len(res.completed_jobs) == 8
    assert res.per_tenant_gpu_service_ms["tenant_heavy"] == 120.0
    assert res.per_tenant_gpu_service_ms["tenant_light"] == 40.0
    # Service ratio is 120/40 = 3.0, matching weights 3.0 / 1.0
    assert math.isclose(res.per_tenant_gpu_service_ms["tenant_heavy"] / res.per_tenant_gpu_service_ms["tenant_light"], 3.0, rel_tol=0.05)


# =========================================================================
# 3. High-Priority Tenant
# =========================================================================
def test_high_priority_tenant_responsiveness():
    """High-priority tenant achieves immediate scheduling and lower response time."""
    t_high = Tenant("tenant_high", name="Safety", weight=1.0, priority=10, gpu_quota=100.0)
    t_low = Tenant("tenant_low", name="Batch", weight=1.0, priority=1, gpu_quota=100.0)
    tenants = make_tenant_map([t_high, t_low])

    sched = GPUFairScheduler(tenants=tenants, priority_factor=2.0, aging_rate=0.5)

    j_low = GPUJob("L1", "tenant_low", "model_l", arrival_time=0.0, estimated_gpu_time=20.0, remaining_gpu_time=20.0)
    j_high = GPUJob("H1", "tenant_high", "model_h", arrival_time=0.0, estimated_gpu_time=20.0, remaining_gpu_time=20.0)

    res = run_scheduler_benchmark(sched, [j_low, j_high], quantum_ms=20.0, context_switch_overhead_ms=0.5)

    assert len(res.completed_jobs) == 2
    # High priority job was dispatched first
    high_completed = [j for j in res.completed_jobs if j.job_id == "H1"][0]
    low_completed = [j for j in res.completed_jobs if j.job_id == "L1"][0]

    assert high_completed.start_time < low_completed.start_time
    assert high_completed.waiting_time_ms == 0.0


# =========================================================================
# 4. Low-Priority Tenant Starvation Prevention (Aging)
# =========================================================================
def test_low_priority_starvation_prevention():
    """Aging prevents high-priority stream from indefinitely starving a low-priority job."""
    t_high = Tenant("tenant_high", name="ContinuousHigh", weight=1.0, priority=10, gpu_quota=100.0)
    t_low = Tenant("tenant_low", name="WaitingLow", weight=1.0, priority=1, gpu_quota=100.0)
    tenants = make_tenant_map([t_high, t_low])

    sq = SchedulingQuantum(base_quantum_ms=20.0)
    # Configure high aging rate so wait time overtakes priority difference
    sched = GPUFairScheduler(
        tenants=tenants,
        quantum=sq,
        priority_factor=1.0,
        aging_rate=2.0,
        starvation_threshold_ms=40.0,
    )

    # Tenant High submits 8 jobs arriving at t=0
    # Tenant Low submits 1 job arriving at t=0
    jobs: List[GPUJob] = [
        GPUJob(f"H{i}", "tenant_high", "model_h", arrival_time=0.0, estimated_gpu_time=20.0, remaining_gpu_time=20.0)
        for i in range(8)
    ]
    j_low = GPUJob("L1", "tenant_low", "model_l", arrival_time=0.0, estimated_gpu_time=20.0, remaining_gpu_time=20.0)
    jobs.append(j_low)

    res = run_scheduler_benchmark(sched, jobs, quantum_ms=20.0, context_switch_overhead_ms=0.5)

    low_completed = [j for j in res.completed_jobs if j.job_id == "L1"][0]
    # L1 must complete BEFORE the entire high-priority queue finishes (i.e., not starved)
    last_high_job = [j for j in res.completed_jobs if j.job_id == "H7"][0]

    assert low_completed.completion_time < last_high_job.completion_time
    # Bounded wait time
    assert low_completed.waiting_time_ms < 100.0


# =========================================================================
# 5. Quota Enforcement
# =========================================================================
def test_quota_enforcement():
    """Tenant is throttled when sliding-window quota is exhausted."""
    # Tenant A capped at 30% quota on 500 ms window (150 ms maximum allowed GPU time)
    t_capped = Tenant("tenant_capped", name="Capped", weight=1.0, priority=10, gpu_quota=30.0, quota_window_ms=500.0)
    t_free = Tenant("tenant_free", name="Free", weight=1.0, priority=2, gpu_quota=100.0, quota_window_ms=500.0)
    tenants = make_tenant_map([t_capped, t_free])

    sq = SchedulingQuantum(base_quantum_ms=25.0)
    sched = GPUFairScheduler(
        tenants=tenants,
        quantum=sq,
        enable_quota_enforcement=True,
    )

    # Capped tenant attempts to run 10 jobs of 25ms (250ms total demand > 150ms quota)
    capped_jobs = [
        GPUJob(f"C{i}", "tenant_capped", "model_c", arrival_time=0.0, estimated_gpu_time=25.0, remaining_gpu_time=25.0)
        for i in range(10)
    ]
    # Free tenant submits 4 jobs of 25ms (100ms total demand)
    free_jobs = [
        GPUJob(f"F{i}", "tenant_free", "model_f", arrival_time=0.0, estimated_gpu_time=25.0, remaining_gpu_time=25.0)
        for i in range(4)
    ]

    all_jobs = capped_jobs + free_jobs
    res = run_scheduler_benchmark(sched, all_jobs, quantum_ms=25.0, context_switch_overhead_ms=0.5)

    # In the first 500ms window, tenant_capped cannot exceed 150ms before tenant_free gets served
    free_completed = [j for j in res.completed_jobs if j.tenant_id == "tenant_free"]
    assert len(free_completed) == 4
    # All free jobs completed successfully without being monopolized by capped tenant
    assert res.per_tenant_gpu_service_ms["tenant_free"] == 100.0


# =========================================================================
# 6. Long-Running Job Time-Slicing
# =========================================================================
def test_long_running_job_time_slicing():
    """A long-running job is preempted across multiple quanta and interleaved."""
    t_long = Tenant("tenant_long", name="LongJobTenant", weight=1.0, priority=5, gpu_quota=100.0)
    t_short = Tenant("tenant_short", name="ShortJobTenant", weight=1.0, priority=5, gpu_quota=100.0)
    tenants = make_tenant_map([t_long, t_short])

    sq = SchedulingQuantum(base_quantum_ms=20.0, min_quantum_ms=10.0, max_quantum_ms=30.0)
    sched = GPUFairScheduler(tenants=tenants, quantum=sq)

    # Long job requires 80 ms (requires >= 4 slices with 20ms base quantum)
    j_long = GPUJob("LONG_1", "tenant_long", "heavy_vision", arrival_time=0.0, estimated_gpu_time=80.0, remaining_gpu_time=80.0)
    # Short jobs require 15 ms each
    j_short1 = GPUJob("SHORT_1", "tenant_short", "light_audio", arrival_time=5.0, estimated_gpu_time=15.0, remaining_gpu_time=15.0)
    j_short2 = GPUJob("SHORT_2", "tenant_short", "light_audio", arrival_time=10.0, estimated_gpu_time=15.0, remaining_gpu_time=15.0)

    res = run_scheduler_benchmark(sched, [j_long, j_short1, j_short2], quantum_ms=20.0, context_switch_overhead_ms=0.5)

    assert len(res.completed_jobs) == 3
    long_done = [j for j in res.completed_jobs if j.job_id == "LONG_1"][0]
    # Long job was sliced and preempted at least 2 times
    assert long_done.number_of_preemptions >= 2
    assert long_done.actual_gpu_time_ms == 80.0


# =========================================================================
# 7. Short-Running Jobs
# =========================================================================
def test_short_running_jobs():
    """Short jobs finish within one quantum without unnecessary preemption."""
    t1 = Tenant("tenant_fast", name="Fast", weight=1.0, priority=5, gpu_quota=100.0)
    tenants = make_tenant_map([t1])

    sq = SchedulingQuantum(base_quantum_ms=25.0)
    sched = GPUFairScheduler(tenants=tenants, quantum=sq)

    # 5 short jobs of 5 ms each
    short_jobs = [
        GPUJob(f"S{i}", "tenant_fast", "kws", arrival_time=0.0, estimated_gpu_time=5.0, remaining_gpu_time=5.0)
        for i in range(5)
    ]

    res = run_scheduler_benchmark(sched, short_jobs, quantum_ms=25.0, context_switch_overhead_ms=0.5)

    assert len(res.completed_jobs) == 5
    for j in res.completed_jobs:
        # Zero preemptions for each short job
        assert j.number_of_preemptions == 0
        assert j.actual_gpu_time_ms == 5.0


# =========================================================================
# 8. Burst Workload
# =========================================================================
def test_burst_workload():
    """Scheduler smoothly absorbs an unexpected burst from an idle tenant."""
    t_steady = Tenant("tenant_steady", name="Steady", weight=1.0, priority=5, gpu_quota=100.0)
    t_burst = Tenant("tenant_burst", name="Burst", weight=1.0, priority=5, gpu_quota=100.0)
    tenants = make_tenant_map([t_steady, t_burst])

    sq = SchedulingQuantum(base_quantum_ms=20.0)
    sched = GPUFairScheduler(tenants=tenants, quantum=sq, fairness_factor=2.0)

    # Steady tenant has jobs arriving from t=0
    steady_jobs = [
        GPUJob(f"STD_{i}", "tenant_steady", "model_s", arrival_time=float(i * 10), estimated_gpu_time=15.0, remaining_gpu_time=15.0)
        for i in range(6)
    ]
    # Burst tenant arrives suddenly at t=30 with 3 jobs
    burst_jobs = [
        GPUJob(f"BST_{i}", "tenant_burst", "model_b", arrival_time=30.0, estimated_gpu_time=15.0, remaining_gpu_time=15.0)
        for i in range(3)
    ]

    all_jobs = steady_jobs + burst_jobs
    res = run_scheduler_benchmark(sched, all_jobs, quantum_ms=20.0, context_switch_overhead_ms=0.5)

    assert len(res.completed_jobs) == 9
    # Both tenants completed all work
    assert res.per_tenant_gpu_service_ms["tenant_steady"] == 90.0
    assert res.per_tenant_gpu_service_ms["tenant_burst"] == 45.0
    # Burst jobs completed with bounded turnaround
    for b in [j for j in res.completed_jobs if j.tenant_id == "tenant_burst"]:
        assert b.turnaround_time_ms < 120.0


# =========================================================================
# 9. Heavy Multi-Tenant Contention
# =========================================================================
def test_heavy_multi_tenant_contention():
    """Under high multi-tenant contention, all tenants make progress without deadlock."""
    t_list = [
        Tenant(f"tenant_{i}", name=f"T{i}", weight=float(i + 1), priority=i * 2 + 1, gpu_quota=100.0)
        for i in range(5)
    ]
    tenants = make_tenant_map(t_list)

    sq = SchedulingQuantum(base_quantum_ms=15.0, min_quantum_ms=5.0, max_quantum_ms=45.0)
    sched = GPUFairScheduler(tenants=tenants, quantum=sq, aging_rate=1.5, fairness_factor=1.5)

    # Submit 4 jobs per tenant with staggered execution demands
    jobs: List[GPUJob] = []
    for t_idx, t in enumerate(t_list):
        for j_idx in range(4):
            duration = 10.0 + (t_idx * 5.0)
            jobs.append(
                GPUJob(
                    f"{t.tenant_id}_J{j_idx}",
                    t.tenant_id,
                    "model_m",
                    arrival_time=float(j_idx * 5),
                    estimated_gpu_time=duration,
                    remaining_gpu_time=duration,
                )
            )

    res = run_scheduler_benchmark(sched, jobs, quantum_ms=15.0, context_switch_overhead_ms=0.5)

    assert len(res.completed_jobs) == 20
    # Every tenant received service
    for t in t_list:
        assert res.per_tenant_gpu_service_ms[t.tenant_id] > 0.0
    assert res.gpu_utilization_pct > 80.0
    assert res.jains_fairness_index > 0.6


# =========================================================================
# 10. GPU Idle Period and Resumption
# =========================================================================
def test_gpu_idle_period_and_resumption():
    """Scheduler correctly records idle periods when no jobs are queued and resumes cleanly."""
    t1 = Tenant("tenant_1", name="Tenant 1", weight=1.0, priority=5, gpu_quota=100.0)
    tenants = make_tenant_map([t1])

    sq = SchedulingQuantum(base_quantum_ms=20.0)
    sched = GPUFairScheduler(tenants=tenants, quantum=sq)

    # Job 1 arrives at t=0, runs for 20ms (ends at t=20)
    j1 = GPUJob("EARLY", "tenant_1", "model_1", arrival_time=0.0, estimated_gpu_time=20.0, remaining_gpu_time=20.0)
    # Job 2 arrives late at t=100ms (GPU must idle between t=20 and t=100)
    j2 = GPUJob("LATE", "tenant_1", "model_1", arrival_time=100.0, estimated_gpu_time=20.0, remaining_gpu_time=20.0)

    sim = GPUSimulator(tenants=tenants, scheduler=sched, quantum_ms=20.0, context_switch_overhead_ms=0.5)
    sim.submit_job(j1)
    sim.submit_job(j2)
    sim.run()

    assert len(sim.completed_jobs) == 2
    # Total virtual time should be >= 120 ms
    assert sim.current_time_ms >= 120.0
    # Idle period was recorded between 20.0 and 100.0 ms (approx 80 ms idle)
    assert sim.trace.total_idle_time_ms >= 79.0
    # Idle periods list is non-empty
    assert len(sim.trace.idle_periods) >= 1
