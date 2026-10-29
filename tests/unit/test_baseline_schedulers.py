"""Deterministic unit tests for baseline scheduling policies."""

import pytest
from src.core.types import SchedulingQuantum, JobState
from src.core.job import GPUJob
from src.core.tenant import Tenant
from src.core.queues import TenantQueueManager
from src.scheduler.fifo import FIFOScheduler
from src.scheduler.round_robin import RoundRobinScheduler
from src.scheduler.priority import PriorityScheduler
from src.scheduler.weighted_round_robin import WeightedRoundRobinScheduler
from src.scheduler.evaluator import run_scheduler_benchmark, SchedulerEvaluationResult


# ============================================================================
# Test 1: FIFO Behaves Strictly as FIFO
# ============================================================================

def test_fifo_behaves_as_fifo():
    t_a = Tenant("Tenant_A", priority=5, weight=1.0, gpu_quota=50.0)
    t_b = Tenant("Tenant_B", priority=5, weight=1.0, gpu_quota=50.0)
    tenants = {"Tenant_A": t_a, "Tenant_B": t_b}

    scheduler = FIFOScheduler(tenants=tenants)

    # Job A1 arrives at t=0 (duration 20ms)
    # Job B1 arrives at t=2 (duration 10ms)
    # Job A2 arrives at t=5 (duration 15ms)
    jobs = [
        GPUJob("A1", "Tenant_A", "m", arrival_time=0.0, estimated_gpu_time=20.0, remaining_gpu_time=20.0),
        GPUJob("B1", "Tenant_B", "m", arrival_time=2.0, estimated_gpu_time=10.0, remaining_gpu_time=10.0),
        GPUJob("A2", "Tenant_A", "m", arrival_time=5.0, estimated_gpu_time=15.0, remaining_gpu_time=15.0),
    ]

    result = run_scheduler_benchmark(scheduler, jobs, quantum_ms=10.0, context_switch_overhead_ms=0.0)

    # In FIFO: A1 runs 0-20ms, B1 runs 20-30ms, A2 runs 30-45ms
    completed = {j.job_id: j for j in result.completed_jobs}
    assert completed["A1"].start_time == 0.0
    assert completed["A1"].completion_time == 20.0
    assert completed["A1"].number_of_preemptions == 0

    assert completed["B1"].start_time == 20.0
    assert completed["B1"].completion_time == 30.0
    assert completed["B1"].number_of_preemptions == 0

    assert completed["A2"].start_time == 30.0
    assert completed["A2"].completion_time == 45.0
    assert completed["A2"].number_of_preemptions == 0

    # Total time = 45.0ms, GPU utilization = 100%
    assert result.total_simulation_time_ms == 45.0
    assert result.gpu_utilization_pct == 100.0
    assert result.completed_jobs_count == 3


# ============================================================================
# Test 2: Round Robin Respects the GPU Quantum
# ============================================================================

def test_round_robin_respects_quantum():
    quantum_ms = 10.0
    t_a = Tenant("Tenant_A", priority=5, weight=1.0, gpu_quota=50.0)
    t_b = Tenant("Tenant_B", priority=5, weight=1.0, gpu_quota=50.0)
    tenants = {"Tenant_A": t_a, "Tenant_B": t_b}

    sq = SchedulingQuantum(base_quantum_ms=quantum_ms)
    scheduler = RoundRobinScheduler(tenants=tenants, quantum=sq)

    # Both jobs take 25ms
    # Expected interleaved slices: A(10) -> B(10) -> A(10) -> B(10) -> A(5) -> B(5)
    jobs = [
        GPUJob("A1", "Tenant_A", "m", arrival_time=0.0, estimated_gpu_time=25.0, remaining_gpu_time=25.0),
        GPUJob("B1", "Tenant_B", "m", arrival_time=0.0, estimated_gpu_time=25.0, remaining_gpu_time=25.0),
    ]

    result = run_scheduler_benchmark(scheduler, jobs, quantum_ms=quantum_ms, context_switch_overhead_ms=0.0)

    completed = {j.job_id: j for j in result.completed_jobs}
    # Both jobs were preempted exactly 2 times at 10ms slice boundaries
    assert completed["A1"].number_of_preemptions == 2
    assert completed["B1"].number_of_preemptions == 2

    # A1 started at 0.0, finished at 45.0
    # B1 started at 10.0, finished at 50.0
    assert completed["A1"].start_time == 0.0
    assert completed["A1"].completion_time == 45.0
    assert completed["B1"].start_time == 10.0
    assert completed["B1"].completion_time == 50.0

    assert result.total_simulation_time_ms == 50.0
    assert result.gpu_utilization_pct == 100.0


# ============================================================================
# Test 3: Priority Scheduling Respects Priority
# ============================================================================

def test_priority_scheduling_respects_priority():
    # Tenant High has priority 9, Tenant Low has priority 2
    t_high = Tenant("High", priority=9, weight=1.0, gpu_quota=50.0)
    t_low = Tenant("Low", priority=2, weight=1.0, gpu_quota=50.0)
    tenants = {"High": t_high, "Low": t_low}

    scheduler = PriorityScheduler(tenants=tenants, preemptive_quantum=True)

    # Both arrive at t=0
    jobs = [
        GPUJob("L1", "Low", "m", arrival_time=0.0, estimated_gpu_time=20.0, remaining_gpu_time=20.0),
        GPUJob("H1", "High", "m", arrival_time=0.0, estimated_gpu_time=15.0, remaining_gpu_time=15.0),
        GPUJob("H2", "High", "m", arrival_time=0.0, estimated_gpu_time=10.0, remaining_gpu_time=10.0),
    ]

    result = run_scheduler_benchmark(scheduler, jobs, quantum_ms=10.0, context_switch_overhead_ms=0.0)

    # High priority jobs (H1, H2) must finish before Low priority job (L1) gets GPU time!
    completed = {j.job_id: j for j in result.completed_jobs}
    assert completed["H1"].completion_time <= 25.0
    assert completed["H2"].completion_time <= 25.0
    # L1 must complete strictly after all High jobs complete
    assert completed["L1"].completion_time == 45.0
    assert completed["L1"].start_time >= 25.0


# ============================================================================
# Test 4: Weighted Round Robin Provides Proportional Service
# ============================================================================

def test_weighted_round_robin_proportional_service():
    # Tenant A has weight 3.0, Tenant B has weight 1.0 -> 3:1 service ratio
    t_a = Tenant("Tenant_A", priority=5, weight=3.0, gpu_quota=75.0)
    t_b = Tenant("Tenant_B", priority=5, weight=1.0, gpu_quota=25.0)
    tenants = {"Tenant_A": t_a, "Tenant_B": t_b}

    sq = SchedulingQuantum(base_quantum_ms=10.0)
    scheduler = WeightedRoundRobinScheduler(tenants=tenants, quantum=sq)

    # Each tenant submits multiple continuous 10ms jobs
    jobs = []
    # Tenant A: 6 jobs of 10ms each = 60ms total
    for i in range(6):
        jobs.append(GPUJob(f"A_{i}", "Tenant_A", "m", arrival_time=0.0, estimated_gpu_time=10.0, remaining_gpu_time=10.0))
    # Tenant B: 2 jobs of 10ms each = 20ms total
    for i in range(2):
        jobs.append(GPUJob(f"B_{i}", "Tenant_B", "m", arrival_time=0.0, estimated_gpu_time=10.0, remaining_gpu_time=10.0))

    result = run_scheduler_benchmark(scheduler, jobs, quantum_ms=10.0, context_switch_overhead_ms=0.0)

    # Verify per-tenant service
    service_a = result.per_tenant_gpu_service_ms["Tenant_A"]
    service_b = result.per_tenant_gpu_service_ms["Tenant_B"]

    assert service_a == 60.0
    assert service_b == 20.0
    # Ratio = 60 / 20 = 3.0 (matches weight ratio 3.0 : 1.0)
    assert (service_a / service_b) == pytest.approx(3.0, rel=1e-3)
    # Proportional weighted allocation gives high Jain's fairness index
    assert result.jains_fairness_index == pytest.approx(1.0, rel=1e-3)


# ============================================================================
# Test 5: Metrics Production Across All 4 Schedulers
# ============================================================================

@pytest.mark.parametrize("scheduler_cls", [
    FIFOScheduler,
    RoundRobinScheduler,
    PriorityScheduler,
    WeightedRoundRobinScheduler,
])
def test_all_schedulers_produce_required_metrics(scheduler_cls):
    t1 = Tenant("T1", priority=7, weight=2.0, gpu_quota=60.0)
    t2 = Tenant("T2", priority=3, weight=1.0, gpu_quota=40.0)
    tenants = {"T1": t1, "T2": t2}

    scheduler = scheduler_cls(tenants=tenants)

    jobs = [
        GPUJob("j1", "T1", "m", arrival_time=0.0, estimated_gpu_time=12.0, remaining_gpu_time=12.0),
        GPUJob("j2", "T2", "m", arrival_time=0.0, estimated_gpu_time=8.0, remaining_gpu_time=8.0),
    ]

    result = run_scheduler_benchmark(scheduler, jobs, quantum_ms=10.0, context_switch_overhead_ms=0.5)

    # Verify all required output metrics exist and are non-negative
    assert isinstance(result, SchedulerEvaluationResult)
    assert result.completed_jobs_count == 2
    assert result.total_simulation_time_ms > 0.0
    assert result.waiting_time_mean_ms >= 0.0
    assert result.response_time_mean_ms >= 0.0
    assert result.turnaround_time_mean_ms >= 0.0
    assert result.throughput_jobs_per_sec > 0.0
    assert 0.0 < result.gpu_utilization_pct <= 100.0
    assert "T1" in result.per_tenant_gpu_service_ms
    assert "T2" in result.per_tenant_gpu_service_ms
    assert result.per_tenant_gpu_service_ms["T1"] == 12.0
    assert result.per_tenant_gpu_service_ms["T2"] == 8.0
    assert len(result.execution_trace) > 0

    # Summary table formatted string
    summary = result.summary_table()
    assert "Evaluation Summary" in summary
    assert "Per-Tenant GPU Service Allocation" in summary
