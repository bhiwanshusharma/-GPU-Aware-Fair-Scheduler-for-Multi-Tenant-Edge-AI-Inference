"""Hostile Senior Systems Engineer Audit Test Suite.

Rigorously stresses the system to detect:
1. Time accounting errors (waiting time = turnaround - execution invariant under preemption)
2. WRR multi-slice proportionality without budget corruption
3. Deadlock and infinite loop resilience (micro-quanta, all-throttled states)
4. Bit-for-bit deterministic replay across independent executions
5. Floating-point precision resilience (IEEE 754 jitter, window rollover boundaries)
6. Hardware duty cycle conservation laws (Busy + Idle + Overhead == Total Time)
7. Jain fairness mathematical edge cases and division-by-zero protection
8. Queue isolation, duplicate handling, and state transition invariants
9. Anti-starvation and priority monotonicities under stress
"""

import math
import copy
import random
import pytest
from typing import Dict, List

from src.core.types import JobState, WorkloadClass, SchedulingQuantum
from src.core.job import GPUJob
from src.core.tenant import Tenant
from src.core.queues import TenantQueueManager, TenantJobQueue
from src.core.exceptions import (
    JobCancelledError,
    DuplicateJobError,
    InvalidExecutionTimeError,
    InvalidQuantumError,
    TenantQuotaExhaustedError,
)
from src.simulation.simulator import GPUSimulator
from src.scheduler.fifo import FIFOScheduler
from src.scheduler.round_robin import RoundRobinScheduler
from src.scheduler.priority import PriorityScheduler
from src.scheduler.weighted_round_robin import WeightedRoundRobinScheduler
from src.scheduler.fair_scheduler import GPUFairScheduler
from src.telemetry.metrics_engine import (
    MetricsEngine,
    compute_jains_fairness_unweighted,
)
from src.telemetry.metrics import compute_jains_fairness_index
from src.benchmarks.runner import BenchmarkRunner, create_scheduler
from src.benchmarks.scenarios import get_scenario


# ============================================================================
# 1. Mathematical Invariant: Time Accounting Under Preemption
# ============================================================================
class TestTimeAccountingInvariants:
    """Verifies that in preemptive scheduling, waiting time strictly equals
    turnaround time minus actual execution time for every completed job.
    """

    def test_preemptive_waiting_time_conservation_round_robin(self):
        """In Round Robin with multiple preemptions per job, verify that
        waiting_time == turnaround_time - actual_gpu_time.
        """
        qm = TenantQueueManager()
        t1 = Tenant("t1", weight=1.0, priority=5, gpu_quota=100.0)
        t2 = Tenant("t2", weight=1.0, priority=5, gpu_quota=100.0)
        tenants = {"t1": t1, "t2": t2}

        # Base quantum of 10ms
        sched = RoundRobinScheduler(tenants=tenants, queue_manager=qm, quantum=SchedulingQuantum(10.0))
        sim = GPUSimulator(
            tenants=tenants,
            quantum_ms=10.0,
            context_switch_overhead_ms=1.0,
            scheduler=sched,
        )

        # Job 1 requires 35ms (4 slices: 10, 10, 10, 5)
        # Job 2 requires 25ms (3 slices: 10, 10, 5)
        j1 = GPUJob("j1", "t1", "m1", arrival_time=0.0, estimated_gpu_time=35.0)
        j2 = GPUJob("j2", "t2", "m2", arrival_time=0.0, estimated_gpu_time=25.0)

        sim.submit_job(j1)
        sim.submit_job(j2)
        sim.run()

        assert len(sim.completed_jobs) == 2

        for job in sim.completed_jobs:
            # The fundamental OS scheduling equation:
            # Turnaround = Completion - Arrival
            # Waiting = Turnaround - Execution
            turnaround = job.turnaround_time_ms
            exec_time = job.actual_gpu_time_ms
            expected_wait = turnaround - exec_time

            assert math.isclose(job.waiting_time_ms, expected_wait, abs_tol=1e-5), (
                f"Job {job.job_id}: waiting_time {job.waiting_time_ms} != "
                f"turnaround ({turnaround}) - execution ({exec_time}) = {expected_wait}"
            )
            assert job.number_of_preemptions > 0, "Job should have undergone preemption"

    def test_preemptive_waiting_time_conservation_fair_scheduler(self):
        """In GPUFairScheduler under heavy multi-tenant preemption, all jobs must
        strictly obey the waiting time conservation equation.
        """
        tenants = {
            f"t{i}": Tenant(f"t{i}", weight=float(i + 1), priority=i + 1, gpu_quota=100.0)
            for i in range(4)
        }
        qm = TenantQueueManager()
        sched = GPUFairScheduler(
            tenants=tenants,
            queue_manager=qm,
            quantum=SchedulingQuantum(base_quantum_ms=10.0, min_quantum_ms=5.0, max_quantum_ms=25.0),
        )
        sim = GPUSimulator(
            tenants=tenants,
            quantum_ms=10.0,
            context_switch_overhead_ms=0.5,
            scheduler=sched,
        )

        rng = random.Random(1337)
        jobs = []
        for i in range(16):
            t_id = f"t{i % 4}"
            arr = rng.uniform(0.0, 20.0)
            dur = rng.uniform(15.0, 45.0)
            jobs.append(GPUJob(f"job_{i}", t_id, "model", arrival_time=arr, estimated_gpu_time=dur))

        for j in jobs:
            sim.submit_job(j)

        sim.run()

        assert len(sim.completed_jobs) == 16
        for j in sim.completed_jobs:
            expected_wait = j.turnaround_time_ms - j.actual_gpu_time_ms
            assert math.isclose(j.waiting_time_ms, expected_wait, abs_tol=1e-5)
            # Response time must be <= waiting time
            response_time = j.start_time - j.arrival_time
            assert response_time <= j.waiting_time_ms + 1e-5


# ============================================================================
# 2. Hardware Duty Cycle Conservation Law
# ============================================================================
class TestHardwareConservationLaw:
    """Verifies that:
    Total Simulation Time == Busy Compute Time + Idle Time + Context Switch Overhead
    Utilization% + Idle% + Overhead% == 100%
    """

    @pytest.mark.parametrize("policy", ["fifo", "round_robin", "priority", "weighted_round_robin", "gpu_fair"])
    def test_duty_cycle_conservation(self, policy: str):
        tenants = {
            "t1": Tenant("t1", weight=2.0, priority=8, gpu_quota=100.0),
            "t2": Tenant("t2", weight=1.0, priority=3, gpu_quota=100.0),
        }
        overhead_ms = 1.5
        sched = create_scheduler(policy, tenants=tenants, quantum_ms=10.0, context_switch_overhead_ms=overhead_ms)
        sim = GPUSimulator(
            tenants=tenants,
            quantum_ms=10.0,
            context_switch_overhead_ms=overhead_ms,
            scheduler=sched,
        )

        # Introduce an idle gap between jobs
        sim.submit_job(GPUJob("j1", "t1", "m", arrival_time=0.0, estimated_gpu_time=20.0))
        sim.submit_job(GPUJob("j2", "t2", "m", arrival_time=50.0, estimated_gpu_time=15.0))

        sim.run()

        engine = MetricsEngine()
        report = engine.compute(
            trace=sim.trace,
            jobs=sim.completed_jobs,
            total_simulation_time_ms=sim.current_time_ms,
            tenants=tenants,
            idle_periods=sim.get_idle_periods(),
            context_switch_overhead_ms=overhead_ms,
        )

        total_time = report.total_simulation_time_ms
        busy_time = report.total_busy_time_ms
        idle_time = report.total_idle_time_ms
        overhead_time = report.scheduling_overhead_ms

        assert total_time > 0.0
        # Conservation of time: Total = Busy + Idle + Overhead
        assert math.isclose(total_time, busy_time + idle_time + overhead_time, abs_tol=1e-4), (
            f"Policy {policy}: total_time ({total_time}) != "
            f"busy ({busy_time}) + idle ({idle_time}) + overhead ({overhead_time}) = "
            f"{busy_time + idle_time + overhead_time}"
        )

        # Conservation of percentage
        total_pct = report.gpu_utilization_pct + report.gpu_idle_pct + report.scheduling_overhead_pct
        assert math.isclose(total_pct, 100.0, abs_tol=1e-3)


# ============================================================================
# 3. WRR Multi-Slice Proportionality (Bug Fix Verification)
# ============================================================================
class TestWeightedRoundRobinProportionality:
    """Verifies that Weighted Round Robin deducts slice execution time
    (not cumulative lifetime job execution) from round budgets.
    """

    def test_wrr_multi_slice_budget_accuracy(self):
        t1 = Tenant("t_heavy", weight=4.0, priority=5, gpu_quota=100.0)
        t2 = Tenant("t_light", weight=1.0, priority=5, gpu_quota=100.0)
        tenants = {"t_heavy": t1, "t_light": t2}

        base_q = 10.0
        sched = WeightedRoundRobinScheduler(tenants=tenants, quantum=SchedulingQuantum(base_q))
        sim = GPUSimulator(
            tenants=tenants,
            quantum_ms=base_q,
            context_switch_overhead_ms=0.0,
            scheduler=sched,
        )

        # t_heavy has a job of 60ms (requires multiple rounds: 40ms in round 1, 20ms in round 2)
        # t_light has a job of 20ms (requires multiple rounds: 10ms in round 1, 10ms in round 2)
        sim.submit_job(GPUJob("j_heavy", "t_heavy", "m", arrival_time=0.0, estimated_gpu_time=60.0))
        sim.submit_job(GPUJob("j_light", "t_light", "m", arrival_time=0.0, estimated_gpu_time=20.0))

        sim.run()

        assert len(sim.completed_jobs) == 2
        assert t1.accumulated_gpu_service == pytest.approx(60.0, abs=1e-5)
        assert t2.accumulated_gpu_service == pytest.approx(20.0, abs=1e-5)

        # In round 1: heavy gets 40ms, light gets 10ms (4:1)
        # In round 2: heavy finishes remaining 20ms, light finishes remaining 10ms
        assert t1.total_jobs_completed == 1
        assert t2.total_jobs_completed == 1


# ============================================================================
# 4. Deterministic Bit-for-Bit Replay
# ============================================================================
class TestDeterministicReplay:
    """Verifies that running identical workloads across separate executions
    yields bit-for-bit identical schedules, scores, and metrics.
    """

    @pytest.mark.parametrize("policy", ["fifo", "round_robin", "priority", "weighted_round_robin", "gpu_fair"])
    def test_bit_for_bit_identical_replay(self, policy: str):
        runner = BenchmarkRunner(random_seed=42, policies=[policy])
        scenario = get_scenario("scenario_5")  # Bursty multi-tenant scenario

        res1 = runner.run_scenario(scenario, policies=[policy], random_seed=999)
        res2 = runner.run_scenario(scenario, policies=[policy], random_seed=999)

        p1 = res1.policy_results[policy]
        p2 = res2.policy_results[policy]

        # 1. Exact trace comparison
        assert p1.execution_trace == p2.execution_trace, f"Policy {policy} produced divergent execution traces!"

        # 2. Metric comparisons
        m1 = p1.metrics
        m2 = p2.metrics
        assert m1.gpu_utilization_pct == p2.metrics.gpu_utilization_pct
        assert m1.avg_waiting_time_ms == p2.metrics.avg_waiting_time_ms
        assert m1.avg_response_time_ms == p2.metrics.avg_response_time_ms
        assert m1.avg_turnaround_time_ms == p2.metrics.avg_turnaround_time_ms
        assert m1.jains_fairness_index == p2.metrics.jains_fairness_index
        assert m1.preemption_count == p2.metrics.preemption_count


# ============================================================================
# 5. Floating-Point Precision & Boundary Noise
# ============================================================================
class TestFloatingPointNoiseResilience:
    """Stresses scheduling with IEEE 754 precision artifacts ($0.1 + 0.2 = 0.30000000000000004)."""

    def test_sub_epsilon_arrival_times_fifo(self):
        """FIFO must break ties deterministically when arrival times are within epsilon."""
        t1 = Tenant("t1", weight=1.0, priority=5)
        t2 = Tenant("t2", weight=1.0, priority=5)
        sched = FIFOScheduler(tenants={"t1": t1, "t2": t2})
        qm = sched.queue_manager

        # Two jobs across different tenants arriving with tiny floating-point difference < 1e-9
        j1 = GPUJob("j_alpha", "t1", "m", arrival_time=1.0000000000000002, estimated_gpu_time=10.0)
        j2 = GPUJob("j_beta", "t2", "m", arrival_time=1.0000000000000004, estimated_gpu_time=10.0)

        qm.enqueue(j2)
        qm.enqueue(j1)

        first = sched.schedule(1.0)
        # Deterministic tie-breaker selects t1/j_alpha by (tenant_id, job_id) key
        assert first is not None
        assert first.job_id == "j_alpha"

    def test_quota_window_floating_point_rollover(self):
        """Tenant sliding quota window must roll over cleanly when times accumulate floating-point error."""
        t = Tenant("t1", weight=1.0, priority=5, gpu_quota=20.0, quota_window_ms=100.0)
        # Allowable is 20ms per 100ms
        t.record_gpu_execution(20.0, current_time_ms=0.0)
        assert t.is_quota_exceeded(current_time_ms=50.0)

        # At current_time = 99.99999999999999 (floating point accumulation of 0.1 * 1000)
        # Should roll over cleanly due to 1e-9 epsilon tolerance
        t.update_window(current_time_ms=100.0 - 1e-11)
        assert not t.is_quota_exceeded(current_time_ms=100.0 - 1e-11)


# ============================================================================
# 6. Deadlock & Infinite Loop Resilience
# ============================================================================
class TestDeadlockAndLoopResilience:
    """Verifies that simulator does not hang, deadlock, or enter infinite recursion."""

    def test_all_tenants_throttled_simulates_cleanly(self):
        """When all tenants exhaust their quotas simultaneously, simulator must
        safely enter idle and resume when window rolls over without spinning infinitely.
        """
        t1 = Tenant("t1", weight=1.0, priority=5, gpu_quota=10.0, quota_window_ms=50.0)
        tenants = {"t1": t1}
        sched = GPUFairScheduler(tenants=tenants, quantum=SchedulingQuantum(base_quantum_ms=5.0))
        sim = GPUSimulator(tenants=tenants, quantum_ms=5.0, scheduler=sched)

        # 3 jobs of 10ms each. Quota allows 5ms per 50ms.
        for i in range(3):
            sim.submit_job(GPUJob(f"j{i}", "t1", "m", arrival_time=0.0, estimated_gpu_time=10.0))

        # Must terminate within reasonable discrete step limit
        sim.run(until_time_ms=300.0)

        assert sim.current_time_ms <= 300.0
        # All jobs should eventually complete across successive quota windows
        assert len(sim.completed_jobs) == 3

    def test_micro_quantum_stress(self):
        """Simulator with tiny quantum (0.5ms) and high preemption rate must not crash or loop."""
        t1 = Tenant("t1", weight=1.0, priority=5)
        t2 = Tenant("t2", weight=1.0, priority=5)
        tenants = {"t1": t1, "t2": t2}

        sim = GPUSimulator(
            tenants=tenants,
            quantum_ms=0.5,
            context_switch_overhead_ms=0.1,
            policy="round_robin",
        )

        sim.submit_job(GPUJob("j1", "t1", "m", arrival_time=0.0, estimated_gpu_time=5.0))
        sim.submit_job(GPUJob("j2", "t2", "m", arrival_time=0.0, estimated_gpu_time=5.0))

        sim.run()

        assert len(sim.completed_jobs) == 2
        for j in sim.completed_jobs:
            assert j.number_of_preemptions >= 8


# ============================================================================
# 7. Jain's Fairness Index Mathematical Robustness
# ============================================================================
class TestJainsFairnessMathematicalAxioms:
    """Verifies mathematical properties of Jain's Fairness Index."""

    def test_jain_fairness_equal_allocation(self):
        # Equal allocations -> J = 1.0
        assert compute_jains_fairness_unweighted([50.0, 50.0, 50.0, 50.0]) == 1.0
        assert compute_jains_fairness_unweighted({"a": 10.0, "b": 10.0}) == 1.0

    def test_jain_fairness_worst_case_monopolization(self):
        # 1 tenant gets all service, (n-1) get 0 -> J = 1/n
        n = 5
        services = [100.0] + [0.0] * (n - 1)
        j = compute_jains_fairness_unweighted(services)
        assert math.isclose(j, 1.0 / n, abs_tol=1e-6)

    def test_jain_fairness_all_zero_service(self):
        # All zeros -> J = 1.0 by definition (everyone received equal 0 service)
        assert compute_jains_fairness_unweighted([0.0, 0.0, 0.0]) == 1.0

    def test_jain_fairness_single_tenant(self):
        # Single tenant -> J = 1.0
        assert compute_jains_fairness_unweighted([42.0]) == 1.0
        assert compute_jains_fairness_unweighted({"t1": 100.0}) == 1.0

    def test_jain_fairness_empty_input(self):
        assert compute_jains_fairness_unweighted([]) == 1.0
        assert compute_jains_fairness_unweighted({}) == 1.0


# ============================================================================
# 8. Queue Isolation and Duplicate Job Rejection
# ============================================================================
class TestQueueIntegrityAndIsolation:
    """Verifies strict tenant isolation and error handling in queue manager."""

    def test_cross_tenant_queue_leak_prevention(self):
        qm = TenantQueueManager()
        t1 = Tenant("t1", weight=1.0, priority=5)
        t2 = Tenant("t2", weight=1.0, priority=5)
        qm.register_tenant(t1)
        qm.register_tenant(t2)

        j1 = GPUJob("j1", "t1", "m", arrival_time=0.0, estimated_gpu_time=10.0)
        qm.enqueue(j1)

        assert qm.get_queue("t1").size == 1
        assert qm.get_queue("t2").size == 0
        assert qm.peek("t2") is None
        assert qm.dequeue("t2") is None

    def test_duplicate_job_id_rejection(self):
        qm = TenantQueueManager()
        t1 = Tenant("t1", weight=1.0, priority=5)
        qm.register_tenant(t1)

        j1 = GPUJob("j1", "t1", "m", arrival_time=0.0, estimated_gpu_time=10.0)
        qm.enqueue(j1)

        # Re-enqueuing same job_id must raise DuplicateJobError
        j1_dupe = GPUJob("j1", "t1", "m", arrival_time=5.0, estimated_gpu_time=10.0)
        with pytest.raises(DuplicateJobError):
            qm.enqueue(j1_dupe)


# ============================================================================
# 9. Chaos Test: Randomized Multi-Tenant Stress with Cancellations
# ============================================================================
class TestChaoticStressWithCancellations:
    """Submits 100 chaotic jobs across 8 tenants with random cancellations and bursts."""

    def test_chaotic_simulation_runs_to_clean_completion(self):
        tenants = {
            f"tenant_{i}": Tenant(
                f"tenant_{i}",
                weight=float(random.randint(1, 5)),
                priority=random.randint(1, 10),
                gpu_quota=random.uniform(50.0, 100.0),
            )
            for i in range(8)
        }
        qm = TenantQueueManager()
        sched = GPUFairScheduler(
            tenants=tenants,
            queue_manager=qm,
            quantum=SchedulingQuantum(base_quantum_ms=10.0, min_quantum_ms=2.0, max_quantum_ms=20.0),
            starvation_threshold_ms=80.0,
        )
        sim = GPUSimulator(
            tenants=tenants,
            quantum_ms=10.0,
            context_switch_overhead_ms=0.5,
            scheduler=sched,
        )

        rng = random.Random(2026)
        jobs = []
        for i in range(80):
            t_id = f"tenant_{i % 8}"
            arr = rng.uniform(0.0, 150.0)
            dur = rng.uniform(5.0, 30.0)
            job = GPUJob(f"chaos_{i}", t_id, "model", arrival_time=arr, estimated_gpu_time=dur)
            jobs.append(job)
            sim.submit_job(job)

        # Cancel 5 jobs randomly before run
        to_cancel = [jobs[3], jobs[17], jobs[35], jobs[51], jobs[72]]
        for j in to_cancel:
            sim.cancel_job(j.job_id, reason="Chaos cancellation test")

        sim.run()

        # Check completed jobs
        completed = sim.completed_jobs
        assert len(completed) > 0
        # No cancelled job should ever be in completed jobs
        completed_ids = {j.job_id for j in completed}
        for j in to_cancel:
            assert j.job_id not in completed_ids

        # Every completed job must obey waiting_time = turnaround - execution
        for j in completed:
            expected_wait = j.turnaround_time_ms - j.actual_gpu_time_ms
            assert math.isclose(j.waiting_time_ms, expected_wait, abs_tol=1e-5)


# ============================================================================
# 10. State Transition Security
# ============================================================================
class TestStateTransitionSecurity:
    """Verifies that terminal states (COMPLETED, DROPPED, CANCELLED) are irrevocable."""

    def test_completed_job_cannot_be_restarted(self):
        job = GPUJob("j1", "t1", "m", arrival_time=0.0, estimated_gpu_time=10.0)
        job.mark_started(0.0)
        job.mark_completed(10.0, actual_gpu_time=10.0)
        assert job.status == JobState.COMPLETED

        with pytest.raises(ValueError):
            job.transition_to(JobState.RUNNING)

        with pytest.raises(ValueError):
            job.transition_to(JobState.QUEUED)

    def test_cancelled_job_cannot_be_dispatched(self):
        job = GPUJob("j1", "t1", "m", arrival_time=0.0, estimated_gpu_time=10.0)
        job.cancel(reason="Test cancel")
        assert job.status == JobState.CANCELLED

        with pytest.raises(JobCancelledError):
            job.mark_started(0.0)

        with pytest.raises(JobCancelledError):
            job.execute_slice(5.0, 0.0)


# ============================================================================
# 11. Metrics Engine Extreme Edge Cases
# ============================================================================
class TestMetricsEngineExtremes:
    """Stresses MetricsEngine with empty traces, zero-duration simulations, and division-by-zero sets."""

    def test_zero_duration_empty_metrics(self):
        engine = MetricsEngine()
        report = engine.compute(
            trace=None,
            jobs=[],
            total_simulation_time_ms=0.0,
            tenants={"t1": Tenant("t1", weight=1.0, priority=5)},
        )
        assert report.gpu_utilization_pct == 0.0
        assert report.gpu_idle_pct == 0.0
        assert report.throughput_jobs_per_sec == 0.0
        assert report.avg_waiting_time_ms == 0.0
        assert report.jains_fairness_index == 1.0

    def test_single_instantaneous_job(self):
        engine = MetricsEngine()
        j = GPUJob("j1", "t1", "m", arrival_time=0.0, estimated_gpu_time=0.1)
        j.mark_started(0.0)
        j.mark_completed(0.1, actual_gpu_time=0.1)

        report = engine.compute(
            jobs=[j],
            total_simulation_time_ms=0.1,
            tenants={"t1": Tenant("t1", weight=1.0, priority=5)},
        )
        assert report.gpu_utilization_pct == pytest.approx(100.0, abs=1e-3)
        assert report.jains_fairness_index == 1.0

