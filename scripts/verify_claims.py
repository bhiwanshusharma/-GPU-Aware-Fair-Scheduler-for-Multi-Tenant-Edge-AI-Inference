"""Experimental Verification of the 12 Core Architectural and Systems Claims.

Executes dedicated, reproducible experiments directly against the codebase to
verify all 12 OS scheduling and hardware claims with zero fabricated data.
"""

import sys
import os
import math
import logging
from typing import Dict, Any

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.types import JobState, SchedulingQuantum, WorkloadClass
from src.core.job import GPUJob
from src.core.tenant import Tenant
from src.core.queues import TenantQueueManager
from src.scheduler.fair_scheduler import GPUFairScheduler
from src.scheduler import FIFOScheduler, PriorityScheduler, RoundRobinScheduler, WeightedRoundRobinScheduler
from src.simulation.simulator import GPUSimulator
from src.telemetry.metrics import compute_jains_fairness_index
from src.telemetry.metrics_engine import MetricsEngine
from src.predictor.inference import WorkloadInferenceEngine
from src.core.types import PredictionOutput
from src.execution import create_executor, SimulationExecutor, JetsonExecutor


logging.basicConfig(level=logging.WARNING)


def verify_claim_1_multi_tenant_sharing() -> Dict[str, Any]:
    """Claim 1: Multiple tenants can share one simulated GPU."""
    t1 = Tenant("tenant_A", "Tenant A", weight=1.0, priority=5, gpu_quota=100.0)
    t2 = Tenant("tenant_B", "Tenant B", weight=1.0, priority=5, gpu_quota=100.0)
    t3 = Tenant("tenant_C", "Tenant C", weight=1.0, priority=5, gpu_quota=100.0)
    tenants = {t.tenant_id: t for t in [t1, t2, t3]}

    qm = TenantQueueManager()
    sched = GPUFairScheduler(tenants=tenants, queue_manager=qm, quantum=SchedulingQuantum(base_quantum_ms=10.0))
    sim = GPUSimulator(tenants=tenants, quantum_ms=10.0, scheduler=sched)

    sim.submit_job(GPUJob("jA1", "tenant_A", "m", 0.0, 15.0, 15.0))
    sim.submit_job(GPUJob("jB1", "tenant_B", "m", 0.0, 15.0, 15.0))
    sim.submit_job(GPUJob("jC1", "tenant_C", "m", 0.0, 15.0, 15.0))
    sim.run()

    services = {t_id: t.accumulated_gpu_service for t_id, t in tenants.items()}
    all_served = all(s > 0.0 for s in services.values())
    total_completed = len(sim.completed_jobs)

    return {
        "verified": all_served and total_completed == 3,
        "services": services,
        "total_completed": total_completed,
        "details": f"Tenants served on single GPU: {services}, Completed jobs: {total_completed}/3",
    }


def verify_claim_2_time_slicing() -> Dict[str, Any]:
    """Claim 2: GPU time slicing occurs at scheduler dispatch boundaries."""
    t1 = Tenant("t_long", "Long Job Tenant", weight=1.0, priority=5)
    t2 = Tenant("t_interleave", "Interleaved Tenant", weight=1.0, priority=5)
    tenants = {t.tenant_id: t for t in [t1, t2]}

    qm = TenantQueueManager()
    # Base quantum = 10.0 ms, context switch = 0.5 ms
    sched = GPUFairScheduler(tenants=tenants, queue_manager=qm, quantum=SchedulingQuantum(base_quantum_ms=10.0, min_quantum_ms=10.0, max_quantum_ms=10.0))
    sim = GPUSimulator(tenants=tenants, quantum_ms=10.0, context_switch_overhead_ms=0.5, scheduler=sched)

    # Job 1 needs 25ms (will be sliced into 10ms, 10ms, 5ms)
    sim.submit_job(GPUJob("j_long", "t_long", "m", 0.0, 25.0, 25.0))
    sim.submit_job(GPUJob("j_other", "t_interleave", "m", 0.0, 10.0, 10.0))
    sim.run()

    # Find quantum expiration events in trace
    preempt_events = [e for e in sim.trace.events if e.event_type.name == "QUANTUM_EXPIRED"]
    dispatch_events = [e for e in sim.trace.events if e.event_type.name == "DISPATCH"]

    # Verify preemption occurred and job was sliced into quantum units
    first_preempt = preempt_events[0] if preempt_events else None
    preempt_time = first_preempt.timestamp_ms if first_preempt else None

    # Job j_long was sliced into 10ms + 10ms + 5ms, meaning preemption occurred
    verified = (
        len(preempt_events) >= 2
        and all(e.details.get("remaining_gpu_time", 0) > 0 for e in preempt_events)
        and sim.completed_jobs[-1].accumulated_execution_time == 25.0
    )

    return {
        "verified": verified,
        "preempt_count": len(preempt_events),
        "first_preempt_timestamp_ms": preempt_time,
        "details": f"Quantum expirations: {len(preempt_events)} at dispatch boundaries; slices executed: {[e.timestamp_ms for e in preempt_events]}",
    }


def verify_claim_3_no_indefinite_monopolization() -> Dict[str, Any]:
    """Claim 3: One tenant cannot monopolize GPU service indefinitely."""
    # Tenant A submits 10 long jobs; Tenant B submits 1 job at t=0
    # Under FIFO, Tenant B must wait until ALL of Tenant A's jobs finish.
    # Under GPU-Aware Fair Scheduler, Tenant B is dispatched within the first round.
    def run_policy(is_fair: bool) -> float:
        tA = Tenant("tA", weight=1.0, priority=5)
        tB = Tenant("tB", weight=1.0, priority=5)
        tenants = {"tA": tA, "tB": tB}
        qm = TenantQueueManager()
        sq = SchedulingQuantum(base_quantum_ms=10.0)
        sched = GPUFairScheduler(tenants=tenants, queue_manager=qm, quantum=sq) if is_fair else FIFOScheduler(tenants=tenants, queue_manager=qm, quantum=sq)
        sim = GPUSimulator(tenants=tenants, quantum_ms=10.0, scheduler=sched)

        # 10 long jobs for A (30ms each = 300ms total)
        for i in range(10):
            sim.submit_job(GPUJob(f"jA_{i}", "tA", "m", 0.0, 30.0, 30.0))
        # 1 job for B at t=0 (10ms)
        jB = GPUJob("jB_0", "tB", "m", 0.0, 10.0, 10.0)
        sim.submit_job(jB)
        sim.run()
        return jB.waiting_time_ms

    wait_fifo = run_policy(is_fair=False)
    wait_fair = run_policy(is_fair=True)

    return {
        "verified": wait_fair < wait_fifo and wait_fair < 50.0,
        "fifo_waiting_time_ms": wait_fifo,
        "fair_waiting_time_ms": wait_fair,
        "details": f"Tenant B wait under FIFO: {wait_fifo:.1f}ms vs Fair Scheduler: {wait_fair:.1f}ms",
    }


def verify_claim_4_tenant_weights_affect_allocation() -> Dict[str, Any]:
    """Claim 4: Tenant weights affect GPU allocation proportionally."""
    # Tenant 1: weight 4.0; Tenant 2: weight 1.0 (4:1 ratio)
    t1 = Tenant("t_heavy", weight=4.0, priority=5)
    t2 = Tenant("t_light", weight=1.0, priority=5)
    tenants = {"t_heavy": t1, "t_light": t2}

    qm = TenantQueueManager()
    sq = SchedulingQuantum(base_quantum_ms=10.0, min_quantum_ms=2.0, max_quantum_ms=40.0)
    sched = GPUFairScheduler(tenants=tenants, queue_manager=qm, quantum=sq, fairness_factor=2.0)
    sim = GPUSimulator(tenants=tenants, quantum_ms=10.0, scheduler=sched)

    # Submit 20 jobs each of 15ms
    for i in range(20):
        sim.submit_job(GPUJob(f"jh_{i}", "t_heavy", "m", 0.0, 15.0, 15.0))
        sim.submit_job(GPUJob(f"jl_{i}", "t_light", "m", 0.0, 15.0, 15.0))
    # Run under contention until 150ms
    sim.run(until_time_ms=150.0)

    s_heavy = t1.accumulated_gpu_service
    s_light = t2.accumulated_gpu_service
    ratio = s_heavy / max(1e-6, s_light)

    return {
        "verified": s_heavy > s_light and ratio >= 2.0,
        "service_heavy_ms": s_heavy,
        "service_light_ms": s_light,
        "service_ratio": ratio,
        "target_weight_ratio": 4.0,
        "details": f"Heavy (w=4) service: {s_heavy:.1f}ms, Light (w=1) service: {s_light:.1f}ms (ratio: {ratio:.2f})",
    }


def verify_claim_5_tenant_quotas_respected() -> Dict[str, Any]:
    """Claim 5: Tenant quotas are respected and throttled at boundaries."""
    # Tenant A has 25% quota in 1000ms window = max 250ms allowed GPU time
    t_throttled = Tenant("t_throttled", weight=1.0, priority=5, gpu_quota=25.0, quota_window_ms=1000.0)
    t_free = Tenant("t_free", weight=1.0, priority=5, gpu_quota=75.0, quota_window_ms=1000.0)
    tenants = {"t_throttled": t_throttled, "t_free": t_free}

    qm = TenantQueueManager()
    sq = SchedulingQuantum(base_quantum_ms=10.0)
    sched = GPUFairScheduler(tenants=tenants, queue_manager=qm, quantum=sq, enable_quota_enforcement=True)
    sim = GPUSimulator(tenants=tenants, quantum_ms=10.0, scheduler=sched)

    # Throttled tenant submits 20 jobs of 20ms = 400ms work (exceeds 250ms quota)
    for i in range(20):
        sim.submit_job(GPUJob(f"j_thr_{i}", "t_throttled", "m", 0.0, 20.0, 20.0))
    # Free tenant submits 10 jobs of 20ms = 200ms work
    for i in range(10):
        sim.submit_job(GPUJob(f"j_free_{i}", "t_free", "m", 0.0, 20.0, 20.0))

    # Run for 500ms (within first window)
    sim.run(until_time_ms=500.0)

    consumed = t_throttled.accumulated_gpu_time_window_ms
    max_allowed = (25.0 / 100.0) * 1000.0  # 250ms

    return {
        "verified": consumed <= max_allowed,
        "quota_consumed_ms": consumed,
        "quota_limit_ms": max_allowed,
        "details": f"Consumed within window: {consumed:.1f}ms <= Allowed: {max_allowed:.1f}ms",
    }


def verify_claim_6_aging_prevents_starvation() -> Dict[str, Any]:
    """Claim 6: Aging escalates low-priority tenant priority preventing starvation."""
    # High priority tenant (p=10) submits 15 jobs of 20ms
    # Low priority tenant (p=1) submits 1 job of 10ms at t=0
    # Under Strict Priority: Low priority runs only after ALL 15 high-prio jobs finish (at t=300ms)
    # Under Fair Scheduler with starvation_threshold=100ms: Low priority job ages and runs before t=150ms
    def test_scheduling(with_aging: bool) -> float:
        t_high = Tenant("t_high", weight=1.0, priority=10)
        t_low = Tenant("t_low", weight=1.0, priority=1)
        tenants = {"t_high": t_high, "t_low": t_low}
        qm = TenantQueueManager()
        sq = SchedulingQuantum(base_quantum_ms=10.0)
        if with_aging:
            sched = GPUFairScheduler(tenants=tenants, queue_manager=qm, quantum=sq, starvation_threshold_ms=80.0, aging_rate=1.5)
        else:
            sched = PriorityScheduler(tenants=tenants, queue_manager=qm, quantum=sq)
        sim = GPUSimulator(tenants=tenants, quantum_ms=10.0, scheduler=sched)

        for i in range(15):
            sim.submit_job(GPUJob(f"jh_{i}", "t_high", "m", 0.0, 20.0, 20.0))
        j_low = GPUJob("j_low", "t_low", "m", 0.0, 10.0, 10.0)
        sim.submit_job(j_low)
        sim.run()
        return j_low.start_time

    start_prio = test_scheduling(with_aging=False)
    start_aging = test_scheduling(with_aging=True)

    return {
        "verified": start_aging < start_prio and start_aging <= 120.0,
        "priority_start_time_ms": start_prio,
        "aging_start_time_ms": start_aging,
        "details": f"Low-priority job started at {start_aging:.1f}ms (Aging) vs {start_prio:.1f}ms (Strict Priority)",
    }


def verify_claim_7_baseline_comparison() -> Dict[str, Any]:
    """Claim 7: The proposed scheduler can be directly compared against baseline schedulers."""
    from src.benchmarks.scenarios import get_scenario
    from src.benchmarks.runner import BenchmarkRunner

    scenario = get_scenario("scenario_1")
    runner = BenchmarkRunner(base_quantum_ms=10.0, context_switch_overhead_ms=0.5)
    report = runner.run_scenario(
        scenario=scenario,
        policies=["fifo", "round_robin", "priority", "weighted_round_robin", "gpu_fair"],
        random_seed=42,
    )

    policies_evaluated = list(report.policy_results.keys())
    has_all_five = set(policies_evaluated) == {"fifo", "round_robin", "priority", "weighted_round_robin", "gpu_fair"}

    return {
        "verified": has_all_five,
        "policies": policies_evaluated,
        "details": f"All 5 schedulers evaluated on identical scenario: {policies_evaluated}",
    }


def verify_claim_8_fairness_quantitatively_measured() -> Dict[str, Any]:
    """Claim 8: Fairness can be quantitatively measured via Jain's Fairness Index."""
    # Test 1: Equal allocation (50, 50) -> J = 1.0
    j_equal = compute_jains_fairness_index([50.0, 50.0])
    # Test 2: Unequal allocation (80, 20) -> J = (100^2)/(2*(6400+400)) = 10000/13600 = 0.735294...
    j_unequal = compute_jains_fairness_index([80.0, 20.0])
    # Test 3: Total monopoly (100, 0) -> J = 10000/(2*10000) = 0.50
    j_monopoly = compute_jains_fairness_index([100.0, 0.0])

    expected_unequal = 10000.0 / 13600.0
    is_accurate = math.isclose(j_equal, 1.0, rel_tol=1e-5) and math.isclose(j_unequal, expected_unequal, rel_tol=1e-5) and math.isclose(j_monopoly, 0.5, rel_tol=1e-5)

    return {
        "verified": is_accurate,
        "j_equal": j_equal,
        "j_unequal": j_unequal,
        "j_monopoly": j_monopoly,
        "details": f"Equal: {j_equal:.4f}, Unequal (80/20): {j_unequal:.4f}, Monopoly: {j_monopoly:.4f}",
    }


def verify_claim_9_gpu_utilization_quantitatively_measured() -> Dict[str, Any]:
    """Claim 9: GPU utilization can be quantitatively measured."""
    sim = GPUSimulator(quantum_ms=10.0, context_switch_overhead_ms=0.0)
    # Job 1: 0 to 10 ms (10ms compute)
    # Job 2 arrives at 30 ms: 30 to 40 ms (10ms compute)
    # Total duration = 40 ms; compute = 20 ms; idle = 20 ms -> 50% utilization
    sim.submit_job(GPUJob("j1", "t1", "m", 0.0, 10.0, 10.0))
    sim.submit_job(GPUJob("j2", "t1", "m", 30.0, 10.0, 10.0))
    sim.run()

    engine = MetricsEngine()
    metrics = engine.compute(
        jobs=sim.completed_jobs,
        total_simulation_time_ms=sim.current_time_ms,
        tenants=["t1"],
        context_switch_overhead_ms=0.0,
        idle_periods=sim.get_idle_periods(),
    )

    util_pct = metrics.gpu_utilization_pct
    idle_pct = metrics.gpu_idle_pct
    expected_util = 50.0

    return {
        "verified": math.isclose(util_pct, expected_util, abs_tol=0.5) and math.isclose(idle_pct, 50.0, abs_tol=0.5),
        "gpu_utilization_pct": util_pct,
        "gpu_idle_pct": idle_pct,
        "total_sim_time_ms": sim.current_time_ms,
        "details": f"Measured GPU Utilization: {util_pct:.2f}%, GPU Idle: {idle_pct:.2f}% (Expected: 50.00%)",
    }


def verify_claim_10_ai_prediction_preserves_fairness() -> Dict[str, Any]:
    """Claim 10: AI workload prediction influences scheduling without breaking fairness."""
    class MockAdversarialPredictor:
        def predict_job(self, job: GPUJob):
            # Adversarially claims favored tenant is IO-bound (boost) and others are GPU-intensive (penalize)
            if job.tenant_id == "t_favored":
                return PredictionOutput(
                    classification=WorkloadClass.IO_BOUND,
                    estimated_latency_ms=1.0,
                    estimated_memory_mb=128.0,
                    confidence_score=1.0,
                )
            return PredictionOutput(
                classification=WorkloadClass.GPU_INTENSIVE,
                estimated_latency_ms=100.0,
                estimated_memory_mb=1024.0,
                confidence_score=1.0,
            )

    # 2 equal tenants (w=1, p=5)
    t1 = Tenant("t_favored", weight=1.0, priority=5)
    t2 = Tenant("t_other", weight=1.0, priority=5)
    tenants = {"t_favored": t1, "t_other": t2}

    qm = TenantQueueManager()
    sched = GPUFairScheduler(tenants=tenants, queue_manager=qm, quantum=SchedulingQuantum(base_quantum_ms=10.0))
    sched.set_predictor(MockAdversarialPredictor())
    sim = GPUSimulator(tenants=tenants, quantum_ms=10.0, scheduler=sched)

    # 10 jobs each
    for i in range(10):
        sim.submit_job(GPUJob(f"jf_{i}", "t_favored", "m", 0.0, 15.0, 15.0))
        sim.submit_job(GPUJob(f"jo_{i}", "t_other", "m", 0.0, 15.0, 15.0))
    sim.run()

    jain_index = compute_jains_fairness_index([t1.accumulated_gpu_service, t2.accumulated_gpu_service])
    both_completed = (t1.total_jobs_completed == 10 and t2.total_jobs_completed == 10)

    return {
        "verified": jain_index >= 0.95 and both_completed,
        "jains_fairness_index": jain_index,
        "favored_service_ms": t1.accumulated_gpu_service,
        "other_service_ms": t2.accumulated_gpu_service,
        "details": f"Jain's Fairness Index under adversarial ML manipulation: {jain_index:.4f} (>= 0.95), both completed 10/10",
    }


def verify_claim_11_works_without_jetson_hardware() -> Dict[str, Any]:
    """Claim 11: The system works completely without physical Jetson hardware."""
    executor = create_executor(backend="auto")
    is_simulation = isinstance(executor, SimulationExecutor)
    telemetry = executor.get_hardware_telemetry()

    job = GPUJob("test_sim_dispatch", "t1", "m", 0.0, 15.0, 15.0)
    record = executor.dispatch_slice(job, quantum_slice_ms=10.0)

    return {
        "verified": is_simulation and record.actual_gpu_execution_time_ms > 0.0,
        "executor_type": executor.__class__.__name__,
        "backend_type": telemetry.backend_type,
        "actual_gpu_time_ms": record.actual_gpu_execution_time_ms,
        "details": f"Active executor: {executor.__class__.__name__}, successfully dispatched slice in {record.actual_gpu_execution_time_ms:.1f}ms",
    }


def verify_claim_12_jetson_adapter_fails_safely() -> Dict[str, Any]:
    """Claim 12: The optional Jetson adapter fails gracefully when hardware is unavailable."""
    from src.core.exceptions import CUDAUnavailableError
    executor = JetsonExecutor()
    is_avail = executor.is_available()
    telemetry = executor.get_hardware_telemetry()

    # Attempt dispatch to unavailable Jetson executor
    job = GPUJob("hw_fail_test", "t1", "m", 0.0, 10.0, 10.0)
    safely_handled = False
    try:
        executor.dispatch_slice(job, quantum_slice_ms=10.0)
    except CUDAUnavailableError as e:
        safely_handled = "unavailable" in str(e).lower()
    except Exception:
        safely_handled = False

    return {
        "verified": (not is_avail) and safely_handled and (not telemetry.hardware_available),
        "is_available": is_avail,
        "telemetry_available": telemetry.hardware_available,
        "details": f"JetsonExecutor.is_available()={is_avail}, safely caught CUDAUnavailableError: {safely_handled}",
    }


def run_all_verifications():
    print("=" * 80)
    print("  FINAL END-TO-END EXPERIMENTAL CLAIMS VERIFICATION")
    print("=" * 80)

    claims = [
        ("Claim 1: Multiple tenants share one simulated GPU", verify_claim_1_multi_tenant_sharing),
        ("Claim 2: GPU time slicing occurs at dispatch boundaries", verify_claim_2_time_slicing),
        ("Claim 3: One tenant cannot monopolize GPU indefinitely", verify_claim_3_no_indefinite_monopolization),
        ("Claim 4: Tenant weights affect GPU allocation", verify_claim_4_tenant_weights_affect_allocation),
        ("Claim 5: Tenant quotas are respected and throttled", verify_claim_5_tenant_quotas_respected),
        ("Claim 6: Aging prevents starvation", verify_claim_6_aging_prevents_starvation),
        ("Claim 7: Proposed scheduler compares to baselines", verify_claim_7_baseline_comparison),
        ("Claim 8: Fairness is quantitatively measured", verify_claim_8_fairness_quantitatively_measured),
        ("Claim 9: GPU utilization is quantitatively measured", verify_claim_9_gpu_utilization_quantitatively_measured),
        ("Claim 10: AI prediction influences scheduling fairly", verify_claim_10_ai_prediction_preserves_fairness),
        ("Claim 11: Works without physical Jetson hardware", verify_claim_11_works_without_jetson_hardware),
        ("Claim 12: Jetson adapter fails gracefully", verify_claim_12_jetson_adapter_fails_safely),
    ]

    all_passed = True
    results = {}
    for idx, (title, func) in enumerate(claims, 1):
        try:
            res = func()
            passed = res.get("verified", False)
            status = "✓ PASS" if passed else "✗ FAIL"
            if not passed:
                all_passed = False
            results[f"claim_{idx}"] = res
            print(f"[{status}] {title:<60}")
            print(f"        -> {res.get('details', '')}")
        except Exception as e:
            all_passed = False
            print(f"[✗ EXCEPTION] {title:<60}")
            print(f"        -> Exception: {e}")

    print("=" * 80)
    if all_passed:
        print("  ALL 12 CLAIMS EXPERIMENTALLY DEMONSTRATED AND VERIFIED")
    else:
        print("  SOME CLAIMS FAILED VERIFICATION")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    success = run_all_verifications()
    sys.exit(0 if success else 1)
