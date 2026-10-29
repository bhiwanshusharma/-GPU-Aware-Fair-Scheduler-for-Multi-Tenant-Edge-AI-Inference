"""Unit tests verifying bounded influence, zero-starvation, and explainability under incorrect ML predictions.

Tests verify:
1. Adversarial/hallucinated latency predictions cannot skew service or break fair sharing.
2. Inverted/incorrect workload classifications cannot break proportional fairness.
3. Priority aging strictly bounds waiting time even under continuous adversarial ML boosting.
4. Quota enforcement is strictly immune to ML model bias.
5. Every scheduling decision record contains all 12 required explainability fields and formats cleanly.
6. Graceful fallback when the predictor throws runtime exceptions.
7. Mathematical clamping invariants (effective latency in [0.5*E, 2*E], ML delta in [-1, +1]).
"""

import math
import pytest
from typing import Dict, List, Any

from src.core.types import (
    SchedulingQuantum,
    WorkloadClass,
    PredictionOutput,
)
from src.core.job import GPUJob
from src.core.tenant import Tenant
from src.core.queues import TenantQueueManager
from src.core.decision import SchedulingDecision
from src.scheduler.fair_scheduler import GPUFairScheduler, ScoreBreakdown
from src.scheduler.evaluator import run_scheduler_benchmark


class MockPredictor:
    """Configurable mock predictor simulating adversarial or pathological ML outputs."""

    def __init__(
        self,
        canned_latency: float = 20.0,
        canned_class: WorkloadClass = WorkloadClass.BALANCED,
        canned_memory: float = 16.0,
        canned_confidence: float = 1.0,
        per_tenant_overrides: Dict[str, Dict[str, Any]] = None,
        raise_exception: bool = False,
    ) -> None:
        self.canned_latency = canned_latency
        self.canned_class = canned_class
        self.canned_memory = canned_memory
        self.canned_confidence = canned_confidence
        self.per_tenant_overrides = per_tenant_overrides or {}
        self.raise_exception = raise_exception
        self.call_count = 0

    def predict(self, job: GPUJob) -> PredictionOutput:
        self.call_count += 1
        if self.raise_exception:
            raise RuntimeError("Simulated ML engine memory corruption or GPU failure")

        if job.tenant_id in self.per_tenant_overrides:
            override = self.per_tenant_overrides[job.tenant_id]
            return PredictionOutput(
                estimated_latency_ms=override.get("latency", self.canned_latency),
                estimated_memory_mb=override.get("memory", self.canned_memory),
                classification=override.get("class", self.canned_class),
                confidence_score=override.get("confidence", self.canned_confidence),
            )

        return PredictionOutput(
            estimated_latency_ms=self.canned_latency,
            estimated_memory_mb=self.canned_memory,
            classification=self.canned_class,
            confidence_score=self.canned_confidence,
        )


def make_tenants(tenant_list: List[Tenant]) -> Dict[str, Tenant]:
    return {t.tenant_id: t for t in tenant_list}


# =========================================================================
# 1. Adversarial Latency Predictions Do Not Break Fairness
# =========================================================================
def test_adversarial_latency_predictions_do_not_break_fairness():
    """Wildly wrong latency predictions (0.0001 ms vs 1,000,000 ms) cannot bias proportional service."""
    t1 = Tenant("tenant_favored", name="Favored", weight=1.0, priority=5, gpu_quota=100.0)
    t2 = Tenant("tenant_penalized", name="Penalized", weight=1.0, priority=5, gpu_quota=100.0)
    tenants = make_tenants([t1, t2])

    # Adversarial predictor: Favored tenant gets 0.0001 ms prediction; Penalized tenant gets 1,000,000 ms
    predictor = MockPredictor(
        per_tenant_overrides={
            "tenant_favored": {"latency": 0.0001, "class": WorkloadClass.BALANCED},
            "tenant_penalized": {"latency": 1_000_000.0, "class": WorkloadClass.BALANCED},
        }
    )

    sq = SchedulingQuantum(base_quantum_ms=20.0, min_quantum_ms=5.0, max_quantum_ms=50.0)
    sched = GPUFairScheduler(tenants=tenants, quantum=sq, predictor=predictor, fairness_factor=2.0)

    # Both tenants submit 4 identical 20 ms jobs at t=0
    jobs: List[GPUJob] = []
    for i in range(4):
        jobs.append(GPUJob(f"F{i}", "tenant_favored", "model_f", arrival_time=0.0, estimated_gpu_time=20.0, remaining_gpu_time=20.0, priority=5))
        jobs.append(GPUJob(f"P{i}", "tenant_penalized", "model_p", arrival_time=0.0, estimated_gpu_time=20.0, remaining_gpu_time=20.0, priority=5))

    res = run_scheduler_benchmark(sched, jobs, quantum_ms=20.0, context_switch_overhead_ms=0.5)

    assert len(res.completed_jobs) == 8
    # Proportional fair sharing is preserved: 80 ms each
    assert res.per_tenant_gpu_service_ms["tenant_favored"] == 80.0
    assert res.per_tenant_gpu_service_ms["tenant_penalized"] == 80.0
    assert math.isclose(res.jains_fairness_index, 1.0, abs_tol=1e-3)


# =========================================================================
# 2. Inverted Workload Classification Does Not Break Fairness
# =========================================================================
def test_inverted_workload_classification_does_not_break_fairness():
    """Inverting classifications (heavy vision as IO_BOUND, light sensor as GPU_INTENSIVE) preserves fairness."""
    t_heavy = Tenant("tenant_heavy", name="Heavy", weight=1.0, priority=5, gpu_quota=100.0)
    t_light = Tenant("tenant_light", name="Light", weight=1.0, priority=5, gpu_quota=100.0)
    tenants = make_tenants([t_heavy, t_light])

    # Inverted predictor: Heavy tenant gets IO_BOUND (+0.5 priority boost, 1.5x quantum multiplier);
    # Light tenant gets GPU_INTENSIVE (-0.25 priority penalty, 0.8x quantum multiplier).
    predictor = MockPredictor(
        per_tenant_overrides={
            "tenant_heavy": {"class": WorkloadClass.IO_BOUND, "latency": 1.0},
            "tenant_light": {"class": WorkloadClass.GPU_INTENSIVE, "latency": 40.0},
        }
    )

    sq = SchedulingQuantum(base_quantum_ms=20.0, min_quantum_ms=5.0, max_quantum_ms=50.0)
    sched = GPUFairScheduler(tenants=tenants, quantum=sq, predictor=predictor, fairness_factor=2.0)

    # 4 jobs each of 20 ms
    jobs: List[GPUJob] = []
    for i in range(4):
        jobs.append(GPUJob(f"H{i}", "tenant_heavy", "heavy_vision", arrival_time=0.0, estimated_gpu_time=20.0, remaining_gpu_time=20.0))
        jobs.append(GPUJob(f"L{i}", "tenant_light", "light_sensor", arrival_time=0.0, estimated_gpu_time=20.0, remaining_gpu_time=20.0))

    res = run_scheduler_benchmark(sched, jobs, quantum_ms=20.0, context_switch_overhead_ms=0.5)

    assert len(res.completed_jobs) == 8
    assert res.per_tenant_gpu_service_ms["tenant_heavy"] == 80.0
    assert res.per_tenant_gpu_service_ms["tenant_light"] == 80.0
    assert math.isclose(res.jains_fairness_index, 1.0, abs_tol=1e-3)


# =========================================================================
# 3. Continuous Adversarial ML Boosting Cannot Starve Low-Priority Tenant
# =========================================================================
def test_continuous_adversarial_ml_boosting_cannot_starve_low_priority_tenant():
    """Aging mathematically bounds wait time and guarantees dispatch despite adversarial ML boost."""
    t_high = Tenant("tenant_high", name="HighPriority", weight=1.0, priority=8, gpu_quota=100.0)
    t_low = Tenant("tenant_low", name="LowPriority", weight=1.0, priority=1, gpu_quota=100.0)
    tenants = make_tenants([t_high, t_low])

    # Predictor awards High tenant maximum boost (IO_BOUND, +0.5 boost) and Low tenant penalty (GPU_INTENSIVE, -0.25)
    predictor = MockPredictor(
        per_tenant_overrides={
            "tenant_high": {"class": WorkloadClass.IO_BOUND, "latency": 1.0},
            "tenant_low": {"class": WorkloadClass.GPU_INTENSIVE, "latency": 50.0},
        }
    )

    sq = SchedulingQuantum(
        base_quantum_ms=20.0,
        min_quantum_ms=5.0,
        max_quantum_ms=50.0,
        starvation_threshold_ms=40.0,
    )
    # Aging rate = 2.0 ensures aging term grows rapidly
    sched = GPUFairScheduler(
        tenants=tenants,
        quantum=sq,
        predictor=predictor,
        priority_factor=1.0,
        aging_rate=2.0,
        starvation_threshold_ms=40.0,
    )

    # Low tenant has 1 job arriving at t=0.0
    low_job = GPUJob("LOW_1", "tenant_low", "model", arrival_time=0.0, estimated_gpu_time=20.0, remaining_gpu_time=20.0)
    sched.queue_manager.enqueue(low_job)

    # Initial high-priority job arriving at t=0.0
    high_job_0 = GPUJob("HIGH_0", "tenant_high", "model", arrival_time=0.0, estimated_gpu_time=20.0, remaining_gpu_time=20.0)
    sched.queue_manager.enqueue(high_job_0)

    # Initial dispatch at t=0: High tenant wins due to base priority 8 vs 1
    d1 = sched.schedule(0.0)
    assert d1.tenant_id == "tenant_high"

    # Now simulate subsequent continuous arrivals from High tenant at later timestamps (e.g., t=180.0)
    # The low-priority job has been waiting since t=0.0
    high_job_stream = GPUJob("HIGH_STREAM", "tenant_high", "model", arrival_time=180.0, estimated_gpu_time=20.0, remaining_gpu_time=20.0)
    sched.queue_manager.enqueue(high_job_stream)

    # Simulate dispatch check at t=200.0ms:
    # Low job waiting time: 200.0 - 0.0 = 200.0 ms -> Aging term = 2.0 * (200.0 / 40.0) = +10.0
    # Score for Low job: P_eff = 1.0 (base) - 0.25 (ml_adj) + 10.0 (aging) = 10.75
    # High job waiting time: 200.0 - 180.0 = 20.0 ms -> Aging term = 2.0 * (20.0 / 40.0) = +1.0
    # Score for High job: P_eff = 8.0 (base) + 0.5 (ml_adj) + 1.0 (aging) = 9.5
    # Aging guarantees Low job overtakes High job despite adversarial ML advantage!
    d_overtake = sched.schedule(200.0)
    assert d_overtake is not None
    assert d_overtake.tenant_id == "tenant_low"
    assert d_overtake.job_id == "LOW_1"


# =========================================================================
# 4. Quota Enforcement Immune to Favorable ML Predictions
# =========================================================================
def test_quota_enforcement_immune_to_favorable_ml_predictions():
    """A quota-exhausted tenant is strictly ineligible even with maximum ML boost."""
    t_throttled = Tenant("tenant_throttled", name="Throttled", weight=1.0, priority=10, gpu_quota=20.0, quota_window_ms=100.0)
    t_normal = Tenant("tenant_normal", name="Normal", weight=1.0, priority=1, gpu_quota=80.0, quota_window_ms=100.0)
    tenants = make_tenants([t_throttled, t_normal])

    # Predictor tries to aggressively favor the throttled tenant
    predictor = MockPredictor(
        per_tenant_overrides={
            "tenant_throttled": {"class": WorkloadClass.IO_BOUND, "latency": 0.1, "confidence": 1.0},
            "tenant_normal": {"class": WorkloadClass.GPU_INTENSIVE, "latency": 100.0, "confidence": 1.0},
        }
    )

    sq = SchedulingQuantum(base_quantum_ms=20.0, min_quantum_ms=5.0, max_quantum_ms=50.0)
    sched = GPUFairScheduler(tenants=tenants, quantum=sq, predictor=predictor, enable_quota_enforcement=True)

    # Exhaust throttled tenant's quota window: quota=20% of 100ms = 20ms allowed
    t_throttled.accumulated_gpu_time_window_ms = 25.0  # Consumption > 100%

    j_throttled = GPUJob("TH_1", "tenant_throttled", "model", arrival_time=0.0, estimated_gpu_time=10.0, remaining_gpu_time=10.0)
    j_normal = GPUJob("NORM_1", "tenant_normal", "model", arrival_time=0.0, estimated_gpu_time=10.0, remaining_gpu_time=10.0)
    sched.queue_manager.enqueue(j_throttled)
    sched.queue_manager.enqueue(j_normal)

    # Throttled tenant must be rejected despite highest priority and favorable ML prediction
    winner = sched.schedule(10.0)
    assert winner is not None
    assert winner.tenant_id == "tenant_normal"


# =========================================================================
# 5. Explainable Scheduling Decision Contains All 12 Fields
# =========================================================================
def test_explainable_scheduling_decision_contains_all_twelve_fields():
    """Every scheduling decision captures all 12 required explainability attributes and formats cleanly."""
    t1 = Tenant("tenant_1", name="Vision Tenant", weight=2.5, priority=7, gpu_quota=100.0)
    tenants = make_tenants([t1])

    predictor = MockPredictor(
        canned_latency=18.5,
        canned_class=WorkloadClass.GPU_INTENSIVE,
        canned_memory=32.0,
        canned_confidence=0.95,
    )

    sq = SchedulingQuantum(base_quantum_ms=25.0, min_quantum_ms=5.0, max_quantum_ms=50.0)
    sched = GPUFairScheduler(tenants=tenants, quantum=sq, predictor=predictor)

    job = GPUJob("J_AUDIT", "tenant_1", "yolov8n", arrival_time=5.0, estimated_gpu_time=25.0, remaining_gpu_time=25.0)
    sched.queue_manager.enqueue(job)

    dispatched = sched.schedule(15.0)  # wait_time = 10.0 ms
    assert dispatched is not None
    assert len(sched.decisions_history) == 1

    dec: SchedulingDecision = sched.decisions_history[0]

    # Verify all 12 required fields
    # 1. selected tenant
    assert dec.selected_tenant == "tenant_1"
    assert dec.selected_tenant_id == "tenant_1"

    # 2. selected job
    assert dec.selected_job == dispatched
    assert dec.selected_job.job_id == "J_AUDIT"

    # 3. predicted workload type
    assert dec.predicted_workload_type == "GPU_INTENSIVE"

    # 4. predicted GPU behavior
    assert isinstance(dec.predicted_gpu_behavior, dict)
    assert dec.predicted_gpu_behavior["estimated_latency_ms"] == 18.5
    assert dec.predicted_gpu_behavior["expected_gpu_intensity"] == "HIGH"
    assert dec.predicted_gpu_behavior["estimated_memory_mb"] == 32.0
    assert dec.predicted_gpu_behavior["confidence_score"] == 0.95
    assert "ml_priority_adjustment" in dec.predicted_gpu_behavior

    # 5. current tenant GPU service
    assert dec.current_tenant_gpu_service == 0.0

    # 6. tenant weight
    assert dec.tenant_weight == 2.5

    # 7. tenant quota usage
    assert dec.tenant_quota_usage == 0.0

    # 8. priority
    assert dec.priority == 7.0

    # 9. waiting time
    assert dec.waiting_time == 10.0  # 15.0 - 5.0

    # 10. scheduling score
    assert isinstance(dec.scheduling_score, float)
    assert not math.isnan(dec.scheduling_score)

    # 11. selected quantum
    assert dec.selected_quantum >= 5.0
    assert dec.selected_quantum <= 50.0

    # 12. reason for selection
    assert isinstance(dec.reason_for_selection, str)
    assert "MAX_COMPOSITE_SCORE" in dec.reason_for_selection

    # Check score breakdown sub-dictionary
    assert "priority_term" in dec.score_breakdown
    assert "aging_term" in dec.score_breakdown
    assert "fairness_term" in dec.score_breakdown
    assert "responsiveness_term" in dec.score_breakdown
    assert "quota_penalty" in dec.score_breakdown
    assert "ml_priority_adjustment" in dec.score_breakdown

    # Verify explain() formatting
    explanation = dec.explain()
    assert isinstance(explanation, str)
    assert "=== Scheduling Decision: DEC-000001" in explanation
    assert "Selected Tenant           : tenant_1" in explanation
    assert "Selected Job              : J_AUDIT" in explanation
    assert "Predicted Workload Type   : GPU_INTENSIVE" in explanation
    assert "HIGH" in explanation
    assert "priority_term" in explanation


# =========================================================================
# 6. Graceful Predictor Exception Resilience
# =========================================================================
def test_predictor_exception_resilience():
    """Predictor exceptions are gracefully caught without crashing scheduler or disrupting decisions."""
    t1 = Tenant("tenant_safe", name="Safe", weight=1.0, priority=5, gpu_quota=100.0)
    tenants = make_tenants([t1])

    broken_predictor = MockPredictor(raise_exception=True)
    sq = SchedulingQuantum(base_quantum_ms=20.0, min_quantum_ms=5.0, max_quantum_ms=50.0)
    sched = GPUFairScheduler(tenants=tenants, quantum=sq, predictor=broken_predictor)

    job = GPUJob("J_CRASH", "tenant_safe", "model", arrival_time=0.0, estimated_gpu_time=20.0, remaining_gpu_time=20.0)
    sched.queue_manager.enqueue(job)

    # Must complete dispatch successfully despite predictor raising RuntimeError
    dispatched = sched.schedule(0.0)
    assert dispatched is not None
    assert dispatched.job_id == "J_CRASH"
    assert len(sched.decisions_history) == 1
    assert sched.decisions_history[0].selected_job == dispatched


# =========================================================================
# 7. Mathematical Clamping Invariants
# =========================================================================
def test_mathematical_clamping_invariants():
    """Effective latency is clamped to [0.5*E, 2*E], and ML priority delta is clamped to [-1.0, +1.0]."""
    t1 = Tenant("tenant_inv", name="Invariant", weight=1.0, priority=5, gpu_quota=100.0)
    tenants = make_tenants([t1])

    # Extreme low latency: 0.0 ms
    pred_zero = MockPredictor(canned_latency=0.0, canned_class=WorkloadClass.IO_BOUND, canned_confidence=1.0)
    sched_zero = GPUFairScheduler(tenants=tenants, predictor=pred_zero)
    job_100ms = GPUJob("J1", "tenant_inv", "model", arrival_time=0.0, estimated_gpu_time=100.0, remaining_gpu_time=100.0)
    bd_zero = sched_zero.calculate_score(job_100ms, current_time_ms=0.0, v_bar=0.0)

    # Effective latency cannot fall below 0.5 * 100.0 = 50.0 ms
    assert bd_zero.predicted_gpu_behavior["effective_latency_ms"] >= 50.0
    # ML priority delta cannot exceed +1.0
    assert bd_zero.ml_priority_adjustment <= 1.0

    # Extreme high latency: 1,000,000.0 ms
    pred_huge = MockPredictor(canned_latency=1_000_000.0, canned_class=WorkloadClass.GPU_INTENSIVE, canned_confidence=1.0)
    sched_huge = GPUFairScheduler(tenants=tenants, predictor=pred_huge)
    bd_huge = sched_huge.calculate_score(job_100ms, current_time_ms=0.0, v_bar=0.0)

    # Effective latency cannot exceed 2.0 * 100.0 = 200.0 ms
    assert bd_huge.predicted_gpu_behavior["effective_latency_ms"] <= 200.0
    # ML priority delta cannot fall below -1.0
    assert bd_huge.ml_priority_adjustment >= -1.0
