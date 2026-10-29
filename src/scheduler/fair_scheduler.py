"""GPU-Aware Fair Scheduler (P-DWFQ / FairScheduler) for Multi-Tenant Edge AI.

This module implements the proposed GPU-Aware Fair Scheduler combining:
1. Dynamic GPU time-slicing (clamped within [Q_min, Q_max]).
2. Weighted fair sharing based on normalized service lag (virtual runtime).
3. Priority enforcement with starvation prevention via continuous priority aging.
4. Sliding-window GPU quota compliance with soft backpressure and hard eligibility gating.
5. Short-job responsiveness bias (SRPT heuristic).
6. Comprehensive scheduling decision tracking and history telemetry.

Mathematical Formulation:
-------------------------
For any eligible candidate job j belonging to tenant i at simulation time t:

1. Quota Headroom Ratio:
       C_i(t) = U_i(t) / Q_i_max
   where U_i(t) is consumed GPU execution time in window W_i, and
   Q_i_max = W_i * (gpu_quota_pct / 100).
   If enable_quota_enforcement is True and C_i(t) >= 1.0, tenant i is ineligible.

2. Normalized Service (Virtual Runtime) & Fair Deficit:
       v_i(t) = S_i(t) / w_i
       v_bar(t) = sum_{k in A_elig} S_k(t) / sum_{k in A_elig} w_k
       F_i(t) = v_bar(t) - v_i(t)
   where S_i(t) is total accumulated GPU service (ms) delivered to tenant i,
   and w_i is the tenant's weight.

3. Priority and Starvation-Preventing Aging:
       P(j, t) = k_p * p_i + k_a * ( (t - arrival_time_j) / T_starve )
   where p_i is base priority, k_p is priority_factor, k_a is aging_rate,
   and T_starve is starvation_threshold_ms.

4. Short-Job Responsiveness:
       R(j) = Q_base / max(remaining_gpu_time_j, Q_base)

5. Quota Headroom Penalty:
       H_i(t) = k_q * C_i(t)

6. Composite Fair Dispatch Score:
       S(j, t) = P(j, t) + k_f * F_i(t) + k_r * R(j) - H_i(t)

7. Dynamic Quantum Allocation:
       rho_i = w_i / mean_w
       mu_i = 1.0 + tanh( F_i(t) / Q_base )
       Q_target = Q_base * rho_i * mu_i
       Q_allocated = clamp(Q_target, Q_min, Q_max)
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from src.core.types import SchedulingQuantum, PolicyType
from src.core.job import GPUJob
from src.core.tenant import Tenant
from src.core.queues import TenantQueueManager
from src.core.decision import SchedulingDecision
from src.core.scheduler_config import SchedulerConfiguration
from src.scheduler.base import BaseScheduler
from src.core.exceptions import (
    UnknownTenantError,
    SchedulerStarvationError,
    TenantQuotaExhaustedError,
)

logger = logging.getLogger(__name__)


@dataclass
class ScoreBreakdown:
    """Detailed components of the mathematical scheduling score for a candidate job."""
    job_id: str
    tenant_id: str
    priority_term: float
    aging_term: float
    fairness_term: float
    responsiveness_term: float
    quota_penalty: float
    total_score: float
    quota_consumption_ratio: float
    is_quota_eligible: bool
    ml_priority_adjustment: float = 0.0
    predicted_workload_type: str = "BALANCED"
    predicted_gpu_behavior: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TenantDispatchRecord:
    """Historical record of an individual tenant dispatch."""
    timestamp_ms: float
    job_id: str
    allocated_quantum_ms: float
    actual_executed_ms: float
    preempted: bool


class GPUFairScheduler(BaseScheduler):
    """GPU-Aware Fair Scheduler implementing the mathematically defined dispatch score.
    
    Provides strict starvation prevention, proportional weighted fair service,
    configurable time quanta, and sliding-window quota clamping.
    """

    def __init__(
        self,
        tenants: Optional[Dict[str, Tenant]] = None,
        queue_manager: Optional[TenantQueueManager] = None,
        quantum: Optional[SchedulingQuantum] = None,
        config: Optional[SchedulerConfiguration] = None,
        predictor: Optional[Any] = None,
        priority_factor: float = 1.0,
        aging_rate: float = 1.0,
        fairness_factor: float = 1.0,
        responsiveness_factor: float = 0.5,
        quota_factor: float = 1.0,
        starvation_threshold_ms: float = 100.0,
        enable_quota_enforcement: bool = True,
        enable_workload_multiplier: bool = True,
        context_switch_overhead_ms: float = 0.5,
    ) -> None:
        # If a SchedulerConfiguration is provided, inherit parameters
        if config is not None:
            quantum = config.to_scheduling_quantum()
            aging_rate = config.aging_alpha
            starvation_threshold_ms = config.starvation_threshold_ms
            enable_quota_enforcement = config.enable_quota_enforcement
            enable_workload_multiplier = config.enable_workload_multiplier

        super().__init__(
            tenants=tenants,
            queue_manager=queue_manager,
            quantum=quantum,
            name="GPU-Aware Fair Scheduler",
        )

        self.predictor = predictor
        self.enable_workload_multiplier = bool(enable_workload_multiplier)

        # Core mathematical score weighting factors
        if priority_factor < 0.0:
            raise ValueError(f"priority_factor must be non-negative, got {priority_factor}")
        self.priority_factor = float(priority_factor)

        if aging_rate < 0.0:
            raise ValueError(f"aging_rate must be non-negative, got {aging_rate}")
        self.aging_rate = float(aging_rate)

        if fairness_factor < 0.0:
            raise ValueError(f"fairness_factor must be non-negative, got {fairness_factor}")
        self.fairness_factor = float(fairness_factor)

        if responsiveness_factor < 0.0:
            raise ValueError(f"responsiveness_factor must be non-negative, got {responsiveness_factor}")
        self.responsiveness_factor = float(responsiveness_factor)

        if quota_factor < 0.0:
            raise ValueError(f"quota_factor must be non-negative, got {quota_factor}")
        self.quota_factor = float(quota_factor)

        if starvation_threshold_ms <= 0.0:
            raise ValueError(f"starvation_threshold_ms must be positive, got {starvation_threshold_ms}")
        self.starvation_threshold_ms = float(starvation_threshold_ms)

        self.enable_quota_enforcement = bool(enable_quota_enforcement)
        self.context_switch_overhead_ms = float(context_switch_overhead_ms)

        # Tracking state
        self._last_allocated_quantum: Dict[str, float] = {}
        self.decisions_history: List[SchedulingDecision] = []
        self.dispatch_history: Dict[str, List[TenantDispatchRecord]] = {
            t_id: [] for t_id in self.tenants.keys()
        }
        self.decision_counter: int = 0

    def set_predictor(self, predictor: Optional[Any]) -> None:
        """Attaches or detaches an AI workload predictor."""
        self.predictor = predictor

    def get_workload_multiplier(self, workload_class: Any) -> float:
        """Dynamic workload multiplier psi(c) scaling quantum based on workload class."""
        if not self.enable_workload_multiplier:
            return 1.0
        c_str = str(workload_class.value if hasattr(workload_class, "value") else workload_class).upper()
        if c_str == "GPU_INTENSIVE":
            return 0.8  # prevent HoL blocking from massive matrix compute
        elif c_str in ("MIXED", "BALANCED"):
            return 1.0  # baseline
        elif c_str == "CPU_BOUND":
            return 1.2  # accelerate small CPU-bound tasks
        elif c_str == "IO_BOUND":
            return 1.5  # flush quick I/O-bound requests
        return 1.0

    def compute_quota_ratio(self, tenant: Tenant, current_time_ms: float) -> Tuple[float, bool]:
        """Calculates quota consumption ratio C_i(t) and eligibility indicator."""
        tenant.update_window(current_time_ms)
        allowed_gpu_time = (tenant.gpu_quota / 100.0) * tenant.quota_window_ms
        if allowed_gpu_time <= 0.0:
            return 1.0, False

        consumption_ratio = tenant.accumulated_gpu_time_window_ms / allowed_gpu_time
        is_eligible = not (self.enable_quota_enforcement and consumption_ratio >= 1.0)
        return consumption_ratio, is_eligible

    def calculate_score(
        self,
        job: GPUJob,
        current_time_ms: float,
        v_bar: float,
    ) -> ScoreBreakdown:
        """Evaluates the formal mathematical dispatch score S(j, t) for candidate job j.
        
        Args:
            job: The pending inference job.
            current_time_ms: Current simulation clock in milliseconds.
            v_bar: System average normalized virtual service across active eligible tenants.
            
        Returns:
            ScoreBreakdown containing individual terms and the final composite score.
        """
        if job.tenant_id not in self.tenants:
            logger.error(f"Unknown tenant ID in candidate job: '{job.tenant_id}'")
            raise UnknownTenantError(
                f"Candidate job '{job.job_id}' belongs to unknown tenant '{job.tenant_id}'. "
                f"Registered tenants: {list(self.tenants.keys())}"
            )
        tenant = self.tenants[job.tenant_id]

        # 1. Quota Check
        quota_ratio, is_eligible = self.compute_quota_ratio(tenant, current_time_ms)

        # Workload Predictor Integration with Bounded Influence
        pred = None
        if self.predictor is not None:
            try:
                if hasattr(self.predictor, "predict"):
                    pred = self.predictor.predict(job)
                elif hasattr(self.predictor, "predict_job"):
                    pred = self.predictor.predict_job(job)
            except Exception as e:
                logger.warning(f"Predictor raised exception: {e}; falling back to static attributes.")
                pred = None

        if pred is not None:
            raw_lat = float(getattr(pred, "estimated_latency_ms", job.remaining_gpu_time))
            if math.isnan(raw_lat) or math.isinf(raw_lat) or raw_lat < 0.0:
                logger.warning(f"Sanitizing invalid predicted latency: {raw_lat} -> {job.remaining_gpu_time}")
                raw_pred_latency = float(job.remaining_gpu_time)
            else:
                raw_pred_latency = raw_lat

            raw_conf = float(getattr(pred, "confidence_score", 1.0))
            if math.isnan(raw_conf) or not (0.0 <= raw_conf <= 1.0):
                confidence = 1.0
            else:
                confidence = raw_conf

            pred_class = getattr(pred, "classification", job.workload_type)
            pred_mem = float(getattr(pred, "estimated_memory_mb", getattr(job, "predicted_memory_mb", 0.0)))
        else:
            raw_pred_latency = float(job.remaining_gpu_time)
            pred_class = job.workload_type
            pred_mem = float(getattr(job, "predicted_memory_mb", 0.0))
            confidence = 1.0

        # Workload classification string
        wl_str = str(pred_class.value if hasattr(pred_class, "value") else pred_class).upper()

        # Bounded Priority Adjustment: Delta_ml in [-1.0, +1.0]
        # Prevents ML from overtaking priority tiers (Delta p >= 1.0) or overriding aging
        if "IO_BOUND" in wl_str:
            raw_delta = 0.5 * max(0.0, min(1.0, confidence))
        elif "CPU_BOUND" in wl_str:
            raw_delta = 0.25 * max(0.0, min(1.0, confidence))
        elif "GPU_INTENSIVE" in wl_str:
            raw_delta = -0.25 * max(0.0, min(1.0, confidence))
        else:
            raw_delta = 0.0

        ml_priority_adjustment = max(-1.0, min(1.0, raw_delta)) if self.predictor is not None else 0.0

        # 2. Priority and Aging Term: P(j, t) = k_p * p_i + Delta_ml + k_a * (W_j / T_starve)
        wait_time = max(0.0, current_time_ms - job.arrival_time)
        if wait_time >= self.starvation_threshold_ms:
            logger.debug(
                f"Starvation threat: Job '{job.job_id}' (tenant='{job.tenant_id}') waiting time "
                f"{wait_time:.1f}ms exceeds starvation threshold {self.starvation_threshold_ms:.1f}ms. "
                f"Dynamic aging escalation active."
            )

        base_priority_term = self.priority_factor * float(tenant.priority)
        priority_term = base_priority_term + ml_priority_adjustment
        aging_term = self.aging_rate * (wait_time / self.starvation_threshold_ms)

        # Update tenant effective priority for telemetry
        tenant.effective_priority = tenant.priority + (aging_term / max(1e-6, self.priority_factor))

        # 3. Normalized Fair Service Lag: F_i(t) = v_bar(t) - (S_i(t) / w_i)
        v_i = tenant.accumulated_gpu_service / max(1e-6, tenant.weight)
        fairness_term = self.fairness_factor * (v_bar - v_i)

        # 4. Short-Job Responsiveness Term with Bounded Latency Blending:
        # gamma_ml in [0, 0.25]. Latency clamped within [0.5 * E_j, 2.0 * E_j].
        q_base = self.quantum.base_quantum_ms
        if self.predictor is not None:
            gamma_ml = 0.25 * max(0.0, min(1.0, confidence))
            blended_latency = (1.0 - gamma_ml) * job.remaining_gpu_time + gamma_ml * raw_pred_latency
            effective_latency = max(0.5 * job.remaining_gpu_time, min(2.0 * job.remaining_gpu_time, blended_latency))
            effective_latency = max(0.1, effective_latency)
        else:
            effective_latency = max(0.1, job.remaining_gpu_time)

        responsiveness_term = self.responsiveness_factor * (
            q_base / max(effective_latency, q_base)
        )

        # 5. Quota Penalty: H_i(t) = k_q * C_i(t)
        quota_penalty = self.quota_factor * min(1.0, quota_ratio)

        # 6. Composite Score
        total_score = (
            priority_term
            + aging_term
            + fairness_term
            + responsiveness_term
            - quota_penalty
        )

        # Expected GPU Intensity categorization
        if "GPU_INTENSIVE" in wl_str:
            gpu_intensity = "HIGH"
        elif "CPU_BOUND" in wl_str or "IO_BOUND" in wl_str:
            gpu_intensity = "LOW"
        else:
            gpu_intensity = "MEDIUM"

        predicted_gpu_behavior = {
            "estimated_latency_ms": raw_pred_latency,
            "effective_latency_ms": effective_latency,
            "estimated_memory_mb": pred_mem,
            "expected_gpu_intensity": gpu_intensity,
            "confidence_score": confidence,
            "ml_priority_adjustment": ml_priority_adjustment,
        }

        return ScoreBreakdown(
            job_id=job.job_id,
            tenant_id=job.tenant_id,
            priority_term=priority_term,
            aging_term=aging_term,
            fairness_term=fairness_term,
            responsiveness_term=responsiveness_term,
            quota_penalty=quota_penalty,
            total_score=total_score,
            quota_consumption_ratio=quota_ratio,
            is_quota_eligible=is_eligible,
            ml_priority_adjustment=ml_priority_adjustment,
            predicted_workload_type=wl_str,
            predicted_gpu_behavior=predicted_gpu_behavior,
        )

    def calculate_candidate_score(
        self,
        job: GPUJob,
        current_time_ms: float,
        v_bar: float,
        quota_ratio: float,
    ) -> float:
        """Fast numeric dispatch score evaluation for candidate filtering in hot path."""
        if self.predictor is not None:
            return self.calculate_score(job, current_time_ms, v_bar).total_score

        tenant = self.tenants[job.tenant_id]

        # 1. Aging term
        wait_time = current_time_ms - job.arrival_time
        if wait_time < 0.0:
            wait_time = 0.0
        aging_term = self.aging_rate * (wait_time / self.starvation_threshold_ms)

        # Update tenant effective priority for telemetry
        tenant.effective_priority = tenant.priority + (aging_term / max(1e-6, self.priority_factor))

        # 2. Priority term
        priority_term = self.priority_factor * float(tenant.priority)

        # 3. Fairness lag term
        v_i = tenant.accumulated_gpu_service / max(1e-6, tenant.weight)
        fairness_term = self.fairness_factor * (v_bar - v_i)

        # 4. Responsiveness term
        q_base = self.quantum.base_quantum_ms
        eff_lat = job.remaining_gpu_time
        if eff_lat < 0.1:
            eff_lat = 0.1
        resp_term = self.responsiveness_factor * (q_base / (eff_lat if eff_lat > q_base else q_base))

        # 5. Quota penalty
        q_pen = self.quota_factor * (quota_ratio if quota_ratio < 1.0 else 1.0)

        return priority_term + aging_term + fairness_term + resp_term - q_pen

    def calculate_quantum(
        self,
        tenant: Tenant,
        active_tenants: List[Tenant],
        current_time_ms: float,
        candidate_job: Optional[GPUJob] = None,
        v_bar: Optional[float] = None,
    ) -> float:
        """Determines the allocated time quantum slice Q_i(t) for the dispatched tenant.
        
        Q_target = Q_base * (w_i / mean_w) * (1.0 + tanh( F_i(t) / Q_base )) * psi(c)
        clamped to [Q_min, Q_max].
        """
        q_base = self.quantum.base_quantum_ms
        q_min = self.quantum.min_quantum_ms
        q_max = self.quantum.max_quantum_ms

        if not active_tenants:
            return q_base

        mean_w = sum(t.weight for t in active_tenants) / len(active_tenants)
        weight_ratio = tenant.weight / max(1e-6, mean_w)

        # Service lag modulation
        if v_bar is None:
            total_service = sum(t.accumulated_gpu_service for t in active_tenants)
            total_weight = sum(t.weight for t in active_tenants)
            v_bar = total_service / max(1e-6, total_weight)
        v_i = tenant.accumulated_gpu_service / max(1e-6, tenant.weight)
        fairness_lag = v_bar - v_i

        modulation = 1.0 + math.tanh(fairness_lag / max(1.0, q_base))

        # Workload characteristic multiplier psi(c)
        if candidate_job is not None:
            if self.predictor is not None:
                try:
                    if hasattr(self.predictor, "predict"):
                        pred = self.predictor.predict(candidate_job)
                    elif hasattr(self.predictor, "predict_job"):
                        pred = self.predictor.predict_job(candidate_job)
                    else:
                        pred = None
                    psi = self.get_workload_multiplier(pred.classification if pred else candidate_job.workload_type)
                except Exception:
                    psi = self.get_workload_multiplier(candidate_job.workload_type)
            else:
                psi = self.get_workload_multiplier(candidate_job.workload_type)
        else:
            psi = 1.0

        target_quantum = q_base * weight_ratio * modulation * psi

        clamped_quantum = max(q_min, min(q_max, target_quantum))

        # Check remaining quota headroom in current window
        if self.enable_quota_enforcement:
            allowed_gpu_time = (tenant.gpu_quota / 100.0) * tenant.quota_window_ms
            headroom = max(0.0, allowed_gpu_time - tenant.accumulated_gpu_time_window_ms)
            if headroom > 0.0:
                # Cap slice strictly by available quota headroom so quota is not exceeded
                clamped_quantum = min(clamped_quantum, headroom)

        return clamped_quantum

    def schedule(self, current_time_ms: float) -> Optional[GPUJob]:
        """Selects the highest scoring eligible inference job across tenant queues.
        
        Args:
            current_time_ms: Current simulation clock in milliseconds.
            
        Returns:
            The winning GPUJob instance, or None if no eligible job is available.
        """
        active_tenant_ids = self.queue_manager.get_active_tenant_ids()
        if not active_tenant_ids:
            return None

        # Filter tenants that are quota-eligible and record quota ratios
        eligible_tenants: List[Tenant] = []
        quota_ratios: Dict[str, float] = {}
        for t_id in active_tenant_ids:
            t = self.tenants.get(t_id)
            if t is None:
                continue
            ratio, is_eligible = self.compute_quota_ratio(t, current_time_ms)
            if is_eligible:
                eligible_tenants.append(t)
                quota_ratios[t_id] = ratio

        if not eligible_tenants:
            # All active tenants currently throttled by quota
            logger.info("All active tenants currently throttled by quota limits; awaiting window reset.")
            return None

        # Compute average normalized virtual service v_bar across eligible active tenants
        total_service = sum(t.accumulated_gpu_service for t in eligible_tenants)
        total_weight = sum(t.weight for t in eligible_tenants)
        v_bar = total_service / max(1e-6, total_weight)

        # Score head candidate jobs
        best_job: Optional[GPUJob] = None
        highest_score = -float("inf")

        # Deterministic candidate evaluation
        for tenant in eligible_tenants:
            candidate_job = self.queue_manager.peek(tenant.tenant_id)
            if candidate_job is None:
                continue

            q_ratio = quota_ratios[tenant.tenant_id]
            score = self.calculate_candidate_score(candidate_job, current_time_ms, v_bar, q_ratio)

            # Tie-breaking: higher score, earlier arrival, lexicographical tenant_id
            if score > highest_score:
                highest_score = score
                best_job = candidate_job
            elif math.isclose(score, highest_score, abs_tol=1e-9):
                if best_job is not None:
                    if candidate_job.arrival_time < best_job.arrival_time - 1e-9:
                        highest_score = score
                        best_job = candidate_job
                    elif math.isclose(candidate_job.arrival_time, best_job.arrival_time, abs_tol=1e-9):
                        # Strict deterministic tie-breaker by (tenant_id, job_id)
                        cand_key = (candidate_job.tenant_id, candidate_job.job_id)
                        best_key = (best_job.tenant_id, best_job.job_id)
                        if cand_key < best_key:
                            highest_score = score
                            best_job = candidate_job

        if best_job is None:
            return None

        # Materialize full score breakdown for the single winning job
        best_score_breakdown = self.calculate_score(best_job, current_time_ms, v_bar)

        # Calculate dynamic quantum slice for winner
        winning_tenant = self.tenants[best_job.tenant_id]
        allocated_quantum = self.calculate_quantum(
            winning_tenant,
            eligible_tenants,
            current_time_ms,
            candidate_job=best_job,
            v_bar=v_bar,
        )
        self._last_allocated_quantum[best_job.job_id] = allocated_quantum

        # Dequeue the selected job from tenant ingress queue
        dispatched_job = self.queue_manager.dequeue(best_job.tenant_id)
        if dispatched_job is None:
            return None

        self.total_dispatches += 1
        self.decision_counter += 1

        # Record atomic scheduling decision
        deficit_snapshot = {
            t.tenant_id: round(v_bar - (t.accumulated_gpu_service / max(1e-6, t.weight)), 2)
            for t in eligible_tenants
        }

        reason_for_sel = (
            f"MAX_COMPOSITE_SCORE: score={best_score_breakdown.total_score:.3f} "
            f"(P={best_score_breakdown.priority_term:.2f}, "
            f"Aging={best_score_breakdown.aging_term:.2f}, "
            f"FairLag={best_score_breakdown.fairness_term:.2f}, "
            f"Resp={best_score_breakdown.responsiveness_term:.2f}, "
            f"QPen={best_score_breakdown.quota_penalty:.2f})"
        )

        score_dict = {
            "priority_term": best_score_breakdown.priority_term,
            "aging_term": best_score_breakdown.aging_term,
            "fairness_term": best_score_breakdown.fairness_term,
            "responsiveness_term": best_score_breakdown.responsiveness_term,
            "quota_penalty": best_score_breakdown.quota_penalty,
            "total_score": best_score_breakdown.total_score,
            "ml_priority_adjustment": best_score_breakdown.ml_priority_adjustment,
        }

        tenant_quota_usage = winning_tenant.accumulated_gpu_time_window_ms
        job_wait_time = max(0.0, current_time_ms - dispatched_job.arrival_time)

        decision = SchedulingDecision(
            decision_id=f"DEC-{self.decision_counter:06d}",
            timestamp=current_time_ms,
            selected_job=dispatched_job,
            selected_tenant_id=dispatched_job.tenant_id,
            allocated_quantum_ms=allocated_quantum,
            reason="MAX_COMPOSITE_SCORE",
            active_tenants_count=len(eligible_tenants),
            deficit_snapshot=deficit_snapshot,
            metadata={
                "score": best_score_breakdown.total_score,
                "priority_term": best_score_breakdown.priority_term,
                "aging_term": best_score_breakdown.aging_term,
                "fairness_term": best_score_breakdown.fairness_term,
                "responsiveness_term": best_score_breakdown.responsiveness_term,
                "quota_penalty": best_score_breakdown.quota_penalty,
                "ml_priority_adjustment": best_score_breakdown.ml_priority_adjustment,
            },
            predicted_workload_type=best_score_breakdown.predicted_workload_type,
            predicted_gpu_behavior=best_score_breakdown.predicted_gpu_behavior,
            current_tenant_gpu_service=winning_tenant.accumulated_gpu_service,
            tenant_weight=winning_tenant.weight,
            tenant_quota_usage=tenant_quota_usage,
            priority=float(winning_tenant.priority),
            waiting_time=job_wait_time,
            scheduling_score=best_score_breakdown.total_score,
            reason_for_selection=reason_for_sel,
            score_breakdown=score_dict,
        )
        self.decisions_history.append(decision)

        return dispatched_job

    def get_allocated_quantum(self, job: GPUJob) -> float:
        """Returns the dynamically allocated quantum slice for the given job."""
        if job.job_id in self._last_allocated_quantum:
            return self._last_allocated_quantum[job.job_id]
        t = self.tenants.get(job.tenant_id)
        if t:
            return self.calculate_quantum(
                t,
                list(self.tenants.values()),
                job.arrival_time,
                candidate_job=job,
            )
        return self.quantum.base_quantum_ms

    def check_starvation_violations(
        self,
        current_time_ms: float,
        critical_multiplier: float = 10.0,
        strict: bool = False,
    ) -> List[str]:
        """Audits waiting jobs for starvation violations exceeding critical thresholds."""
        starved_jobs: List[str] = []
        critical_threshold = critical_multiplier * self.starvation_threshold_ms

        for t_id in self.queue_manager.get_active_tenant_ids():
            q = self.queue_manager.get_queue(t_id)
            head = q.peek()
            if head is not None:
                wait_time = max(0.0, current_time_ms - head.arrival_time)
                if wait_time >= critical_threshold:
                    msg = (
                        f"Critical starvation violation: Job '{head.job_id}' (tenant='{t_id}') "
                        f"waited {wait_time:.1f}ms >= critical bound {critical_threshold:.1f}ms."
                    )
                    logger.critical(msg)
                    starved_jobs.append(head.job_id)
                    if strict:
                        raise SchedulerStarvationError(msg)
        return starved_jobs

    def on_job_completed(
        self,
        job: GPUJob,
        actual_gpu_time_ms: float,
        completion_time_ms: float,
    ) -> None:
        """Records telemetry on successful inference job completion."""
        self._last_allocated_quantum.pop(job.job_id, None)
        if job.tenant_id in self.dispatch_history:
            record = TenantDispatchRecord(
                timestamp_ms=completion_time_ms,
                job_id=job.job_id,
                allocated_quantum_ms=actual_gpu_time_ms,
                actual_executed_ms=actual_gpu_time_ms,
                preempted=False,
            )
            self.dispatch_history[job.tenant_id].append(record)

    def on_job_preempted(self, job: GPUJob, current_time_ms: float) -> None:
        """Records telemetry when a job is preempted at quantum boundary."""
        allocated_q = self._last_allocated_quantum.pop(job.job_id, self.quantum.base_quantum_ms)
        if job.tenant_id in self.dispatch_history:
            record = TenantDispatchRecord(
                timestamp_ms=current_time_ms,
                job_id=job.job_id,
                allocated_quantum_ms=allocated_q,
                actual_executed_ms=allocated_q,
                preempted=True,
            )
            self.dispatch_history[job.tenant_id].append(record)

    def reset(self) -> None:
        """Resets all dynamic state, history, and tenant counters."""
        super().reset()
        self._last_allocated_quantum.clear()
        self.decisions_history.clear()
        self.decision_counter = 0
        for t_id in self.dispatch_history:
            self.dispatch_history[t_id].clear()


# Aliases for architectural compatibility
PDWFQScheduler = GPUFairScheduler
FairScheduler = GPUFairScheduler
