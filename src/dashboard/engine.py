"""Dashboard Simulation Engine API.

Provides a clean facade between monitoring interfaces (UI/CLI/Web) and the
underlying OS scheduling and simulation engines. Enforces strict architectural
separation: the dashboard UI never contains scheduling math or queue dispatch
logic; it solely invokes this engine.
"""

import copy
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple

from src.core.types import SchedulingQuantum, WorkloadClass
from src.core.job import GPUJob
from src.core.tenant import Tenant
from src.scheduler.base import BaseScheduler
from src.benchmarks.runner import create_scheduler, normalize_policy_name, CANONICAL_POLICIES
from src.simulation.simulator import GPUSimulator
from src.telemetry.metrics_engine import MetricsEngine, SystemMetricsReport
from src.visualization.gantt import parse_trace_to_slices, ExecutionSlice


@dataclass
class TenantConfigInput:
    """Configurable tenant parameters input from UI controls."""
    tenant_id: str
    name: str = ""
    weight: float = 1.0
    priority: int = 5
    gpu_quota_pct: float = 100.0


@dataclass
class SimulationRunConfig:
    """Comprehensive parameters for a simulated scheduler experiment."""
    tenants: List[TenantConfigInput] = field(default_factory=list)
    policy: str = "gpu_fair"
    base_quantum_ms: float = 20.0
    min_quantum_ms: float = 5.0
    max_quantum_ms: float = 100.0
    context_switch_overhead_ms: float = 0.5
    aging_rate: float = 0.15
    starvation_threshold_ms: float = 100.0
    workload_distribution: str = "balanced"  # balanced, bursty, mixed_cpu_gpu, short_long_bimodal, heavy_light
    workload_intensity: str = "normal"      # normal, high, stress
    total_jobs: int = 24
    random_seed: int = 42
    use_predictor: bool = False


@dataclass
class SimulationDashboardSnapshot:
    """Consolidated state snapshot consumed by the monitoring dashboard."""
    total_simulation_time_ms: float
    gpu_utilization_pct: float
    active_tenants: List[Dict[str, Any]]
    tenant_allocation_pct: Dict[str, float]
    tenant_service_ms: Dict[str, float]
    current_running_job: Optional[Dict[str, Any]]
    waiting_jobs: List[Dict[str, Any]]
    scheduling_decisions: List[Dict[str, Any]]
    execution_slices: List[ExecutionSlice]
    raw_trace: str
    metrics: SystemMetricsReport
    completed_jobs: List[Dict[str, Any]]
    policy_name: str
    scenario_description: str


class DashboardSimulationEngine:
    """Simulation and scheduler monitoring orchestration engine."""

    @staticmethod
    def generate_workload(
        tenant_ids: List[str],
        distribution: str = "balanced",
        intensity: str = "normal",
        total_jobs: int = 24,
        seed: int = 42,
    ) -> List[GPUJob]:
        """Generates deterministic, parameterized multi-tenant inference job sequence."""
        rng = random.Random(seed)
        jobs: List[GPUJob] = []

        if not tenant_ids:
            return jobs

        n_tenants = len(tenant_ids)
        jobs_per_tenant = max(1, total_jobs // n_tenants)

        # Arrival rate factor based on intensity
        arrival_step = 8.0 if intensity == "normal" else (4.0 if intensity == "high" else 2.0)

        job_idx = 0
        if distribution == "bursty":
            # Tenant 0 gets sudden burst of arrivals, other tenants get periodic arrivals
            burst_tenant = tenant_ids[0]
            for i in range(jobs_per_tenant * 2):
                t_arr = 5.0 + rng.uniform(0.0, 3.0)  # Sudden burst around 5ms
                jobs.append(
                    GPUJob(
                        job_id=f"BURST_{job_idx}",
                        tenant_id=burst_tenant,
                        model_name="yolov8_nano",
                        arrival_time=t_arr,
                        estimated_gpu_time=12.0,
                        remaining_gpu_time=12.0,
                        priority=5,
                        deadline=t_arr + 60.0,
                        workload_type=WorkloadClass.BALANCED,
                    )
                )
                job_idx += 1

            for t_id in tenant_ids[1:]:
                for i in range(jobs_per_tenant):
                    t_arr = i * arrival_step * 2.0 + rng.uniform(0.0, 2.0)
                    jobs.append(
                        GPUJob(
                            job_id=f"STEADY_{job_idx}",
                            tenant_id=t_id,
                            model_name="resnet50",
                            arrival_time=t_arr,
                            estimated_gpu_time=15.0,
                            remaining_gpu_time=15.0,
                            priority=5,
                            deadline=t_arr + 80.0,
                            workload_type=WorkloadClass.BALANCED,
                        )
                    )
                    job_idx += 1

        elif distribution == "short_long_bimodal":
            # Even tenants get long bulk jobs; odd tenants get short interactive jobs
            for i in range(jobs_per_tenant):
                for t_idx, t_id in enumerate(tenant_ids):
                    t_arr = i * arrival_step + rng.uniform(0.0, 1.5)
                    is_short = (t_idx % 2 == 1)
                    est_time = 4.0 if is_short else 35.0
                    model = "audio_keyword" if is_short else "llama_7b_int4"
                    jobs.append(
                        GPUJob(
                            job_id=f"{'SHORT' if is_short else 'BULK'}_{job_idx}",
                            tenant_id=t_id,
                            model_name=model,
                            arrival_time=t_arr,
                            estimated_gpu_time=est_time,
                            remaining_gpu_time=est_time,
                            priority=8 if is_short else 4,
                            deadline=t_arr + (30.0 if is_short else 120.0),
                            workload_type=WorkloadClass.IO_BOUND if is_short else WorkloadClass.GPU_INTENSIVE,
                        )
                    )
                    job_idx += 1

        elif distribution == "mixed_cpu_gpu":
            workload_types = [
                (WorkloadClass.GPU_INTENSIVE, "transformer_heavy", 28.0),
                (WorkloadClass.CPU_BOUND, "feature_extraction", 8.0),
                (WorkloadClass.IO_BOUND, "camera_stream", 5.0),
                (WorkloadClass.BALANCED, "mobilenet_v3", 14.0),
            ]
            for i in range(jobs_per_tenant):
                for t_idx, t_id in enumerate(tenant_ids):
                    t_arr = i * arrival_step + rng.uniform(0.0, 2.0)
                    w_type, m_name, est_time = workload_types[(i + t_idx) % len(workload_types)]
                    jobs.append(
                        GPUJob(
                            job_id=f"JOB_{job_idx}",
                            tenant_id=t_id,
                            model_name=m_name,
                            arrival_time=t_arr,
                            estimated_gpu_time=est_time,
                            remaining_gpu_time=est_time,
                            priority=5,
                            deadline=t_arr + 90.0,
                            workload_type=w_type,
                        )
                    )
                    job_idx += 1

        elif distribution == "heavy_light":
            # First tenant generates 70% of jobs with large compute; others submit light jobs
            heavy_t = tenant_ids[0]
            for i in range(int(total_jobs * 0.65)):
                t_arr = i * (arrival_step * 0.6) + rng.uniform(0.0, 1.0)
                jobs.append(
                    GPUJob(
                        job_id=f"HVY_{job_idx}",
                        tenant_id=heavy_t,
                        model_name="segmentation_unet",
                        arrival_time=t_arr,
                        estimated_gpu_time=24.0,
                        remaining_gpu_time=24.0,
                        priority=5,
                        deadline=t_arr + 100.0,
                        workload_type=WorkloadClass.GPU_INTENSIVE,
                    )
                )
                job_idx += 1

            for t_id in tenant_ids[1:]:
                for i in range(max(2, total_jobs // (n_tenants * 2))):
                    t_arr = i * arrival_step * 2.5 + rng.uniform(0.0, 2.0)
                    jobs.append(
                        GPUJob(
                            job_id=f"LGT_{job_idx}",
                            tenant_id=t_id,
                            model_name="light_classifier",
                            arrival_time=t_arr,
                            estimated_gpu_time=6.0,
                            remaining_gpu_time=6.0,
                            priority=5,
                            deadline=t_arr + 50.0,
                            workload_type=WorkloadClass.BALANCED,
                        )
                    )
                    job_idx += 1

        else:
            # Default: Balanced interleaved inference
            for i in range(jobs_per_tenant):
                for t_idx, t_id in enumerate(tenant_ids):
                    t_arr = i * arrival_step + t_idx * (arrival_step / n_tenants) + rng.uniform(0.0, 1.0)
                    est_time = 12.0 + rng.uniform(-2.0, 2.0)
                    jobs.append(
                        GPUJob(
                            job_id=f"J_{job_idx}",
                            tenant_id=t_id,
                            model_name="mobilenet_v3",
                            arrival_time=t_arr,
                            estimated_gpu_time=est_time,
                            remaining_gpu_time=est_time,
                            priority=5,
                            deadline=t_arr + 80.0,
                            workload_type=WorkloadClass.BALANCED,
                        )
                    )
                    job_idx += 1

        jobs.sort(key=lambda j: j.arrival_time)
        return jobs

    def run_experiment(self, config: SimulationRunConfig) -> SimulationDashboardSnapshot:
        """Executes a discrete simulation run and compiles monitoring snapshot."""
        # 1. Build Tenant dictionary
        tenant_map: Dict[str, Tenant] = {}
        for tc in config.tenants:
            t = Tenant(
                tenant_id=tc.tenant_id,
                name=tc.name or f"Tenant {tc.tenant_id}",
                weight=tc.weight,
                priority=tc.priority,
                gpu_quota=tc.gpu_quota_pct,
            )
            tenant_map[tc.tenant_id] = t

        # Fallback default if empty
        if not tenant_map:
            t1 = Tenant("tenant_A", name="Tenant A (Robotics)", weight=1.0, priority=5, gpu_quota=100.0)
            t2 = Tenant("tenant_B", name="Tenant B (Analytics)", weight=1.0, priority=5, gpu_quota=100.0)
            tenant_map = {t1.tenant_id: t1, t2.tenant_id: t2}

        # 2. Generate Workload
        tenant_ids = list(tenant_map.keys())
        jobs = self.generate_workload(
            tenant_ids=tenant_ids,
            distribution=config.workload_distribution,
            intensity=config.workload_intensity,
            total_jobs=config.total_jobs,
            seed=config.random_seed,
        )

        # 3. Instantiate Scheduler
        isolated_tenants = {t_id: copy.deepcopy(t) for t_id, t in tenant_map.items()}
        scheduler = create_scheduler(
            policy=config.policy,
            tenants=isolated_tenants,
            quantum_ms=config.base_quantum_ms,
            min_quantum_ms=config.min_quantum_ms,
            max_quantum_ms=config.max_quantum_ms,
            context_switch_overhead_ms=config.context_switch_overhead_ms,
            aging_rate=config.aging_rate,
            starvation_threshold_ms=config.starvation_threshold_ms,
        )

        # 4. Instantiate and run GPUSimulator
        sim = GPUSimulator(
            quantum_ms=config.base_quantum_ms,
            context_switch_overhead_ms=config.context_switch_overhead_ms,
            scheduler=scheduler,
        )

        cloned_jobs = [copy.deepcopy(j) for j in jobs]
        for j in cloned_jobs:
            sim.submit_job(j)

        sim.run()

        # 5. Extract Metrics via MetricsEngine
        engine = MetricsEngine(starvation_threshold_ms=config.starvation_threshold_ms)
        metrics = engine.compute(
            trace=sim.trace,
            jobs=sim.completed_jobs,
            total_simulation_time_ms=sim.current_time_ms,
            tenants=isolated_tenants,
            idle_periods=sim.get_idle_periods(),
            context_switch_overhead_ms=config.context_switch_overhead_ms,
            starvation_threshold_ms=config.starvation_threshold_ms,
        )

        raw_trace = sim.get_execution_trace_text(include_idle=False)
        execution_slices = parse_trace_to_slices(raw_trace)

        # 6. Extract Active Tenant States & Quota Info
        active_tenants_info = []
        tot_service = sum(metrics.per_tenant_gpu_service_ms.values()) or 1.0
        alloc_pct = {}

        for t_id, t in isolated_tenants.items():
            srv = metrics.per_tenant_gpu_service_ms.get(t_id, 0.0)
            pct = (srv / tot_service) * 100.0 if tot_service > 0 else 0.0
            alloc_pct[t_id] = round(pct, 2)

            t_rec = metrics.tenant_records.get(t_id)
            comp_cnt = t_rec.completed_jobs_count if t_rec else 0
            starv_cnt = t_rec.starvation_events_count if t_rec else 0
            wait_ms = t_rec.avg_waiting_time_ms if t_rec else 0.0

            # Quota consumption
            allowed_window_ms = (t.gpu_quota / 100.0) * t.quota_window_ms
            quota_used_pct = round(
                (t.accumulated_gpu_time_window_ms / max(0.1, allowed_window_ms)) * 100.0, 1
            )
            is_throttled = t.is_quota_exceeded(sim.current_time_ms)

            active_tenants_info.append({
                "tenant_id": t_id,
                "name": t.name,
                "weight": t.weight,
                "priority": t.priority,
                "gpu_quota_pct": t.gpu_quota,
                "service_ms": round(srv, 1),
                "allocation_pct": round(pct, 1),
                "quota_used_pct": quota_used_pct,
                "is_throttled": is_throttled,
                "completed_jobs": comp_cnt,
                "avg_waiting_time_ms": round(wait_ms, 1),
                "starvation_events": starv_cnt,
            })

        # 7. Extract Scheduling Decision History (Explainability)
        decisions_list: List[Dict[str, Any]] = []
        if hasattr(scheduler, "decisions_history"):
            for d in scheduler.decisions_history:
                decisions_list.append({
                    "decision_id": d.decision_id,
                    "timestamp_ms": round(d.timestamp, 1),
                    "selected_tenant": d.selected_tenant_id,
                    "selected_job": d.selected_job.job_id if d.selected_job else "None",
                    "quantum_ms": round(d.allocated_quantum_ms, 1),
                    "scheduling_score": round(d.scheduling_score, 4),
                    "predicted_type": d.predicted_workload_type,
                    "reason": d.reason,
                    "score_breakdown": {k: round(v, 4) for k, v in d.score_breakdown.items()},
                })

        # 8. Completed Jobs Telemetry
        completed_jobs_list = [
            {
                "job_id": j.job_id,
                "tenant_id": j.tenant_id,
                "arrival_time_ms": round(j.arrival_time, 1),
                "start_time_ms": round(j.start_time, 1) if j.start_time is not None else 0.0,
                "completion_time_ms": round(j.completion_time, 1) if j.completion_time is not None else 0.0,
                "execution_time_ms": round(j.actual_gpu_time_ms, 1),
                "waiting_time_ms": round(j.waiting_time_ms, 1),
                "response_time_ms": round(max(0.0, (j.start_time - j.arrival_time)) if j.start_time is not None else 0.0, 1),
                "turnaround_time_ms": round(j.turnaround_time_ms, 1),
                "preemptions": j.number_of_preemptions,
                "priority": j.priority,
                "model_name": j.model_name,
                "workload_type": j.workload_type.value if hasattr(j.workload_type, "value") else str(j.workload_type),
            }
            for j in sim.completed_jobs
        ]

        # 9. Current running job (last active job in discrete trace)
        current_job_dict = None
        if execution_slices:
            last_slice = execution_slices[-1]
            matching_job = next((j for j in sim.completed_jobs if j.job_id == last_slice.job_id), None)
            current_job_dict = {
                "job_id": last_slice.job_id,
                "tenant_id": last_slice.tenant_id,
                "slice_duration_ms": round(last_slice.duration_ms, 1),
                "event_type": last_slice.event_type,
                "model_name": matching_job.model_name if matching_job else "Inference Model",
                "remaining_time_ms": 0.0 if last_slice.event_type == "completion" else 5.0,
            }

        # 10. Waiting Jobs remaining or un-dispatched
        waiting_jobs_list = [
            {
                "job_id": j.job_id,
                "tenant_id": j.tenant_id,
                "arrival_time_ms": round(j.arrival_time, 1),
                "estimated_gpu_time_ms": round(j.estimated_gpu_time, 1),
                "priority": j.priority,
            }
            for j in jobs if j.job_id not in [cj["job_id"] for cj in completed_jobs_list]
        ]

        desc_map = {
            "balanced": "Symmetric arrivals of balanced computer vision & speech models",
            "bursty": "Poisson burst arrivals contending with periodic background telemetry",
            "mixed_cpu_gpu": "Heterogeneous GPU-intensive, CPU-bound, and I/O-bound models",
            "short_long_bimodal": "Bimodal mix: 4ms interactive audio vs 35ms LLM kernels",
            "heavy_light": "One dominant high-throughput tenant + sparse lightweight tenants",
        }

        return SimulationDashboardSnapshot(
            total_simulation_time_ms=round(sim.current_time_ms, 2),
            gpu_utilization_pct=round(metrics.gpu_utilization_pct, 2),
            active_tenants=active_tenants_info,
            tenant_allocation_pct=alloc_pct,
            tenant_service_ms={t: round(s, 2) for t, s in metrics.per_tenant_gpu_service_ms.items()},
            current_running_job=current_job_dict,
            waiting_jobs=waiting_jobs_list,
            scheduling_decisions=decisions_list,
            execution_slices=execution_slices,
            raw_trace=raw_trace,
            metrics=metrics,
            completed_jobs=completed_jobs_list,
            policy_name=normalize_policy_name(config.policy),
            scenario_description=desc_map.get(config.workload_distribution, "Multi-tenant workload"),
        )

    def compare_all_policies(self, config: SimulationRunConfig) -> Dict[str, SystemMetricsReport]:
        """Runs the exact same workload against all 5 scheduling policies for comparative evaluation."""
        reports: Dict[str, SystemMetricsReport] = {}
        for pol in CANONICAL_POLICIES:
            cfg_copy = copy.deepcopy(config)
            cfg_copy.policy = pol
            snap = self.run_experiment(cfg_copy)
            reports[pol] = snap.metrics
        return reports
