"""Reproducible multi-tenant benchmarking execution engine.

Evaluates and compares the 5 required scheduling policies:
1. FIFO (FIFOScheduler)
2. Round Robin (RoundRobinScheduler)
3. Priority Scheduling (PriorityScheduler)
4. Weighted Round Robin (WeightedRoundRobinScheduler)
5. GPU-Aware Fair Scheduler (GPUFairScheduler / PDWFQ)

Runs identical, bit-for-bit cloned workloads across all schedulers using
deterministic random seeds. Computes the complete 13 scheduling metrics via
the standalone MetricsEngine, verifies fairness and latency trade-offs, and
persists raw experiment results (JSON, CSV, execution traces).
"""

import os
import sys
import copy
import json
import csv
import io
import time
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Sequence

from src.core.types import SchedulingQuantum
from src.core.job import GPUJob
from src.core.tenant import Tenant
from src.scheduler.base import BaseScheduler
from src.scheduler.fifo import FIFOScheduler
from src.scheduler.round_robin import RoundRobinScheduler
from src.scheduler.priority import PriorityScheduler
from src.scheduler.weighted_round_robin import WeightedRoundRobinScheduler
from src.scheduler.fair_scheduler import GPUFairScheduler
from src.simulation.simulator import GPUSimulator
from src.telemetry.metrics_engine import MetricsEngine, SystemMetricsReport
from src.benchmarks.scenarios import (
    BenchmarkScenario,
    get_scenario,
    get_all_scenarios,
    SCENARIO_FACTORIES,
)

logger = logging.getLogger(__name__)

# Standard canonical policy keys and human-readable names
CANONICAL_POLICIES = [
    "fifo",
    "round_robin",
    "priority",
    "weighted_round_robin",
    "gpu_fair",
]

POLICY_DISPLAY_NAMES = {
    "fifo": "FIFO",
    "round_robin": "Round Robin",
    "priority": "Priority",
    "weighted_round_robin": "Weighted RR",
    "gpu_fair": "GPU-Aware Fair",
}


def normalize_policy_name(policy_name: str) -> str:
    """Normalizes various user input aliases into canonical policy keys."""
    p = policy_name.strip().lower().replace("-", "_").replace(" ", "_")
    if p in ("fifo", "first_in_first_out"):
        return "fifo"
    if p in ("round_robin", "rr", "roundrobin"):
        return "round_robin"
    if p in ("priority", "prio", "strict_priority", "priority_scheduling"):
        return "priority"
    if p in ("weighted_round_robin", "wrr", "weighted_rr"):
        return "weighted_round_robin"
    if p in ("gpu_fair", "fair", "pdwfq", "gpu_aware_fair", "gpu_aware_fair_scheduler", "fair_scheduler"):
        return "gpu_fair"
    raise ValueError(
        f"Unknown scheduling policy '{policy_name}'. Supported: {CANONICAL_POLICIES}"
    )


def create_scheduler(
    policy: str,
    tenants: Dict[str, Tenant],
    quantum_ms: float = 20.0,
    min_quantum_ms: float = 5.0,
    max_quantum_ms: float = 100.0,
    context_switch_overhead_ms: float = 0.5,
    aging_rate: float = 0.15,
    starvation_threshold_ms: float = 100.0,
    predictor: Optional[Any] = None,
) -> BaseScheduler:
    """Instantiates a fresh scheduler instance for the given policy with isolated tenant models."""
    norm = normalize_policy_name(policy)
    quantum = SchedulingQuantum(
        base_quantum_ms=quantum_ms,
        min_quantum_ms=min(min_quantum_ms, quantum_ms),
        max_quantum_ms=max(max_quantum_ms, quantum_ms),
        aging_alpha=aging_rate,
        starvation_threshold_ms=starvation_threshold_ms,
    )

    if norm == "fifo":
        return FIFOScheduler(tenants=tenants, quantum=quantum)
    elif norm == "round_robin":
        return RoundRobinScheduler(tenants=tenants, quantum=quantum)
    elif norm == "priority":
        return PriorityScheduler(tenants=tenants, quantum=quantum)
    elif norm == "weighted_round_robin":
        return WeightedRoundRobinScheduler(tenants=tenants, quantum=quantum)
    elif norm == "gpu_fair":
        return GPUFairScheduler(
            tenants=tenants,
            quantum=quantum,
            aging_rate=aging_rate,
            starvation_threshold_ms=starvation_threshold_ms,
            context_switch_overhead_ms=context_switch_overhead_ms,
            predictor=predictor,
        )
    else:
        raise ValueError(f"Unsupported canonical policy: {norm}")


@dataclass
class PolicyEvaluationResult:
    """Evaluation output for a single scheduler on a specific scenario."""
    policy_key: str
    policy_name: str
    scenario_id: str
    metrics: SystemMetricsReport
    execution_trace: str
    completed_jobs_count: int
    total_jobs_count: int
    simulation_duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_key": self.policy_key,
            "policy_name": self.policy_name,
            "scenario_id": self.scenario_id,
            "simulation_duration_ms": round(self.simulation_duration_ms, 4),
            "completed_jobs_count": self.completed_jobs_count,
            "total_jobs_count": self.total_jobs_count,
            "execution_trace": self.execution_trace,
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class ScenarioBenchmarkResult:
    """Comparative benchmarking results for all evaluated schedulers on a single scenario."""
    scenario_id: str
    scenario_name: str
    description: str
    tenants: Dict[str, Tenant]
    job_count: int
    random_seed: int
    policy_results: Dict[str, PolicyEvaluationResult] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "description": self.description,
            "tenants": {t_id: asdict(t) for t_id, t in self.tenants.items()},
            "job_count": self.job_count,
            "random_seed": self.random_seed,
            "policies": {
                p_key: res.to_dict() for p_key, res in self.policy_results.items()
            },
        }

    def format_comparison_table(self) -> str:
        """Renders an ASCII table comparing all evaluated schedulers side-by-side."""
        headers = ["Metric"] + [
            self.policy_results[p].policy_name
            for p in self.policy_results.keys()
        ]

        rows = [
            ("GPU Utilization (%)", [f"{res.metrics.gpu_utilization_pct:.1f}%" for res in self.policy_results.values()]),
            ("Throughput (jobs/sec)", [f"{res.metrics.throughput_jobs_per_sec:.2f}" for res in self.policy_results.values()]),
            ("Avg Waiting Time (ms)", [f"{res.metrics.avg_waiting_time_ms:.2f}" for res in self.policy_results.values()]),
            ("Avg Response Time (ms)", [f"{res.metrics.avg_response_time_ms:.2f}" for res in self.policy_results.values()]),
            ("Avg Turnaround Time (ms)", [f"{res.metrics.avg_turnaround_time_ms:.2f}" for res in self.policy_results.values()]),
            ("Jain's Fairness Index", [f"{res.metrics.jains_fairness_index:.4f}" for res in self.policy_results.values()]),
            ("Preemptions", [str(res.metrics.preemption_count) for res in self.policy_results.values()]),
            ("Deadline Misses", [str(res.metrics.deadline_misses) for res in self.policy_results.values()]),
            ("Starvation Events", [str(res.metrics.starvation_events) for res in self.policy_results.values()]),
            ("Scheduling Overhead (ms)", [f"{res.metrics.scheduling_overhead_ms:.2f}" for res in self.policy_results.values()]),
        ]

        col_widths = [26] + [max(14, len(h) + 2) for h in headers[1:]]

        lines = [
            f"=== Benchmark Comparison: {self.scenario_name} (Scenario ID: {self.scenario_id}) ===",
            f"Description: {self.description}",
            f"Workload   : {self.job_count} jobs across {len(self.tenants)} tenants (Seed: {self.random_seed})",
            "-" * (sum(col_widths) + len(col_widths) * 3 + 1),
        ]

        header_line = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths)) + " |"
        sep_line = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"
        lines.append(header_line)
        lines.append(sep_line)

        for label, vals in rows:
            row_items = [label] + vals
            lines.append("| " + " | ".join(f"{v:<{w}}" for v, w in zip(row_items, col_widths)) + " |")

        lines.append("-" * (sum(col_widths) + len(col_widths) * 3 + 1))
        return "\n".join(lines)


@dataclass
class BenchmarkSuiteResult:
    """Master benchmark suite containing results across multiple evaluation scenarios."""
    scenarios: Dict[str, ScenarioBenchmarkResult] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    config_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "config": self.config_metadata,
            "scenario_count": len(self.scenarios),
            "scenarios": {s_id: s_res.to_dict() for s_id, s_res in self.scenarios.items()},
        }

    def to_csv_summary(self) -> str:
        """Serializes comparative metrics for all scenarios and policies into flat CSV."""
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow([
            "scenario_id",
            "scenario_name",
            "policy_key",
            "policy_name",
            "gpu_utilization_pct",
            "throughput_jobs_per_sec",
            "avg_waiting_time_ms",
            "avg_response_time_ms",
            "avg_turnaround_time_ms",
            "jains_fairness_index",
            "preemption_count",
            "deadline_misses",
            "starvation_events",
            "scheduling_overhead_ms",
            "completed_jobs",
            "total_jobs",
            "simulation_duration_ms",
        ])

        for s_id, s_res in self.scenarios.items():
            for p_key, p_res in s_res.policy_results.items():
                m = p_res.metrics
                writer.writerow([
                    s_id,
                    s_res.scenario_name,
                    p_key,
                    p_res.policy_name,
                    round(m.gpu_utilization_pct, 4),
                    round(m.throughput_jobs_per_sec, 4),
                    round(m.avg_waiting_time_ms, 4),
                    round(m.avg_response_time_ms, 4),
                    round(m.avg_turnaround_time_ms, 4),
                    round(m.jains_fairness_index, 6),
                    m.preemption_count,
                    m.deadline_misses,
                    m.starvation_events,
                    round(m.scheduling_overhead_ms, 4),
                    p_res.completed_jobs_count,
                    p_res.total_jobs_count,
                    round(p_res.simulation_duration_ms, 4),
                ])
        return out.getvalue()

    def to_csv_tenant_service(self) -> str:
        """Serializes per-tenant GPU service allocations into flat CSV."""
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow([
            "scenario_id",
            "policy_key",
            "tenant_id",
            "gpu_service_ms",
            "avg_waiting_time_ms",
            "completed_jobs_count",
            "starvation_events_count",
        ])

        for s_id, s_res in self.scenarios.items():
            for p_key, p_res in s_res.policy_results.items():
                m = p_res.metrics
                for t_id, s_ms in m.per_tenant_gpu_service_ms.items():
                    t_rec = m.tenant_records.get(t_id)
                    wait_ms = t_rec.avg_waiting_time_ms if t_rec else m.per_tenant_waiting_time_ms.get(t_id, 0.0)
                    completed = t_rec.completed_jobs_count if t_rec else 0
                    starved = t_rec.starvation_events_count if t_rec else m.per_tenant_starvation_events.get(t_id, 0)
                    writer.writerow([
                        s_id,
                        p_key,
                        t_id,
                        round(s_ms, 4),
                        round(wait_ms, 4),
                        completed,
                        starved,
                    ])
        return out.getvalue()

    def export_results(self, output_dir: Union[str, Path], save_traces: bool = True) -> Dict[str, str]:
        """Saves machine-readable raw experiment results, CSV tables, and trace files."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        traces_dir = out_path / "traces"
        if save_traces:
            traces_dir.mkdir(parents=True, exist_ok=True)

        saved_files: Dict[str, str] = {}

        # 1. Raw Master JSON
        json_file = out_path / "benchmark_results.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        saved_files["json"] = str(json_file)

        # 2. Comparative Summary CSV
        summary_csv = out_path / "benchmark_summary.csv"
        with open(summary_csv, "w", encoding="utf-8") as f:
            f.write(self.to_csv_summary())
        saved_files["summary_csv"] = str(summary_csv)

        # 3. Per-Tenant Service CSV
        tenant_csv = out_path / "tenant_service_breakdown.csv"
        with open(tenant_csv, "w", encoding="utf-8") as f:
            f.write(self.to_csv_tenant_service())
        saved_files["tenant_csv"] = str(tenant_csv)

        # 4. Discrete Execution Traces
        if save_traces:
            trace_count = 0
            for s_id, s_res in self.scenarios.items():
                for p_key, p_res in s_res.policy_results.items():
                    trace_filename = traces_dir / f"{s_id}_{p_key}.trace.txt"
                    with open(trace_filename, "w", encoding="utf-8") as f:
                        f.write(p_res.execution_trace)
                    trace_count += 1
            saved_files["traces_dir"] = str(traces_dir)
            saved_files["trace_count"] = str(trace_count)

        return saved_files

    def format_full_summary(self) -> str:
        """Renders consolidated text summary across all scenarios."""
        sections = []
        for s_res in self.scenarios.values():
            sections.append(s_res.format_comparison_table())
        return "\n\n".join(sections)


class BenchmarkRunner:
    """Reproducible benchmarking framework orchestrating multi-tenant scheduling experiments.
    
    Guarantees:
    1. Bit-for-bit identical workloads across all schedulers using deep copying.
    2. Deterministic pseudo-random seeding for job arrivals and duration jitter.
    3. Calculation of all 13 verified metrics using standalone MetricsEngine.
    4. Comprehensive export to JSON, CSV, and plain-text execution traces.
    """

    def __init__(
        self,
        base_quantum_ms: float = 20.0,
        min_quantum_ms: float = 5.0,
        max_quantum_ms: float = 100.0,
        context_switch_overhead_ms: float = 0.5,
        aging_rate: float = 0.15,
        starvation_threshold_ms: float = 100.0,
        random_seed: int = 42,
        policies: Optional[Sequence[str]] = None,
        predictor: Optional[Any] = None,
    ) -> None:
        self.base_quantum_ms = float(base_quantum_ms)
        self.min_quantum_ms = float(min_quantum_ms)
        self.max_quantum_ms = float(max_quantum_ms)
        self.context_switch_overhead_ms = float(context_switch_overhead_ms)
        self.aging_rate = float(aging_rate)
        self.starvation_threshold_ms = float(starvation_threshold_ms)
        self.random_seed = int(random_seed)
        self.predictor = predictor

        if policies is None:
            self.policies = list(CANONICAL_POLICIES)
        else:
            self.policies = [normalize_policy_name(p) for p in policies]

        self.metrics_engine = MetricsEngine(starvation_threshold_ms=self.starvation_threshold_ms)

    def run_scenario(
        self,
        scenario: BenchmarkScenario,
        policies: Optional[Sequence[str]] = None,
        random_seed: Optional[int] = None,
    ) -> ScenarioBenchmarkResult:
        """Executes a single benchmark scenario across all configured scheduling policies.
        
        Args:
            scenario: BenchmarkScenario instance.
            policies: Optional subset of policies to run (defaults to instance policies).
            random_seed: Deterministic random seed (defaults to instance seed).
            
        Returns:
            ScenarioBenchmarkResult comparing all tested schedulers.
        """
        seed = self.random_seed if random_seed is None else int(random_seed)
        target_policies = (
            [normalize_policy_name(p) for p in policies]
            if policies is not None
            else self.policies
        )

        # 1. Generate master job sequence deterministically
        master_jobs = scenario.generate_jobs(seed)
        master_tenants = scenario.tenants

        scenario_res = ScenarioBenchmarkResult(
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.name,
            description=scenario.description,
            tenants=master_tenants,
            job_count=len(master_jobs),
            random_seed=seed,
        )

        logger.info(
            f"Running benchmark '{scenario.name}' ({scenario.scenario_id}) with "
            f"{len(master_jobs)} jobs across {len(master_tenants)} tenants..."
        )

        # 2. Run identical workload against each policy in isolation
        for p_key in target_policies:
            p_display = POLICY_DISPLAY_NAMES.get(p_key, p_key)

            # Deep-copy tenants and jobs to ensure strict zero-leakage isolation
            isolated_tenants = {
                t_id: copy.deepcopy(t) for t_id, t in master_tenants.items()
            }
            isolated_jobs = [copy.deepcopy(j) for j in master_jobs]

            # Instantiate scheduler
            scheduler = create_scheduler(
                policy=p_key,
                tenants=isolated_tenants,
                quantum_ms=self.base_quantum_ms,
                min_quantum_ms=self.min_quantum_ms,
                max_quantum_ms=self.max_quantum_ms,
                context_switch_overhead_ms=self.context_switch_overhead_ms,
                aging_rate=self.aging_rate,
                starvation_threshold_ms=self.starvation_threshold_ms,
                predictor=self.predictor,
            )

            # Create simulator and submit jobs
            sim = GPUSimulator(
                quantum_ms=self.base_quantum_ms,
                context_switch_overhead_ms=self.context_switch_overhead_ms,
                scheduler=scheduler,
            )

            for job in isolated_jobs:
                sim.submit_job(job)

            # Execute discrete event simulation to completion
            sim.run()

            # Compute verified metrics via standalone MetricsEngine
            report = self.metrics_engine.compute(
                trace=sim.trace,
                jobs=sim.completed_jobs,
                total_simulation_time_ms=sim.current_time_ms,
                tenants=isolated_tenants,
                idle_periods=sim.get_idle_periods(),
                context_switch_overhead_ms=self.context_switch_overhead_ms,
                starvation_threshold_ms=self.starvation_threshold_ms,
            )

            trace_text = sim.get_execution_trace_text(include_idle=False)

            policy_res = PolicyEvaluationResult(
                policy_key=p_key,
                policy_name=p_display,
                scenario_id=scenario.scenario_id,
                metrics=report,
                execution_trace=trace_text,
                completed_jobs_count=len(sim.completed_jobs),
                total_jobs_count=len(isolated_jobs),
                simulation_duration_ms=sim.current_time_ms,
            )

            scenario_res.policy_results[p_key] = policy_res

        return scenario_res

    def run_all(
        self,
        scenario_ids: Optional[Sequence[str]] = None,
        random_seed: Optional[int] = None,
    ) -> BenchmarkSuiteResult:
        """Executes all 10 standard benchmark scenarios (or a selected subset).
        
        Args:
            scenario_ids: Optional list of scenario IDs (e.g., ['scenario_1', 'scenario_2']).
            random_seed: Deterministic random seed.
            
        Returns:
            BenchmarkSuiteResult with full comparative evaluation.
        """
        seed = self.random_seed if random_seed is None else int(random_seed)

        if scenario_ids is None or "all" in [s.lower() for s in scenario_ids]:
            scenarios = get_all_scenarios()
        else:
            scenarios = [get_scenario(s_id) for s_id in scenario_ids]

        suite_res = BenchmarkSuiteResult(
            config_metadata={
                "base_quantum_ms": self.base_quantum_ms,
                "min_quantum_ms": self.min_quantum_ms,
                "max_quantum_ms": self.max_quantum_ms,
                "context_switch_overhead_ms": self.context_switch_overhead_ms,
                "aging_rate": self.aging_rate,
                "starvation_threshold_ms": self.starvation_threshold_ms,
                "random_seed": seed,
                "policies": list(self.policies),
            }
        )

        for sc in scenarios:
            res = self.run_scenario(sc, random_seed=seed)
            suite_res.scenarios[sc.scenario_id] = res

        return suite_res
