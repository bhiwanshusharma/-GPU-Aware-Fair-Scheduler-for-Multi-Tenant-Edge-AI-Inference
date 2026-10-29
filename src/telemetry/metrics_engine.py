"""Complete, standalone, scheduler-agnostic metrics engine for multi-tenant GPU scheduling.

Operates strictly on execution traces and job telemetry records without knowing
the internal implementation or policies of individual schedulers.

Computes the 13 required scheduling metrics:
1. GPU utilization (%)
2. GPU idle percentage (%)
3. Throughput (jobs/sec)
4. Average waiting time (ms)
5. Average response time (ms)
6. Average turnaround time (ms)
7. Per-tenant GPU service (ms)
8. Per-tenant waiting time (ms)
9. Number of preemptions
10. Scheduling overhead (ms and %)
11. Deadline misses (count and %)
12. Starvation events (count and per-tenant breakdown)
13. Jain's Fairness Index: J = (sum(x_i)^2) / (n * sum(x_i^2))
"""

import os
import json
import csv
import io
import math
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union

from src.core.job import GPUJob
from src.simulation.trace import SimulationTrace, TraceEvent, TraceEventType

logger = logging.getLogger(__name__)


def compute_jains_fairness_unweighted(services: Union[Dict[str, float], List[float]]) -> float:
    """Computes Jain's Fairness Index according to the exact user specification:
    
        J = (sum(x_i)^2) / (n * sum(x_i^2))
        
    where x_i represents the GPU service received by tenant i, and n is the total
    number of tenants.
    
    Handles all division-by-zero cases gracefully:
    - n <= 1: returns 1.0 (single tenant or empty set is fair by definition).
    - sum(x_i^2) == 0: returns 1.0 (all tenants received equal zero service).
    
    Returns:
        Float bounded in [1/n, 1.0].
    """
    if isinstance(services, dict):
        x = [float(v) for v in services.values()]
    else:
        x = [float(v) for v in services]

    n = len(x)
    if n <= 1:
        return 1.0

    sum_x = sum(x)
    sum_x_sq = sum(val * val for val in x)

    # Division-by-zero protection
    if sum_x_sq <= 0.0 or sum_x <= 0.0:
        return 1.0

    jain = (sum_x * sum_x) / (n * sum_x_sq)
    return float(max(1.0 / n, min(1.0, jain)))


@dataclass
class JobTelemetryRecord:
    """Standardized record of an individual inference job's execution telemetry."""
    job_id: str
    tenant_id: str
    arrival_time_ms: float
    start_time_ms: float
    completion_time_ms: float
    actual_gpu_time_ms: float
    waiting_time_ms: float
    response_time_ms: float
    turnaround_time_ms: float
    deadline_ms: Optional[float] = None
    deadline_missed: bool = False
    starved: bool = False
    number_of_preemptions: int = 0


@dataclass
class TenantTelemetryRecord:
    """Standardized telemetry record for an individual tenant."""
    tenant_id: str
    gpu_service_ms: float
    avg_waiting_time_ms: float
    completed_jobs_count: int
    starvation_events_count: int
    deadline_misses_count: int


@dataclass
class SystemMetricsReport:
    """Complete consolidated performance metrics produced by the Metrics Engine."""
    # 1. GPU utilization
    gpu_utilization_pct: float
    # 2. GPU idle percentage
    gpu_idle_pct: float
    # 3. Throughput (jobs/sec)
    throughput_jobs_per_sec: float
    # 4. Average waiting time (ms)
    avg_waiting_time_ms: float
    # 5. Average response time (ms)
    avg_response_time_ms: float
    # 6. Average turnaround time (ms)
    avg_turnaround_time_ms: float
    # 7. Per-tenant GPU service (ms)
    per_tenant_gpu_service_ms: Dict[str, float]
    # 8. Per-tenant waiting time (ms)
    per_tenant_waiting_time_ms: Dict[str, float]
    # 9. Number of preemptions
    preemption_count: int
    # 10. Scheduling overhead
    scheduling_overhead_ms: float
    scheduling_overhead_pct: float
    # 11. Deadline misses
    deadline_misses: int
    deadline_miss_rate_pct: float
    # 12. Starvation events
    starvation_events: int
    per_tenant_starvation_events: Dict[str, int]
    # 13. Jain's Fairness Index
    jains_fairness_index: float

    # System overview counters
    total_simulation_time_ms: float = 0.0
    total_busy_time_ms: float = 0.0
    total_idle_time_ms: float = 0.0
    total_jobs_submitted: int = 0
    total_jobs_completed: int = 0
    starvation_threshold_ms: float = 100.0

    # Detailed breakdowns
    tenant_records: Dict[str, TenantTelemetryRecord] = field(default_factory=dict)
    job_records: List[JobTelemetryRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Converts complete metrics report into a clean serializable dictionary."""
        return {
            "summary": {
                "gpu_utilization_pct": round(self.gpu_utilization_pct, 4),
                "gpu_idle_pct": round(self.gpu_idle_pct, 4),
                "throughput_jobs_per_sec": round(self.throughput_jobs_per_sec, 4),
                "avg_waiting_time_ms": round(self.avg_waiting_time_ms, 4),
                "avg_response_time_ms": round(self.avg_response_time_ms, 4),
                "avg_turnaround_time_ms": round(self.avg_turnaround_time_ms, 4),
                "preemption_count": self.preemption_count,
                "scheduling_overhead_ms": round(self.scheduling_overhead_ms, 4),
                "scheduling_overhead_pct": round(self.scheduling_overhead_pct, 4),
                "deadline_misses": self.deadline_misses,
                "deadline_miss_rate_pct": round(self.deadline_miss_rate_pct, 4),
                "starvation_events": self.starvation_events,
                "jains_fairness_index": round(self.jains_fairness_index, 6),
                "total_simulation_time_ms": round(self.total_simulation_time_ms, 4),
                "total_busy_time_ms": round(self.total_busy_time_ms, 4),
                "total_idle_time_ms": round(self.total_idle_time_ms, 4),
                "total_jobs_submitted": self.total_jobs_submitted,
                "total_jobs_completed": self.total_jobs_completed,
                "starvation_threshold_ms": round(self.starvation_threshold_ms, 2),
            },
            "per_tenant_gpu_service_ms": {
                t: round(s, 4) for t, s in self.per_tenant_gpu_service_ms.items()
            },
            "per_tenant_waiting_time_ms": {
                t: round(w, 4) for t, w in self.per_tenant_waiting_time_ms.items()
            },
            "per_tenant_starvation_events": dict(self.per_tenant_starvation_events),
            "tenants": {
                t_id: asdict(rec) for t_id, rec in self.tenant_records.items()
            },
            "jobs": [asdict(j) for j in self.job_records],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serializes report into machine-readable JSON format."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_csv_summary(self) -> str:
        """Serializes high-level summary metrics into CSV format."""
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["metric", "value", "unit"])
        summary = self.to_dict()["summary"]
        units = {
            "gpu_utilization_pct": "%",
            "gpu_idle_pct": "%",
            "throughput_jobs_per_sec": "jobs/sec",
            "avg_waiting_time_ms": "ms",
            "avg_response_time_ms": "ms",
            "avg_turnaround_time_ms": "ms",
            "preemption_count": "count",
            "scheduling_overhead_ms": "ms",
            "scheduling_overhead_pct": "%",
            "deadline_misses": "count",
            "deadline_miss_rate_pct": "%",
            "starvation_events": "count",
            "jains_fairness_index": "index",
            "total_simulation_time_ms": "ms",
            "total_busy_time_ms": "ms",
            "total_idle_time_ms": "ms",
            "total_jobs_submitted": "count",
            "total_jobs_completed": "count",
            "starvation_threshold_ms": "ms",
        }
        for k, v in summary.items():
            writer.writerow([k, v, units.get(k, "")])
        return out.getvalue()

    def to_csv_tenants(self) -> str:
        """Serializes per-tenant telemetry breakdown into CSV format."""
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow([
            "tenant_id",
            "gpu_service_ms",
            "avg_waiting_time_ms",
            "completed_jobs_count",
            "starvation_events_count",
            "deadline_misses_count",
        ])
        for t_id, rec in self.tenant_records.items():
            writer.writerow([
                rec.tenant_id,
                rec.gpu_service_ms,
                rec.avg_waiting_time_ms,
                rec.completed_jobs_count,
                rec.starvation_events_count,
                rec.deadline_misses_count,
            ])
        return out.getvalue()

    def to_csv_jobs(self) -> str:
        """Serializes per-job telemetry breakdown into CSV format."""
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow([
            "job_id",
            "tenant_id",
            "arrival_time_ms",
            "start_time_ms",
            "completion_time_ms",
            "actual_gpu_time_ms",
            "waiting_time_ms",
            "response_time_ms",
            "turnaround_time_ms",
            "deadline_ms",
            "deadline_missed",
            "starved",
            "number_of_preemptions",
        ])
        for j in self.job_records:
            writer.writerow([
                j.job_id,
                j.tenant_id,
                j.arrival_time_ms,
                j.start_time_ms,
                j.completion_time_ms,
                j.actual_gpu_time_ms,
                j.waiting_time_ms,
                j.response_time_ms,
                j.turnaround_time_ms,
                j.deadline_ms if j.deadline_ms is not None else "",
                j.deadline_missed,
                j.starved,
                j.number_of_preemptions,
            ])
        return out.getvalue()

    def save_json(self, filepath: str) -> None:
        """Persists metrics report to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    def save_csv(
        self,
        summary_path: str,
        tenant_path: Optional[str] = None,
        job_path: Optional[str] = None,
    ) -> None:
        """Persists metrics tables to CSV files."""
        os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(self.to_csv_summary())

        if tenant_path:
            os.makedirs(os.path.dirname(os.path.abspath(tenant_path)), exist_ok=True)
            with open(tenant_path, "w", encoding="utf-8") as f:
                f.write(self.to_csv_tenants())

        if job_path:
            os.makedirs(os.path.dirname(os.path.abspath(job_path)), exist_ok=True)
            with open(job_path, "w", encoding="utf-8") as f:
                f.write(self.to_csv_jobs())

    def format_summary_table(self) -> str:
        """Renders an ASCII summary table for terminal logging."""
        lines = [
            "===================== System Metrics Summary =====================",
            f"Total Simulation Time   : {self.total_simulation_time_ms:.2f} ms",
            f"Completed / Submitted   : {self.total_jobs_completed} / {self.total_jobs_submitted}",
            f"1.  GPU Utilization     : {self.gpu_utilization_pct:.2f} %",
            f"2.  GPU Idle Percentage : {self.gpu_idle_pct:.2f} %",
            f"3.  System Throughput   : {self.throughput_jobs_per_sec:.2f} jobs/sec",
            f"4.  Average Waiting Time: {self.avg_waiting_time_ms:.2f} ms",
            f"5.  Average Response    : {self.avg_response_time_ms:.2f} ms",
            f"6.  Average Turnaround  : {self.avg_turnaround_time_ms:.2f} ms",
            f"7.  Jain's Fairness (J) : {self.jains_fairness_index:.6f}",
            f"8.  Preemption Count    : {self.preemption_count}",
            f"9.  Scheduling Overhead : {self.scheduling_overhead_ms:.2f} ms ({self.scheduling_overhead_pct:.2f} %)",
            f"10. Deadline Misses     : {self.deadline_misses} ({self.deadline_miss_rate_pct:.1f} %)",
            f"11. Starvation Events   : {self.starvation_events} (threshold: {self.starvation_threshold_ms:.1f} ms)",
            "------------------ Per-Tenant Resource Allocation -----------------",
        ]
        for t_id, svc in sorted(self.per_tenant_gpu_service_ms.items()):
            wait = self.per_tenant_waiting_time_ms.get(t_id, 0.0)
            starve = self.per_tenant_starvation_events.get(t_id, 0)
            lines.append(f"  Tenant '{t_id:<16}': GPU={svc:>7.1f} ms | Wait={wait:>6.1f} ms | Starvations={starve}")
        lines.append("==================================================================")
        return "\n".join(lines)


def _get_field(obj: Any, field_name: str, default: Any = None) -> Any:
    """Safely extracts a field from either an object attribute or dictionary key."""
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    return getattr(obj, field_name, default)


class MetricsEngine:
    """Independent, scheduler-agnostic metrics engine.
    
    Operates strictly on discrete event traces and job telemetry records.
    Never inspects or couples with scheduler policy algorithms.
    """

    def __init__(self, starvation_threshold_ms: float = 100.0) -> None:
        if starvation_threshold_ms <= 0.0:
            raise ValueError(f"starvation_threshold_ms must be positive, got {starvation_threshold_ms}")
        self.starvation_threshold_ms = float(starvation_threshold_ms)

    def compute(
        self,
        trace: Optional[Union[SimulationTrace, List[TraceEvent], List[Dict[str, Any]]]] = None,
        jobs: Optional[List[Union[GPUJob, Dict[str, Any]]]] = None,
        total_simulation_time_ms: Optional[float] = None,
        tenants: Optional[Union[List[str], Dict[str, Any]]] = None,
        idle_periods: Optional[List[Tuple[float, float, float]]] = None,
        context_switch_overhead_ms: Optional[float] = None,
        starvation_threshold_ms: Optional[float] = None,
    ) -> SystemMetricsReport:
        """Calculates all 13 metrics from an execution trace and/or job telemetry.
        
        Args:
            trace: Recorded SimulationTrace or sequence of TraceEvents.
            jobs: List of completed or submitted GPUJob objects or telemetry dicts.
            total_simulation_time_ms: Explicit total simulation duration in ms.
            tenants: Explicit list of tenant IDs or dictionary of Tenants.
            idle_periods: Optional list of (start_ms, end_ms, duration_ms) idle intervals.
            context_switch_overhead_ms: Constant context switch penalty in ms.
            starvation_threshold_ms: Starvation threshold in ms (defaults to instance setting).
            
        Returns:
            SystemMetricsReport containing all 13 verified metrics.
        """
        starve_thresh = (
            float(starvation_threshold_ms)
            if starvation_threshold_ms is not None
            else self.starvation_threshold_ms
        )

        # 1. Discover all active or registered tenants
        discovered_tenants: set[str] = set()
        if tenants is not None:
            if isinstance(tenants, dict):
                discovered_tenants.update(tenants.keys())
            else:
                discovered_tenants.update(tenants)

        # 2. Extract job telemetry records
        job_records: List[JobTelemetryRecord] = []
        if jobs:
            for item in jobs:
                rec = self._convert_job_to_record(item, starve_thresh)
                job_records.append(rec)
                discovered_tenants.add(rec.tenant_id)
        elif trace is not None:
            # Reconstruct job records from trace events if jobs list is not provided
            job_records = self._extract_jobs_from_trace(trace, starve_thresh)
            for j in job_records:
                discovered_tenants.add(j.tenant_id)

        # 3. Determine total simulation time
        total_time_ms = 0.0
        if total_simulation_time_ms is not None and total_simulation_time_ms >= 0.0:
            total_time_ms = float(total_simulation_time_ms)
        elif job_records:
            total_time_ms = max(j.completion_time_ms for j in job_records)
        elif trace is not None:
            events = trace.events if isinstance(trace, SimulationTrace) else trace
            if events:
                total_time_ms = max(
                    getattr(e, "timestamp_ms", e.get("timestamp_ms", 0.0)) for e in events
                )

        # 4. Extract hardware duty cycle (Busy, Idle, Overhead)
        total_busy_time_ms = sum(j.actual_gpu_time_ms for j in job_records)
        total_overhead_ms = self._calculate_overhead(trace, job_records, context_switch_overhead_ms)
        total_idle_time_ms = self._calculate_idle_time(trace, idle_periods, total_time_ms, total_busy_time_ms, total_overhead_ms)

        # Ensure total_time_ms is at least the sum of busy + idle + overhead if not explicitly set
        if total_simulation_time_ms is None:
            total_time_ms = max(total_time_ms, total_busy_time_ms + total_overhead_ms + total_idle_time_ms)

        # 5. Metric 1 & 2: GPU Utilization and Idle Percentage (Guarded against division by zero)
        if total_time_ms > 0.0:
            gpu_utilization_pct = max(0.0, min(100.0, (total_busy_time_ms / total_time_ms) * 100.0))
            gpu_idle_pct = max(0.0, min(100.0, (total_idle_time_ms / total_time_ms) * 100.0))
        else:
            gpu_utilization_pct = 0.0
            gpu_idle_pct = 0.0

        # 6. Metric 3: Throughput (jobs/sec) (Guarded against division by zero)
        total_completed = len(job_records)
        if total_time_ms > 0.0:
            throughput_jobs_per_sec = total_completed / (total_time_ms / 1000.0)
        else:
            throughput_jobs_per_sec = 0.0

        # 7. Metric 4, 5, 6: Average Waiting, Response, and Turnaround Times
        if total_completed > 0:
            avg_waiting_time_ms = sum(j.waiting_time_ms for j in job_records) / total_completed
            avg_response_time_ms = sum(j.response_time_ms for j in job_records) / total_completed
            avg_turnaround_time_ms = sum(j.turnaround_time_ms for j in job_records) / total_completed
        else:
            avg_waiting_time_ms = 0.0
            avg_response_time_ms = 0.0
            avg_turnaround_time_ms = 0.0

        # 8. Metric 7 & 8: Per-tenant GPU service and waiting time
        per_tenant_gpu_service: Dict[str, float] = {t: 0.0 for t in discovered_tenants}
        per_tenant_waiting_lists: Dict[str, List[float]] = {t: [] for t in discovered_tenants}
        per_tenant_starvations: Dict[str, int] = {t: 0 for t in discovered_tenants}
        per_tenant_deadlines: Dict[str, int] = {t: 0 for t in discovered_tenants}
        per_tenant_counts: Dict[str, int] = {t: 0 for t in discovered_tenants}

        for j in job_records:
            per_tenant_gpu_service[j.tenant_id] = (
                per_tenant_gpu_service.get(j.tenant_id, 0.0) + j.actual_gpu_time_ms
            )
            per_tenant_waiting_lists.setdefault(j.tenant_id, []).append(j.waiting_time_ms)
            per_tenant_counts[j.tenant_id] = per_tenant_counts.get(j.tenant_id, 0) + 1
            if j.starved:
                per_tenant_starvations[j.tenant_id] = per_tenant_starvations.get(j.tenant_id, 0) + 1
            if j.deadline_missed:
                per_tenant_deadlines[j.tenant_id] = per_tenant_deadlines.get(j.tenant_id, 0) + 1

        per_tenant_waiting_time: Dict[str, float] = {}
        for t in discovered_tenants:
            w_list = per_tenant_waiting_lists.get(t, [])
            per_tenant_waiting_time[t] = (sum(w_list) / len(w_list)) if w_list else 0.0

        # 9. Metric 9: Number of Preemptions
        preemption_count = self._calculate_preemptions(trace, job_records)

        # 10. Metric 10: Scheduling Overhead (ms and %)
        if total_time_ms > 0.0:
            scheduling_overhead_pct = max(0.0, min(100.0, (total_overhead_ms / total_time_ms) * 100.0))
        else:
            scheduling_overhead_pct = 0.0

        # 11. Metric 11: Deadline Misses (Count and Rate)
        deadline_misses = sum(1 for j in job_records if j.deadline_missed)
        jobs_with_deadline = sum(1 for j in job_records if j.deadline_ms is not None)
        if jobs_with_deadline > 0:
            deadline_miss_rate_pct = (deadline_misses / jobs_with_deadline) * 100.0
        else:
            deadline_miss_rate_pct = 0.0

        # 12. Metric 12: Starvation Events
        starvation_events = sum(1 for j in job_records if j.starved)

        # 13. Metric 13: Jain's Fairness Index
        jains_fairness_index = compute_jains_fairness_unweighted(per_tenant_gpu_service)

        # Build Tenant Records
        tenant_records: Dict[str, TenantTelemetryRecord] = {}
        for t in discovered_tenants:
            tenant_records[t] = TenantTelemetryRecord(
                tenant_id=t,
                gpu_service_ms=per_tenant_gpu_service.get(t, 0.0),
                avg_waiting_time_ms=per_tenant_waiting_time.get(t, 0.0),
                completed_jobs_count=per_tenant_counts.get(t, 0),
                starvation_events_count=per_tenant_starvations.get(t, 0),
                deadline_misses_count=per_tenant_deadlines.get(t, 0),
            )

        return SystemMetricsReport(
            gpu_utilization_pct=gpu_utilization_pct,
            gpu_idle_pct=gpu_idle_pct,
            throughput_jobs_per_sec=throughput_jobs_per_sec,
            avg_waiting_time_ms=avg_waiting_time_ms,
            avg_response_time_ms=avg_response_time_ms,
            avg_turnaround_time_ms=avg_turnaround_time_ms,
            per_tenant_gpu_service_ms=per_tenant_gpu_service,
            per_tenant_waiting_time_ms=per_tenant_waiting_time,
            preemption_count=preemption_count,
            scheduling_overhead_ms=total_overhead_ms,
            scheduling_overhead_pct=scheduling_overhead_pct,
            deadline_misses=deadline_misses,
            deadline_miss_rate_pct=deadline_miss_rate_pct,
            starvation_events=starvation_events,
            per_tenant_starvation_events=per_tenant_starvations,
            jains_fairness_index=jains_fairness_index,
            total_simulation_time_ms=total_time_ms,
            total_busy_time_ms=total_busy_time_ms,
            total_idle_time_ms=total_idle_time_ms,
            total_jobs_submitted=len(jobs) if jobs is not None else total_completed,
            total_jobs_completed=total_completed,
            starvation_threshold_ms=starve_thresh,
            tenant_records=tenant_records,
            job_records=job_records,
        )

    def _convert_job_to_record(self, item: Any, starve_thresh: float) -> JobTelemetryRecord:
        """Normalizes a GPUJob instance or dictionary into a JobTelemetryRecord."""
        if isinstance(item, GPUJob):
            arr = float(item.arrival_time)
            start = float(item.start_time) if item.start_time is not None else arr
            comp = float(item.completion_time) if item.completion_time is not None else start + item.actual_gpu_time_ms
            actual_exec = float(item.actual_gpu_time_ms)
            wait = float(item.waiting_time_ms)
            resp = max(0.0, start - arr)
            turn = float(item.turnaround_time_ms)
            dl = float(item.deadline) if item.deadline is not None else None
            is_miss = bool(dl is not None and comp > dl)
            is_starved = bool(wait > starve_thresh)
            preempts = int(item.number_of_preemptions)
            return JobTelemetryRecord(
                job_id=str(item.job_id),
                tenant_id=str(item.tenant_id),
                arrival_time_ms=arr,
                start_time_ms=start,
                completion_time_ms=comp,
                actual_gpu_time_ms=actual_exec,
                waiting_time_ms=wait,
                response_time_ms=resp,
                turnaround_time_ms=turn,
                deadline_ms=dl,
                deadline_missed=is_miss,
                starved=is_starved,
                number_of_preemptions=preempts,
            )
        else:
            # Dictionary format
            arr = float(item.get("arrival_time_ms", item.get("arrival_time", 0.0)))
            actual_exec = float(item.get("actual_gpu_time_ms", item.get("execution_time_ms", item.get("estimated_gpu_time", 0.0))))
            start = float(item.get("start_time_ms", item.get("start_time", arr)))
            comp = float(item.get("completion_time_ms", item.get("completion_time", start + actual_exec)))
            wait = float(item.get("waiting_time_ms", max(0.0, comp - arr - actual_exec)))
            resp = float(item.get("response_time_ms", max(0.0, start - arr)))
            turn = float(item.get("turnaround_time_ms", max(0.0, comp - arr)))
            raw_dl = item.get("deadline_ms", item.get("deadline", None))
            dl = float(raw_dl) if raw_dl is not None else None
            is_miss = bool(dl is not None and comp > dl)
            is_starved = bool(wait > starve_thresh)
            preempts = int(item.get("number_of_preemptions", item.get("preemptions", 0)))
            return JobTelemetryRecord(
                job_id=str(item.get("job_id", "job")),
                tenant_id=str(item.get("tenant_id", "default")),
                arrival_time_ms=arr,
                start_time_ms=start,
                completion_time_ms=comp,
                actual_gpu_time_ms=actual_exec,
                waiting_time_ms=wait,
                response_time_ms=resp,
                turnaround_time_ms=turn,
                deadline_ms=dl,
                deadline_missed=is_miss,
                starved=is_starved,
                number_of_preemptions=preempts,
            )

    def _extract_jobs_from_trace(
        self,
        trace: Union[SimulationTrace, List[Any]],
        starve_thresh: float,
    ) -> List[JobTelemetryRecord]:
        """Reconstructs job lifecycle telemetry solely from observed trace events."""
        events = trace.events if isinstance(trace, SimulationTrace) else trace
        job_first_dispatch: Dict[str, float] = {}
        job_completion: Dict[str, float] = {}
        job_tenant: Dict[str, str] = {}
        job_total_exec: Dict[str, float] = {}
        job_preempts: Dict[str, int] = {}
        job_details: Dict[str, Dict[str, Any]] = {}

        for ev in events:
            t_ms = float(_get_field(ev, "timestamp_ms", 0.0))
            ev_type = _get_field(ev, "event_type", None)
            j_id = _get_field(ev, "job_id", None)
            t_id = _get_field(ev, "tenant_id", None)
            dtls = _get_field(ev, "details", {}) or {}

            if not j_id:
                continue

            if t_id and j_id not in job_tenant:
                job_tenant[j_id] = t_id

            ev_type_str = str(getattr(ev_type, "name", ev_type))

            # Dispatch or Resume
            if ev_type in (TraceEventType.DISPATCH, TraceEventType.RESUME) or ev_type_str in ("DISPATCH", "RESUME", "TraceEventType.DISPATCH", "TraceEventType.RESUME"):
                if j_id not in job_first_dispatch:
                    job_first_dispatch[j_id] = t_ms
                q_dur = float(dtls.get("allocated_quantum", 0.0))
                job_total_exec[j_id] = job_total_exec.get(j_id, 0.0) + q_dur

            # Preemption
            elif ev_type in (TraceEventType.QUANTUM_EXPIRED, TraceEventType.PREEMPTION) or ev_type_str in ("QUANTUM_EXPIRED", "PREEMPTION", "TraceEventType.QUANTUM_EXPIRED", "TraceEventType.PREEMPTION"):
                job_preempts[j_id] = job_preempts.get(j_id, 0) + 1

            # Completion
            elif ev_type == TraceEventType.COMPLETION or ev_type_str in ("COMPLETION", "TraceEventType.COMPLETION"):
                job_completion[j_id] = t_ms
                job_details[j_id] = dtls

        records: List[JobTelemetryRecord] = []
        for j_id, comp_time in job_completion.items():
            t_id = job_tenant.get(j_id, "unknown")
            dtls = job_details.get(j_id, {})
            start_time = job_first_dispatch.get(j_id, 0.0)
            arr_time = float(dtls.get("arrival_time", dtls.get("arrival_time_ms", 0.0)))
            exec_time = float(dtls.get("actual_gpu_time_ms", job_total_exec.get(j_id, comp_time - start_time)))
            wait_time = float(dtls.get("waiting_time", max(0.0, comp_time - arr_time - exec_time)))
            turn_time = float(dtls.get("turnaround_time", max(0.0, comp_time - arr_time)))
            resp_time = max(0.0, start_time - arr_time)
            raw_dl = dtls.get("deadline", dtls.get("deadline_ms", None))
            dl = float(raw_dl) if raw_dl is not None else None

            records.append(JobTelemetryRecord(
                job_id=j_id,
                tenant_id=t_id,
                arrival_time_ms=arr_time,
                start_time_ms=start_time,
                completion_time_ms=comp_time,
                actual_gpu_time_ms=exec_time,
                waiting_time_ms=wait_time,
                response_time_ms=resp_time,
                turnaround_time_ms=turn_time,
                deadline_ms=dl,
                deadline_missed=bool(dl is not None and comp_time > dl),
                starved=bool(wait_time > starve_thresh),
                number_of_preemptions=job_preempts.get(j_id, 0),
            ))

        return records

    def _calculate_overhead(
        self,
        trace: Optional[Any],
        jobs: List[JobTelemetryRecord],
        context_switch_overhead_ms: Optional[float],
    ) -> float:
        """Determines total scheduling/context switch overhead."""
        if trace is not None:
            events = trace.events if isinstance(trace, SimulationTrace) else trace
            overhead = 0.0
            found_explicit_delay = False
            for ev in events:
                dtls = _get_field(ev, "details", {}) or {}
                ev_type = _get_field(ev, "event_type", None)
                ev_type_str = str(getattr(ev_type, "name", ev_type))
                if "switch_delay" in dtls:
                    overhead += float(dtls["switch_delay"])
                    found_explicit_delay = True
                elif ev_type == TraceEventType.CONTEXT_SWITCH or ev_type_str in ("CONTEXT_SWITCH", "TraceEventType.CONTEXT_SWITCH"):
                    overhead += float(dtls.get("overhead_ms", context_switch_overhead_ms or 0.5))
                    found_explicit_delay = True
            if found_explicit_delay:
                return overhead

        # Fallback to preemption/job count * overhead parameter
        if context_switch_overhead_ms is not None and context_switch_overhead_ms > 0.0:
            total_switches = len(jobs) + sum(j.number_of_preemptions for j in jobs)
            return float(total_switches * context_switch_overhead_ms)

        return 0.0

    def _calculate_idle_time(
        self,
        trace: Optional[Any],
        idle_periods: Optional[List[Tuple[float, float, float]]],
        total_time_ms: float,
        busy_time_ms: float,
        overhead_ms: float,
    ) -> float:
        """Determines total GPU idle duration."""
        if idle_periods:
            return sum(float(dur) for _, _, dur in idle_periods)

        if isinstance(trace, SimulationTrace) and trace.idle_periods:
            return trace.total_idle_time_ms

        if trace is not None:
            events = trace.events if isinstance(trace, SimulationTrace) else trace
            idle_accum = 0.0
            last_idle_start: Optional[float] = None
            for ev in events:
                ev_type = _get_field(ev, "event_type", None)
                ev_type_str = str(getattr(ev_type, "name", ev_type))
                t_ms = float(_get_field(ev, "timestamp_ms", 0.0))
                if ev_type == TraceEventType.IDLE_START or ev_type_str in ("IDLE_START", "TraceEventType.IDLE_START"):
                    last_idle_start = t_ms
                elif (ev_type == TraceEventType.IDLE_END or ev_type_str in ("IDLE_END", "TraceEventType.IDLE_END")) and last_idle_start is not None:
                    idle_accum += max(0.0, t_ms - last_idle_start)
                    last_idle_start = None
            if idle_accum > 0.0:
                return idle_accum

        # Fallback to residual duration
        return max(0.0, total_time_ms - busy_time_ms - overhead_ms)

    def _calculate_preemptions(self, trace: Optional[Any], jobs: List[JobTelemetryRecord]) -> int:
        """Determines total number of preemptions."""
        if isinstance(trace, SimulationTrace) and trace.preemption_events:
            return len(trace.preemption_events)

        if trace is not None:
            events = trace.events if isinstance(trace, SimulationTrace) else trace
            preempt_count = 0
            for ev in events:
                ev_type = _get_field(ev, "event_type", None)
                ev_type_str = str(getattr(ev_type, "name", ev_type))
                if ev_type in (TraceEventType.QUANTUM_EXPIRED, TraceEventType.PREEMPTION) or ev_type_str in (
                    "QUANTUM_EXPIRED", "PREEMPTION", "TraceEventType.QUANTUM_EXPIRED", "TraceEventType.PREEMPTION"
                ):
                    preempt_count += 1
            if preempt_count > 0:
                return preempt_count

        return sum(j.number_of_preemptions for j in jobs)
