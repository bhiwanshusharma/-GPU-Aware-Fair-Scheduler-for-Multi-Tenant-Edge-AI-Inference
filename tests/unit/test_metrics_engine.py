"""Unit tests for the standalone, scheduler-agnostic MetricsEngine.

Verifies:
1. Complete calculation of all 13 required metrics on a manually calculable example.
2. Exact calculation of Jain's Fairness Index: J = (sum(x_i)^2) / (n * sum(x_i^2)).
3. Robust division-by-zero protection across all metrics (empty sets, 0 time, 0 service).
4. Pure trace-driven metric extraction with zero knowledge of scheduler implementations.
5. Serialization to machine-readable JSON and CSV format (both summary and granular tables).
6. File persistence and reload roundtrip for JSON and CSV artifacts.
7. End-to-end integration with discrete simulation execution traces.
"""

import os
import json
import csv
import pytest
from typing import Dict, List

from src.core.types import JobState
from src.core.job import GPUJob
from src.core.tenant import Tenant
from src.simulation.trace import SimulationTrace, TraceEventType
from src.telemetry.metrics_engine import (
    MetricsEngine,
    SystemMetricsReport,
    compute_jains_fairness_unweighted,
)


# =========================================================================
# 1. Manually Calculable Example with Deterministic Verification
# =========================================================================
def test_metrics_engine_manually_calculable_example():
    """Verifies all 13 metrics against hand-calculated, mathematically proven values.
    
    Manual Arithmetic Setup:
    -----------------------
    2 Tenants: Tenant_A, Tenant_B.
    Total simulation duration = 100.0 ms.
    Total switching overhead = 5.0 ms (two context switches of 2.5 ms).
    Total idle time = 15.0 ms (GPU idle from t=85.0 to t=100.0 ms).
    Total active compute time = 80.0 ms (Job A1: 40.0 ms, Job B1: 40.0 ms).
    Total time check: 80.0 (compute) + 5.0 (overhead) + 15.0 (idle) = 100.0 ms.

    Job A1:
      - arrival_time: 0.0 ms
      - start_time: 2.5 ms (dispatched after initial 2.5ms switch)
      - completion_time: 42.5 ms
      - actual_gpu_time: 40.0 ms
      - waiting_time: 2.5 ms
      - response_time: 2.5 ms (2.5 - 0.0)
      - turnaround_time: 42.5 ms (42.5 - 0.0)
      - deadline: 50.0 ms (met: 42.5 <= 50.0)
      - preemptions: 0

    Job B1:
      - arrival_time: 10.0 ms
      - start_time: 45.0 ms (dispatched after 2.5ms switch at t=42.5ms)
      - completion_time: 85.0 ms
      - actual_gpu_time: 40.0 ms
      - waiting_time: 35.0 ms (75.0 turnaround - 40.0 execution)
      - response_time: 35.0 ms (45.0 - 10.0)
      - turnaround_time: 75.0 ms (85.0 - 10.0)
      - deadline: 70.0 ms (MISSED: 85.0 > 70.0)
      - preemptions: 1 (quantum slice boundary)

    Starvation Threshold = 30.0 ms:
      - Job A1 wait = 2.5 ms <= 30.0 -> no starvation.
      - Job B1 wait = 35.0 ms > 30.0 -> 1 starvation event for Tenant_B.

    Exact Expected Metrics:
    1.  GPU utilization       = 80.0 / 100.0 * 100% = 80.0%
    2.  GPU idle percentage   = 15.0 / 100.0 * 100% = 15.0%
    3.  Throughput            = 2 jobs / (100.0 / 1000.0 s) = 20.0 jobs/sec
    4.  Avg waiting time      = (2.5 + 35.0) / 2 = 18.75 ms
    5.  Avg response time     = (2.5 + 35.0) / 2 = 18.75 ms
    6.  Avg turnaround time   = (42.5 + 75.0) / 2 = 58.75 ms
    7.  Per-tenant GPU service= Tenant_A: 40.0 ms, Tenant_B: 40.0 ms
    8.  Per-tenant wait time  = Tenant_A: 2.5 ms, Tenant_B: 35.0 ms
    9.  Number of preemptions = 1
    10. Scheduling overhead   = 5.0 ms (5.0 / 100.0 * 100% = 5.0%)
    11. Deadline misses       = 1 (Job B1 missed out of 2 jobs = 50.0%)
    12. Starvation events     = 1 (Tenant_B: 1, Tenant_A: 0)
    13. Jain's Fairness Index = (40 + 40)^2 / (2 * (40^2 + 40^2)) = 6400 / 6400 = 1.000000
    """
    # Create jobs matching manual arithmetic
    job_a = GPUJob("A1", "tenant_A", "model_a", arrival_time=0.0, estimated_gpu_time=40.0, remaining_gpu_time=0.0, deadline=50.0)
    job_a.start_time = 2.5
    job_a.completion_time = 42.5
    job_a.actual_gpu_time_ms = 40.0
    job_a.accumulated_waiting_time = 2.5
    job_a.accumulated_execution_time = 40.0
    job_a.number_of_preemptions = 0
    job_a.status = JobState.COMPLETED

    job_b = GPUJob("B1", "tenant_B", "model_b", arrival_time=10.0, estimated_gpu_time=40.0, remaining_gpu_time=0.0, deadline=70.0)
    job_b.start_time = 45.0
    job_b.completion_time = 85.0
    job_b.actual_gpu_time_ms = 40.0
    job_b.accumulated_waiting_time = 35.0
    job_b.accumulated_execution_time = 40.0
    job_b.number_of_preemptions = 1
    job_b.status = JobState.COMPLETED

    jobs = [job_a, job_b]

    # Create trace with matching context switches and idle periods
    trace = SimulationTrace()
    trace.record_event(0.0, TraceEventType.DISPATCH, tenant_id="tenant_A", job_id="A1", details={"switch_delay": 2.5, "allocated_quantum": 40.0})
    trace.record_event(42.5, TraceEventType.COMPLETION, tenant_id="tenant_A", job_id="A1")
    trace.record_event(42.5, TraceEventType.DISPATCH, tenant_id="tenant_B", job_id="B1", details={"switch_delay": 2.5, "allocated_quantum": 20.0})
    trace.record_event(65.0, TraceEventType.QUANTUM_EXPIRED, tenant_id="tenant_B", job_id="B1")
    trace.record_event(65.0, TraceEventType.RESUME, tenant_id="tenant_B", job_id="B1", details={"switch_delay": 0.0, "allocated_quantum": 20.0})
    trace.record_event(85.0, TraceEventType.COMPLETION, tenant_id="tenant_B", job_id="B1")
    trace.record_idle_period(start_ms=85.0, end_ms=100.0)

    engine = MetricsEngine(starvation_threshold_ms=30.0)
    report = engine.compute(
        trace=trace,
        jobs=jobs,
        total_simulation_time_ms=100.0,
        tenants=["tenant_A", "tenant_B"],
    )

    # 1. GPU Utilization
    assert report.gpu_utilization_pct == pytest.approx(80.0, abs=1e-5)
    # 2. GPU Idle Percentage
    assert report.gpu_idle_pct == pytest.approx(15.0, abs=1e-5)
    # 3. Throughput
    assert report.throughput_jobs_per_sec == pytest.approx(20.0, abs=1e-5)
    # 4. Average Waiting Time
    assert report.avg_waiting_time_ms == pytest.approx(18.75, abs=1e-5)
    # 5. Average Response Time
    assert report.avg_response_time_ms == pytest.approx(18.75, abs=1e-5)
    # 6. Average Turnaround Time
    assert report.avg_turnaround_time_ms == pytest.approx(58.75, abs=1e-5)
    # 7. Per-tenant GPU Service
    assert report.per_tenant_gpu_service_ms["tenant_A"] == pytest.approx(40.0, abs=1e-5)
    assert report.per_tenant_gpu_service_ms["tenant_B"] == pytest.approx(40.0, abs=1e-5)
    # 8. Per-tenant Waiting Time
    assert report.per_tenant_waiting_time_ms["tenant_A"] == pytest.approx(2.5, abs=1e-5)
    assert report.per_tenant_waiting_time_ms["tenant_B"] == pytest.approx(35.0, abs=1e-5)
    # 9. Number of Preemptions
    assert report.preemption_count == 1
    # 10. Scheduling Overhead
    assert report.scheduling_overhead_ms == pytest.approx(5.0, abs=1e-5)
    assert report.scheduling_overhead_pct == pytest.approx(5.0, abs=1e-5)
    # 11. Deadline Misses
    assert report.deadline_misses == 1
    assert report.deadline_miss_rate_pct == pytest.approx(50.0, abs=1e-5)
    # 12. Starvation Events
    assert report.starvation_events == 1
    assert report.per_tenant_starvation_events["tenant_A"] == 0
    assert report.per_tenant_starvation_events["tenant_B"] == 1
    # 13. Jain's Fairness Index
    assert report.jains_fairness_index == pytest.approx(1.0, abs=1e-5)


# =========================================================================
# 2. Jain's Fairness Index: Hand-Calculated Scenarios
# =========================================================================
def test_jains_fairness_hand_calculated_ratios():
    """Verifies Jain's Fairness Index formula on manually calculated distributions."""
    # Scenario A: 2 tenants, unequal service 60ms vs 20ms
    # sum(x) = 80, sum(x)^2 = 6400
    # sum(x^2) = 3600 + 400 = 4000
    # n * sum(x^2) = 2 * 4000 = 8000
    # J = 6400 / 8000 = 0.80
    services_2t = {"t1": 60.0, "t2": 20.0}
    j_2t = compute_jains_fairness_unweighted(services_2t)
    assert j_2t == pytest.approx(0.80, abs=1e-5)

    # Scenario B: 3 tenants, services 60ms, 30ms, 10ms
    # sum(x) = 100, sum(x)^2 = 10000
    # sum(x^2) = 3600 + 900 + 100 = 4600
    # n * sum(x^2) = 3 * 4600 = 13800
    # J = 10000 / 13800 = 100 / 138 = 0.72463768...
    services_3t = {"t1": 60.0, "t2": 30.0, "t3": 10.0}
    j_3t = compute_jains_fairness_unweighted(services_3t)
    assert j_3t == pytest.approx(100.0 / 138.0, abs=1e-5)

    # Scenario C: 4 tenants, complete monopolization by t1 (100ms, 0, 0, 0)
    # sum(x) = 100, sum(x)^2 = 10000
    # sum(x^2) = 10000
    # n * sum(x^2) = 4 * 10000 = 40000
    # J = 10000 / 40000 = 0.25 = 1/4 (theoretical lower bound for n=4)
    services_mono = {"t1": 100.0, "t2": 0.0, "t3": 0.0, "t4": 0.0}
    j_mono = compute_jains_fairness_unweighted(services_mono)
    assert j_mono == pytest.approx(0.25, abs=1e-5)

    # Scenario D: List input rather than dict
    j_list = compute_jains_fairness_unweighted([60.0, 20.0])
    assert j_list == pytest.approx(0.80, abs=1e-5)


# =========================================================================
# 3. Robust Division-By-Zero Protection
# =========================================================================
def test_metrics_engine_division_by_zero_handling():
    """All metrics safely handle empty sets, zero time, zero service, and missing deadlines."""
    engine = MetricsEngine()

    # Empty simulation input: 0 jobs, 0 time
    empty_report = engine.compute(
        trace=None,
        jobs=[],
        total_simulation_time_ms=0.0,
        tenants=[],
    )

    assert empty_report.gpu_utilization_pct == 0.0
    assert empty_report.gpu_idle_pct == 0.0
    assert empty_report.throughput_jobs_per_sec == 0.0
    assert empty_report.avg_waiting_time_ms == 0.0
    assert empty_report.avg_response_time_ms == 0.0
    assert empty_report.avg_turnaround_time_ms == 0.0
    assert empty_report.preemption_count == 0
    assert empty_report.scheduling_overhead_ms == 0.0
    assert empty_report.scheduling_overhead_pct == 0.0
    assert empty_report.deadline_misses == 0
    assert empty_report.deadline_miss_rate_pct == 0.0
    assert empty_report.starvation_events == 0
    assert empty_report.jains_fairness_index == 1.0  # Empty set is defined as fair

    # Single tenant (n = 1): Jain's index must be 1.0
    single_tenant_jain = compute_jains_fairness_unweighted({"t1": 50.0})
    assert single_tenant_jain == 1.0

    # Multiple tenants with all 0.0 service: sum(x^2) = 0 -> must return 1.0
    all_zero_jain = compute_jains_fairness_unweighted({"t1": 0.0, "t2": 0.0, "t3": 0.0})
    assert all_zero_jain == 1.0

    # Empty list for Jain's index: n = 0 -> must return 1.0
    empty_jain = compute_jains_fairness_unweighted([])
    assert empty_jain == 1.0


# =========================================================================
# 4. Pure Trace-Driven Operation (No Scheduler Dependency)
# =========================================================================
def test_metrics_engine_pure_trace_driven_operation():
    """Engine computes metrics purely from trace events without job objects or scheduler classes."""
    trace = SimulationTrace()
    # Job X: arrives at 0, dispatched at 0 for 20ms, completed at 20ms
    trace.record_event(
        timestamp_ms=0.0,
        event_type=TraceEventType.DISPATCH,
        tenant_id="tenant_X",
        job_id="X1",
        details={"allocated_quantum": 20.0, "switch_delay": 1.0},
    )
    trace.record_event(
        timestamp_ms=20.0,
        event_type=TraceEventType.COMPLETION,
        tenant_id="tenant_X",
        job_id="X1",
        details={"arrival_time": 0.0, "waiting_time": 0.0, "turnaround_time": 20.0, "actual_gpu_time_ms": 20.0},
    )

    # Job Y: arrives at 5, dispatched at 20 for 10ms, preempted at 30, resumed at 30, completed at 40
    trace.record_event(
        timestamp_ms=20.0,
        event_type=TraceEventType.DISPATCH,
        tenant_id="tenant_Y",
        job_id="Y1",
        details={"allocated_quantum": 10.0, "switch_delay": 1.0},
    )
    trace.record_event(
        timestamp_ms=30.0,
        event_type=TraceEventType.QUANTUM_EXPIRED,
        tenant_id="tenant_Y",
        job_id="Y1",
    )
    trace.record_event(
        timestamp_ms=30.0,
        event_type=TraceEventType.RESUME,
        tenant_id="tenant_Y",
        job_id="Y1",
        details={"allocated_quantum": 10.0, "switch_delay": 0.0},
    )
    trace.record_event(
        timestamp_ms=40.0,
        event_type=TraceEventType.COMPLETION,
        tenant_id="tenant_Y",
        job_id="Y1",
        details={"arrival_time": 5.0, "waiting_time": 15.0, "turnaround_time": 35.0, "actual_gpu_time_ms": 20.0},
    )
    trace.record_idle_period(start_ms=40.0, end_ms=50.0)

    engine = MetricsEngine()
    # Compute with only the trace
    report = engine.compute(trace=trace, total_simulation_time_ms=50.0)

    assert report.total_jobs_completed == 2
    assert report.preemption_count == 1
    assert report.scheduling_overhead_ms == 2.0  # 1.0 + 1.0
    assert report.per_tenant_gpu_service_ms["tenant_X"] == 20.0
    assert report.per_tenant_gpu_service_ms["tenant_Y"] == 20.0
    assert report.jains_fairness_index == pytest.approx(1.0, abs=1e-5)
    assert report.gpu_utilization_pct == pytest.approx((40.0 / 50.0) * 100.0, abs=1e-5)
    assert report.gpu_idle_pct == pytest.approx((10.0 / 50.0) * 100.0, abs=1e-5)


# =========================================================================
# 5. Machine-Readable JSON and CSV Output Generation
# =========================================================================
def test_metrics_engine_json_and_csv_generation(tmp_path):
    """Engine produces valid machine-readable JSON and CSV format results."""
    job = GPUJob("J1", "tenant_1", "model", arrival_time=0.0, estimated_gpu_time=20.0, remaining_gpu_time=0.0, deadline=25.0)
    job.start_time = 0.0
    job.completion_time = 20.0
    job.actual_gpu_time_ms = 20.0
    job.accumulated_waiting_time = 0.0
    job.status = JobState.COMPLETED

    engine = MetricsEngine()
    report = engine.compute(jobs=[job], total_simulation_time_ms=25.0)

    # 1. JSON formatting
    json_str = report.to_json()
    data = json.loads(json_str)
    assert "summary" in data
    assert "per_tenant_gpu_service_ms" in data
    assert "tenants" in data
    assert "jobs" in data
    assert data["summary"]["gpu_utilization_pct"] == pytest.approx(80.0, abs=1e-3)
    assert data["summary"]["jains_fairness_index"] == 1.0

    # 2. CSV summary formatting
    csv_summary = report.to_csv_summary()
    reader = csv.reader(csv_summary.strip().split("\n"))
    header = next(reader)
    assert header == ["metric", "value", "unit"]
    rows = {row[0]: float(row[1]) for row in reader}
    assert rows["gpu_utilization_pct"] == pytest.approx(80.0, abs=1e-3)
    assert rows["jains_fairness_index"] == 1.0

    # 3. CSV tenant formatting
    csv_tenants = report.to_csv_tenants()
    t_reader = csv.reader(csv_tenants.strip().split("\n"))
    t_header = next(t_reader)
    assert "tenant_id" in t_header
    assert "gpu_service_ms" in t_header

    # 4. CSV job formatting
    csv_jobs = report.to_csv_jobs()
    j_reader = csv.reader(csv_jobs.strip().split("\n"))
    j_header = next(j_reader)
    assert "job_id" in j_header
    assert "actual_gpu_time_ms" in j_header

    # 5. File persistence roundtrip
    json_file = str(tmp_path / "metrics_test.json")
    csv_summary_file = str(tmp_path / "summary_test.csv")
    csv_tenant_file = str(tmp_path / "tenants_test.csv")
    csv_job_file = str(tmp_path / "jobs_test.csv")

    report.save_json(json_file)
    report.save_csv(csv_summary_file, csv_tenant_file, csv_job_file)

    assert os.path.exists(json_file)
    assert os.path.exists(csv_summary_file)
    assert os.path.exists(csv_tenant_file)
    assert os.path.exists(csv_job_file)

    with open(json_file, "r") as f:
        reloaded = json.load(f)
        assert reloaded["summary"]["total_jobs_completed"] == 1


# =========================================================================
# 6. Terminal Summary Table Formatting
# =========================================================================
def test_metrics_engine_summary_table():
    """Engine formats a clean, multi-line ASCII summary report."""
    job = GPUJob("J1", "tenant_A", "model", arrival_time=0.0, estimated_gpu_time=10.0, remaining_gpu_time=0.0)
    job.start_time = 0.0
    job.completion_time = 10.0
    job.actual_gpu_time_ms = 10.0
    job.status = JobState.COMPLETED

    engine = MetricsEngine()
    report = engine.compute(jobs=[job], total_simulation_time_ms=10.0)

    table_text = report.format_summary_table()
    assert "===================== System Metrics Summary =====================" in table_text
    assert "1.  GPU Utilization" in table_text
    assert "2.  GPU Idle Percentage" in table_text
    assert "3.  System Throughput" in table_text
    assert "4.  Average Waiting Time" in table_text
    assert "7.  Jain's Fairness (J)" in table_text
    assert "Tenant 'tenant_A" in table_text
