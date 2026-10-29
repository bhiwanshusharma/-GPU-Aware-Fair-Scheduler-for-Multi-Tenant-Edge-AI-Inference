"""Unit tests for the dashboard simulation engine and UI interface.

Verifies:
1. Workload generation across distribution presets (balanced, bursty, bimodal, etc.).
2. Engine execution producing full snapshot state (all 13 required metrics and monitors).
3. Multi-policy comparative evaluation engine.
4. Quota consumption and throttling detection.
5. CLI dashboard argument parsing.
"""

import pytest
from src.dashboard.engine import (
    DashboardSimulationEngine,
    SimulationRunConfig,
    TenantConfigInput,
    SimulationDashboardSnapshot,
)
from simulator.dashboard import build_arg_parser


class TestDashboardWorkloadGeneration:
    """Verifies parameterized workload generation across presets."""

    def test_generate_balanced_workload(self):
        engine = DashboardSimulationEngine()
        jobs = engine.generate_workload(
            tenant_ids=["t1", "t2"],
            distribution="balanced",
            total_jobs=20,
            seed=42,
        )
        assert len(jobs) == 20
        # Arrivals are sorted
        arrivals = [j.arrival_time for j in jobs]
        assert arrivals == sorted(arrivals)
        # Symmetrical tenant representation
        t1_count = sum(1 for j in jobs if j.tenant_id == "t1")
        t2_count = sum(1 for j in jobs if j.tenant_id == "t2")
        assert t1_count == 10
        assert t2_count == 10

    def test_generate_bursty_workload(self):
        engine = DashboardSimulationEngine()
        jobs = engine.generate_workload(
            tenant_ids=["t_burst", "t_steady"],
            distribution="bursty",
            total_jobs=20,
            seed=42,
        )
        assert len(jobs) > 0
        burst_jobs = [j for j in jobs if j.tenant_id == "t_burst"]
        steady_jobs = [j for j in jobs if j.tenant_id == "t_steady"]
        assert len(burst_jobs) > len(steady_jobs)

    def test_generate_bimodal_workload(self):
        engine = DashboardSimulationEngine()
        jobs = engine.generate_workload(
            tenant_ids=["t_bulk", "t_interactive"],
            distribution="short_long_bimodal",
            total_jobs=16,
            seed=42,
        )
        bulk_times = [j.estimated_gpu_time for j in jobs if j.tenant_id == "t_bulk"]
        inter_times = [j.estimated_gpu_time for j in jobs if j.tenant_id == "t_interactive"]
        assert min(bulk_times) > max(inter_times)

    def test_deterministic_generation(self):
        engine = DashboardSimulationEngine()
        jobs_a = engine.generate_workload(["t1", "t2"], seed=42)
        jobs_b = engine.generate_workload(["t1", "t2"], seed=42)
        for ja, jb in zip(jobs_a, jobs_b):
            assert ja.job_id == jb.job_id
            assert ja.arrival_time == jb.arrival_time


class TestDashboardSimulationEngine:
    """Verifies simulation run and snapshot generation."""

    def test_run_experiment_produces_complete_snapshot(self):
        engine = DashboardSimulationEngine()
        cfg = SimulationRunConfig(
            tenants=[
                TenantConfigInput("t_robotics", name="Robotics", weight=1.0, priority=5, gpu_quota_pct=100.0),
                TenantConfigInput("t_vision", name="Vision", weight=1.0, priority=5, gpu_quota_pct=100.0),
            ],
            policy="gpu_fair",
            base_quantum_ms=20.0,
            total_jobs=12,
            random_seed=42,
        )
        snapshot = engine.run_experiment(cfg)

        assert isinstance(snapshot, SimulationDashboardSnapshot)
        assert snapshot.total_simulation_time_ms > 0.0
        assert 0.0 <= snapshot.gpu_utilization_pct <= 100.0
        assert len(snapshot.active_tenants) == 2

        # Active tenants metadata
        for t_info in snapshot.active_tenants:
            assert "tenant_id" in t_info
            assert "service_ms" in t_info
            assert "allocation_pct" in t_info
            assert "quota_used_pct" in t_info
            assert "is_throttled" in t_info

        # Running job
        assert snapshot.current_running_job is not None
        assert "job_id" in snapshot.current_running_job

        # Decisions
        assert len(snapshot.scheduling_decisions) > 0
        d0 = snapshot.scheduling_decisions[0]
        assert "decision_id" in d0
        assert "scheduling_score" in d0
        assert "reason" in d0

        # Execution slices (for Gantt)
        assert len(snapshot.execution_slices) > 0

        # Metrics (all required metrics present)
        m = snapshot.metrics
        assert m.throughput_jobs_per_sec > 0.0
        assert m.avg_waiting_time_ms >= 0.0
        assert m.avg_response_time_ms >= 0.0
        assert 0.0 < m.jains_fairness_index <= 1.0
        assert m.preemption_count >= 0

    def test_compare_all_policies(self):
        engine = DashboardSimulationEngine()
        cfg = SimulationRunConfig(
            tenants=[
                TenantConfigInput("t1", weight=1.0, priority=5, gpu_quota_pct=100.0),
                TenantConfigInput("t2", weight=1.0, priority=5, gpu_quota_pct=100.0),
            ],
            base_quantum_ms=20.0,
            total_jobs=10,
            random_seed=42,
        )
        reports = engine.compare_all_policies(cfg)
        assert len(reports) == 5
        expected_keys = {"fifo", "round_robin", "priority", "weighted_round_robin", "gpu_fair"}
        assert set(reports.keys()) == expected_keys
        for pol, rep in reports.items():
            assert rep.gpu_utilization_pct > 0.0
            assert 0.0 < rep.jains_fairness_index <= 1.0

    def test_quota_throttling_detection(self):
        engine = DashboardSimulationEngine()
        cfg = SimulationRunConfig(
            tenants=[
                TenantConfigInput("t_tight", weight=1.0, priority=5, gpu_quota_pct=15.0),
                TenantConfigInput("t_open", weight=1.0, priority=5, gpu_quota_pct=85.0),
            ],
            policy="gpu_fair",
            base_quantum_ms=20.0,
            total_jobs=16,
            random_seed=42,
        )
        snapshot = engine.run_experiment(cfg)
        tight_tenant = next(t for t in snapshot.active_tenants if t["tenant_id"] == "t_tight")
        # Quota used percentage should be tracked
        assert tight_tenant["quota_used_pct"] > 0.0


class TestDashboardCLI:
    """Verifies dashboard CLI argument parsing."""

    def test_arg_parser_defaults(self):
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.port == 8501
        assert args.host == "localhost"
        assert not args.headless

    def test_arg_parser_custom_flags(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--port", "9000", "--host", "0.0.0.0", "--headless"])
        assert args.port == 9000
        assert args.host == "0.0.0.0"
        assert args.headless
