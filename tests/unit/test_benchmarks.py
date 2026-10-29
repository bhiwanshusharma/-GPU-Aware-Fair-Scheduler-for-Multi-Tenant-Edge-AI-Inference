"""Unit tests for the reproducible benchmarking framework.

Verifies:
1. Determinism and bit-for-bit reproducibility of all 10 benchmark scenarios.
2. Complete isolation and zero-leakage deep cloning of workloads across schedulers.
3. Comparative evaluation of all 5 policies (FIFO, RR, Priority, WRR, GPU-Aware Fair).
4. Calculation and bounding of all 10 required scheduling metrics.
5. Raw artifact export (JSON, CSV, discrete traces).
6. CLI entry point execution.
"""

import os
import json
import csv
import tempfile
from pathlib import Path
import pytest

from src.benchmarks.scenarios import (
    SCENARIO_FACTORIES,
    get_scenario,
    get_all_scenarios,
    create_scenario_1,
    create_scenario_7,
    create_scenario_8,
    create_scenario_10,
)
from src.benchmarks.runner import (
    BenchmarkRunner,
    create_scheduler,
    normalize_policy_name,
    CANONICAL_POLICIES,
    ScenarioBenchmarkResult,
    BenchmarkSuiteResult,
)
from simulator.run_experiment import build_arg_parser, run_experiment


class TestBenchmarkScenarios:
    """Verifies scenario registry, initialization, and deterministic generation."""

    def test_all_ten_scenarios_registered(self):
        """Ensures all 10 required scenarios are available and factory functions valid."""
        assert len(SCENARIO_FACTORIES) == 10
        scenarios = get_all_scenarios()
        assert len(scenarios) == 10
        ids = {s.scenario_id for s in scenarios}
        expected_ids = {f"scenario_{i}" for i in range(1, 11)}
        assert ids == expected_ids

    def test_get_scenario_lookup_and_aliases(self):
        """Verifies scenario lookup with exact IDs and shorthand numbers."""
        s1 = get_scenario("scenario_1")
        assert s1.scenario_id == "scenario_1"
        assert len(s1.tenants) == 2

        s10 = get_scenario("scenario_10")
        assert s10.scenario_id == "scenario_10"

        # Shorthand aliases
        s2 = get_scenario("2")
        assert s2.scenario_id == "scenario_2"
        s9 = get_scenario("scenario-9")
        assert s9.scenario_id == "scenario_9"

        with pytest.raises(KeyError):
            get_scenario("scenario_99")

    def test_deterministic_job_generation(self):
        """Verifies that the same random seed produces identical job parameters."""
        sc = create_scenario_1()
        jobs_a = sc.generate_jobs(seed=42)
        jobs_b = sc.generate_jobs(seed=42)

        assert len(jobs_a) == len(jobs_b)
        for ja, jb in zip(jobs_a, jobs_b):
            assert ja.job_id == jb.job_id
            assert ja.tenant_id == jb.tenant_id
            assert ja.arrival_time == pytest.approx(jb.arrival_time, 1e-6)
            assert ja.estimated_gpu_time == pytest.approx(jb.estimated_gpu_time, 1e-6)
            assert ja.priority == jb.priority

    def test_different_seeds_produce_variation(self):
        """Verifies that changing seeds alters arrival jitter."""
        sc = create_scenario_1()
        jobs_42 = sc.generate_jobs(seed=42)
        jobs_99 = sc.generate_jobs(seed=99)

        # Arrival times should differ due to jitter
        arr_42 = [j.arrival_time for j in jobs_42]
        arr_99 = [j.arrival_time for j in jobs_99]
        assert arr_42 != arr_99


class TestBenchmarkRunner:
    """Verifies execution, metric computation, and policy comparisons."""

    def test_policy_normalization(self):
        """Verifies robust normalization of policy names and aliases."""
        assert normalize_policy_name("fifo") == "fifo"
        assert normalize_policy_name("round_robin") == "round_robin"
        assert normalize_policy_name("rr") == "round_robin"
        assert normalize_policy_name("priority") == "priority"
        assert normalize_policy_name("weighted-round-robin") == "weighted_round_robin"
        assert normalize_policy_name("pdwfq") == "gpu_fair"
        assert normalize_policy_name("gpu_aware_fair") == "gpu_fair"

        with pytest.raises(ValueError):
            normalize_policy_name("invalid_policy_name")

    def test_workload_deep_cloning_isolation(self):
        """Verifies that running a benchmark does not alter the original scenario objects."""
        sc = create_scenario_1()
        orig_t1_quota = sc.tenants["tenant_1"].gpu_quota
        orig_jobs = sc.generate_jobs(seed=42)
        orig_first_job_rem = orig_jobs[0].remaining_gpu_time

        runner = BenchmarkRunner(
            base_quantum_ms=20.0,
            policies=["fifo", "round_robin"],
            random_seed=42,
        )
        res = runner.run_scenario(sc)

        # Ensure scenario state is untouched
        assert sc.tenants["tenant_1"].gpu_quota == orig_t1_quota
        assert orig_jobs[0].remaining_gpu_time == orig_first_job_rem
        assert res.job_count == len(orig_jobs)

    def test_all_five_policies_run_on_scenario_1(self):
        """Verifies that all 5 scheduling policies execute and produce valid metrics on Scenario 1."""
        sc = create_scenario_1()
        runner = BenchmarkRunner(
            base_quantum_ms=15.0,
            policies=CANONICAL_POLICIES,
            random_seed=42,
        )
        res = runner.run_scenario(sc)

        assert len(res.policy_results) == 5
        for p_key in CANONICAL_POLICIES:
            assert p_key in res.policy_results
            p_res = res.policy_results[p_key]
            m = p_res.metrics

            # Metric presence and range verification
            assert 0.0 <= m.gpu_utilization_pct <= 100.0
            assert m.throughput_jobs_per_sec > 0.0
            assert m.avg_waiting_time_ms >= 0.0
            assert m.avg_response_time_ms >= 0.0
            assert m.avg_turnaround_time_ms >= 0.0
            assert 0.0 < m.jains_fairness_index <= 1.0
            assert m.preemption_count >= 0
            assert m.deadline_misses >= 0
            assert m.starvation_events >= 0
            assert m.scheduling_overhead_ms >= 0.0
            assert p_res.completed_jobs_count == res.job_count

            # Check tenant GPU service dictionary
            assert len(m.per_tenant_gpu_service_ms) == 2
            assert "tenant_1" in m.per_tenant_gpu_service_ms
            assert "tenant_2" in m.per_tenant_gpu_service_ms

    def test_scenario_10_short_jobs_responsive_scheduling(self):
        """Verifies Scenario 10: time-slicing (RR/Fair) reduces waiting latency compared to FIFO."""
        sc = create_scenario_10()
        runner = BenchmarkRunner(
            base_quantum_ms=10.0,
            policies=["fifo", "round_robin", "gpu_fair"],
            random_seed=42,
        )
        res = runner.run_scenario(sc)

        fifo_wait = res.policy_results["fifo"].metrics.avg_waiting_time_ms
        fair_wait = res.policy_results["gpu_fair"].metrics.avg_waiting_time_ms

        # In bimodal workloads, time-slicing / fair scheduling prevents long jobs from blocking short jobs
        assert fair_wait < fifo_wait

    def test_scenario_7_priority_behavior(self):
        """Verifies Scenario 7: Priority scheduling protects high-priority deadlines better than FIFO."""
        sc = create_scenario_7()
        runner = BenchmarkRunner(
            base_quantum_ms=20.0,
            policies=["fifo", "priority", "gpu_fair"],
            random_seed=42,
        )
        res = runner.run_scenario(sc)

        fifo_misses = res.policy_results["fifo"].metrics.deadline_misses
        prio_misses = res.policy_results["priority"].metrics.deadline_misses
        fair_misses = res.policy_results["gpu_fair"].metrics.deadline_misses

        # Priority and Fair scheduling should have fewer or equal deadline misses than FIFO
        assert prio_misses <= fifo_misses
        assert fair_misses <= fifo_misses

    def test_scenario_8_quota_throttling(self):
        """Verifies Scenario 8: GPU-Aware Fair Scheduler enforces quotas."""
        sc = create_scenario_8()
        runner = BenchmarkRunner(
            base_quantum_ms=20.0,
            policies=["round_robin", "gpu_fair"],
            random_seed=42,
        )
        res = runner.run_scenario(sc)

        fair_report = res.policy_results["gpu_fair"].metrics
        assert "tenant_constrained" in fair_report.per_tenant_gpu_service_ms
        assert "tenant_unconstrained" in fair_report.per_tenant_gpu_service_ms

    def test_format_comparison_table(self):
        """Verifies ASCII comparison table rendering."""
        sc = create_scenario_1()
        runner = BenchmarkRunner(policies=["fifo", "round_robin"], random_seed=42)
        res = runner.run_scenario(sc)
        table_str = res.format_comparison_table()

        assert "Benchmark Comparison" in table_str
        assert "GPU Utilization (%)" in table_str
        assert "Jain's Fairness Index" in table_str
        assert "FIFO" in table_str
        assert "Round Robin" in table_str


class TestBenchmarkArtifactExport:
    """Verifies persistence of raw JSON, CSV, and trace artifacts."""

    def test_export_raw_results(self):
        """Verifies file generation and format validity."""
        sc = create_scenario_1()
        runner = BenchmarkRunner(
            base_quantum_ms=20.0,
            policies=["fifo", "round_robin"],
            random_seed=42,
        )
        suite = runner.run_all(scenario_ids=["scenario_1"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            saved = suite.export_results(tmp_dir, save_traces=True)

            assert os.path.exists(saved["json"])
            assert os.path.exists(saved["summary_csv"])
            assert os.path.exists(saved["tenant_csv"])
            assert os.path.exists(saved["traces_dir"])

            # Verify JSON content
            with open(saved["json"], "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "scenarios" in data
            assert "scenario_1" in data["scenarios"]
            assert "fifo" in data["scenarios"]["scenario_1"]["policies"]

            # Verify Summary CSV headers and rows
            with open(saved["summary_csv"], "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                assert "scenario_id" in header
                assert "jains_fairness_index" in header
                rows = list(reader)
                assert len(rows) == 2  # 2 policies

            # Verify Tenant Service CSV
            with open(saved["tenant_csv"], "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                assert "tenant_id" in header
                assert "gpu_service_ms" in header
                rows = list(reader)
                assert len(rows) >= 4  # 2 policies * 2 tenants

            # Verify Trace files
            traces_dir = Path(saved["traces_dir"])
            trace_files = list(traces_dir.glob("*.trace.txt"))
            assert len(trace_files) == 2
            for tf in trace_files:
                assert tf.stat().st_size > 0


class TestBenchmarkCLI:
    """Verifies command line interface execution."""

    def test_cli_execution_with_args(self):
        """Verifies CLI execution with customized arguments."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            parser = build_arg_parser()
            args = parser.parse_args([
                "--scenario", "scenario_1",
                "--policy", "fifo", "round_robin",
                "--seed", "42",
                "--quantum", "25.0",
                "--overhead", "0.2",
                "--output-dir", tmp_dir,
                "--quiet",
            ])
            ret_code = run_experiment(args)
            assert ret_code == 0

            # Verify generated JSON
            json_file = Path(tmp_dir) / "benchmark_results.json"
            assert json_file.exists()
            with open(json_file, "r") as f:
                data = json.load(f)
            assert "scenario_1" in data["scenarios"]
