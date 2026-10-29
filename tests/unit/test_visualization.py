"""Unit tests for the visualization framework and plot generation.

Verifies:
1. Accurate parsing of discrete execution traces into ExecutionSlice records.
2. Rendering and file output for single and multi-scheduler Gantt charts.
3. Generation of all 10 individual and comparative performance plots.
4. Robustness against zero-value or edge-case telemetry structures.
5. CLI plotting command execution.
"""

import os
import json
import tempfile
from pathlib import Path
import pytest
import matplotlib.pyplot as plt

from src.visualization.gantt import (
    ExecutionSlice,
    parse_trace_to_slices,
    plot_gpu_gantt,
    plot_multi_scheduler_gantt,
)
from src.visualization.metrics_plots import (
    plot_tenant_gpu_allocation,
    plot_gpu_utilization,
    plot_jain_fairness,
    plot_waiting_time,
    plot_response_time,
    plot_throughput,
    plot_preemption_count,
    plot_per_tenant_service,
    plot_algorithm_comparison,
)
from src.visualization.plotter import ResultsVisualizer
from simulator.plot_results import build_arg_parser, run_plotting


class TestTraceParsing:
    """Verifies parsing of raw discrete event traces into execution intervals."""

    def test_parse_simple_trace_with_preemption_and_completion(self):
        """Tests parsing dispatches, quantum expirations, resumes, and completions."""
        trace_text = """
time=0.0
tenant=tenant_A job=A1 dispatched
time=10.0
A1 quantum expired
tenant=tenant_B job=B1 dispatched
time=15.0
tenant=tenant_B job=B1 completed
tenant=tenant_A job=A1 resumed
time=25.0
tenant=tenant_A job=A1 completed
"""
        slices = parse_trace_to_slices(trace_text)
        assert len(slices) == 3

        # Slice 1: A1 dispatched, preempted at 10.0
        s1 = slices[0]
        assert s1.tenant_id == "tenant_A"
        assert s1.job_id == "A1"
        assert s1.start_time_ms == 0.0
        assert s1.end_time_ms == 10.0
        assert s1.duration_ms == 10.0
        assert s1.event_type == "preemption"

        # Slice 2: B1 dispatched, completed at 15.0
        s2 = slices[1]
        assert s2.tenant_id == "tenant_B"
        assert s2.job_id == "B1"
        assert s2.start_time_ms == 10.0
        assert s2.end_time_ms == 15.0
        assert s2.duration_ms == 5.0
        assert s2.event_type == "completion"

        # Slice 3: A1 resumed, completed at 25.0
        s3 = slices[2]
        assert s3.tenant_id == "tenant_A"
        assert s3.job_id == "A1"
        assert s3.start_time_ms == 15.0
        assert s3.end_time_ms == 25.0
        assert s3.duration_ms == 10.0
        assert s3.event_type == "completion"

    def test_parse_empty_trace(self):
        """Ensures empty trace string yields empty slice list."""
        assert parse_trace_to_slices("") == []
        assert parse_trace_to_slices("   \n\n  ") == []


class TestGanttPlotting:
    """Verifies Gantt timeline plot generation."""

    def test_plot_gpu_gantt_single_policy(self):
        """Verifies rendering and file persistence of single-policy Gantt."""
        slices = [
            ExecutionSlice("tenant_A", "A1", 0.0, 10.0, 10.0, "preemption"),
            ExecutionSlice("tenant_B", "B1", 10.0, 20.0, 10.0, "completion"),
            ExecutionSlice("tenant_A", "A1", 20.0, 30.0, 10.0, "completion"),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = os.path.join(tmp_dir, "test_gantt.png")
            fig = plot_gpu_gantt(slices, title="Test Gantt", output_path=out_file, dpi=100)
            assert isinstance(fig, plt.Figure)
            assert os.path.exists(out_file)
            assert os.path.getsize(out_file) > 1000
            plt.close(fig)

    def test_plot_multi_scheduler_gantt(self):
        """Verifies multi-panel stacked Gantt comparison plot."""
        policy_traces = {
            "fifo": "time=0\ntenant=tenant_A job=A1 dispatched\ntime=20\ntenant=tenant_A job=A1 completed",
            "round_robin": "time=0\ntenant=tenant_A job=A1 dispatched\ntime=10\nA1 quantum expired\ntenant=tenant_B job=B1 dispatched\ntime=15\ntenant=tenant_B job=B1 completed\ntenant=tenant_A job=A1 resumed\ntime=25\ntenant=tenant_A job=A1 completed",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = os.path.join(tmp_dir, "test_multi_gantt.png")
            fig = plot_multi_scheduler_gantt(policy_traces, scenario_name="Test Scenario", output_path=out_file, dpi=100)
            assert isinstance(fig, plt.Figure)
            assert os.path.exists(out_file)
            assert os.path.getsize(out_file) > 1000
            plt.close(fig)


class TestMetricsPlots:
    """Verifies individual and comparative performance plots."""

    @pytest.fixture
    def mock_scenario_data(self):
        """Provides representative multi-policy evaluation result structure."""
        return {
            "scenario_id": "scenario_test",
            "scenario_name": "Test Workload Scenario",
            "tenants": {
                "tenant_1": {"weight": 1.0, "priority": 5},
                "tenant_2": {"weight": 2.0, "priority": 10},
            },
            "policies": {
                "fifo": {
                    "policy_name": "FIFO",
                    "metrics": {
                        "summary": {
                            "gpu_utilization_pct": 92.5,
                            "throughput_jobs_per_sec": 45.0,
                            "avg_waiting_time_ms": 35.2,
                            "avg_response_time_ms": 35.2,
                            "avg_turnaround_time_ms": 55.2,
                            "jains_fairness_index": 0.82,
                            "preemption_count": 0,
                            "scheduling_overhead_ms": 0.0,
                        },
                        "per_tenant_gpu_service_ms": {"tenant_1": 100.0, "tenant_2": 100.0},
                    },
                },
                "gpu_fair": {
                    "policy_name": "GPU-Aware Fair",
                    "metrics": {
                        "summary": {
                            "gpu_utilization_pct": 95.0,
                            "throughput_jobs_per_sec": 48.0,
                            "avg_waiting_time_ms": 14.8,
                            "avg_response_time_ms": 10.2,
                            "avg_turnaround_time_ms": 32.5,
                            "jains_fairness_index": 0.985,
                            "preemption_count": 4,
                            "scheduling_overhead_ms": 2.0,
                        },
                        "per_tenant_gpu_service_ms": {"tenant_1": 70.0, "tenant_2": 130.0},
                    },
                },
            },
        }

    def test_plot_tenant_gpu_allocation(self, mock_scenario_data):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_p = os.path.join(tmp_dir, "02_alloc.png")
            fig = plot_tenant_gpu_allocation(mock_scenario_data, output_path=out_p, dpi=100)
            assert os.path.exists(out_p)
            plt.close(fig)

    def test_plot_gpu_utilization(self, mock_scenario_data):
        suite = {"scenarios": {"scenario_test": mock_scenario_data}}
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_p = os.path.join(tmp_dir, "03_util.png")
            fig = plot_gpu_utilization(suite, output_path=out_p, scenario_id="scenario_test", dpi=100)
            assert os.path.exists(out_p)
            plt.close(fig)

    def test_plot_jain_fairness(self, mock_scenario_data):
        suite = {"scenarios": {"scenario_test": mock_scenario_data}}
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_p = os.path.join(tmp_dir, "04_jain.png")
            fig = plot_jain_fairness(suite, output_path=out_p, scenario_id="scenario_test", dpi=100)
            assert os.path.exists(out_p)
            plt.close(fig)

    def test_plot_waiting_time(self, mock_scenario_data):
        suite = {"scenarios": {"scenario_test": mock_scenario_data}}
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_p = os.path.join(tmp_dir, "05_wait.png")
            fig = plot_waiting_time(suite, output_path=out_p, scenario_id="scenario_test", dpi=100)
            assert os.path.exists(out_p)
            plt.close(fig)

    def test_plot_response_time(self, mock_scenario_data):
        suite = {"scenarios": {"scenario_test": mock_scenario_data}}
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_p = os.path.join(tmp_dir, "06_resp.png")
            fig = plot_response_time(suite, output_path=out_p, scenario_id="scenario_test", dpi=100)
            assert os.path.exists(out_p)
            plt.close(fig)

    def test_plot_throughput(self, mock_scenario_data):
        suite = {"scenarios": {"scenario_test": mock_scenario_data}}
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_p = os.path.join(tmp_dir, "07_tput.png")
            fig = plot_throughput(suite, output_path=out_p, scenario_id="scenario_test", dpi=100)
            assert os.path.exists(out_p)
            plt.close(fig)

    def test_plot_preemption_count(self, mock_scenario_data):
        suite = {"scenarios": {"scenario_test": mock_scenario_data}}
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_p = os.path.join(tmp_dir, "08_preempt.png")
            fig = plot_preemption_count(suite, output_path=out_p, scenario_id="scenario_test", dpi=100)
            assert os.path.exists(out_p)
            plt.close(fig)

    def test_plot_per_tenant_service(self, mock_scenario_data):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_p = os.path.join(tmp_dir, "09_service.png")
            fig = plot_per_tenant_service(mock_scenario_data, output_path=out_p, dpi=100)
            assert os.path.exists(out_p)
            plt.close(fig)

    def test_plot_algorithm_comparison(self, mock_scenario_data):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_p = os.path.join(tmp_dir, "10_comparison.png")
            fig = plot_algorithm_comparison(mock_scenario_data, output_path=out_p, dpi=100)
            assert os.path.exists(out_p)
            plt.close(fig)


class TestVisualizerIntegration:
    """Verifies end-to-end plot generation from benchmark results on disk."""

    def test_generate_all_plots_from_results_dir(self):
        """Ensures all 10 plot files are generated from actual benchmark results."""
        if not os.path.exists("results/benchmarks/benchmark_results.json"):
            pytest.skip("benchmark_results.json not present on disk")

        with tempfile.TemporaryDirectory() as tmp_dir:
            vis = ResultsVisualizer.from_results_dir("results/benchmarks")
            saved = vis.generate_all_plots(output_dir=tmp_dir, scenario_id="scenario_10", dpi=100)

            # Check that all 10 required plot categories were generated
            assert "01_gpu_execution_timeline" in saved
            assert "02_tenant_gpu_allocation" in saved
            assert "03_gpu_utilization" in saved
            assert "04_jain_fairness_index" in saved
            assert "05_average_waiting_time" in saved
            assert "06_response_time" in saved
            assert "07_throughput" in saved
            assert "08_preemption_count" in saved
            assert "09_per_tenant_gpu_service" in saved
            assert "10_algorithm_comparison" in saved

            # Verify files exist and are non-empty
            for key, path in saved.items():
                assert os.path.exists(path)
                assert os.path.getsize(path) > 1000

    def test_cli_plotting_command(self):
        """Verifies CLI plotting command."""
        if not os.path.exists("results/benchmarks/benchmark_results.json"):
            pytest.skip("benchmark_results.json not present on disk")

        with tempfile.TemporaryDirectory() as tmp_dir:
            parser = build_arg_parser()
            args = parser.parse_args([
                "--results-dir", "results/benchmarks",
                "--output-dir", tmp_dir,
                "--scenario", "scenario_1",
                "--dpi", "100",
            ])
            ret = run_plotting(args)
            assert ret == 0
            assert os.path.exists(os.path.join(tmp_dir, "01_gpu_execution_timeline.png"))
