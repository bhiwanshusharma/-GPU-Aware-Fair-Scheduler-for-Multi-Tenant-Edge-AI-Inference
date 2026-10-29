"""Master visualizer orchestrating the generation of all 10 evaluation plots.

Loads raw benchmark results (JSON and trace artifacts) and renders:
1. GPU execution timeline (Gantt-style timeline)
2. Tenant GPU allocation
3. GPU utilization
4. Jain fairness index
5. Average waiting time
6. Response time
7. Throughput
8. Preemption count
9. Per-tenant GPU service
10. Comparison between scheduling algorithms
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from src.visualization.gantt import (
    plot_gpu_gantt,
    plot_multi_scheduler_gantt,
    parse_trace_to_slices,
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

logger = logging.getLogger(__name__)


class ResultsVisualizer:
    """Master visualization engine rendering all 10 benchmark plots from experiment outputs."""

    def __init__(self, suite_data: Optional[Dict[str, Any]] = None, traces_dir: Optional[Path] = None) -> None:
        self.suite_data = suite_data or {}
        self.traces_dir = traces_dir

    @classmethod
    def from_results_dir(cls, results_dir: Union[str, Path]) -> "ResultsVisualizer":
        """Constructs visualizer by loading benchmark_results.json and traces from directory."""
        dir_path = Path(results_dir)
        json_file = dir_path / "benchmark_results.json"

        if not json_file.exists():
            # Check if dir_path itself was the json file
            if dir_path.is_file() and dir_path.suffix == ".json":
                json_file = dir_path
                dir_path = dir_path.parent
            else:
                raise FileNotFoundError(
                    f"Results file '{json_file}' not found. "
                    "Run benchmarks first via 'python -m simulator.run_experiment --config configs/example.yaml'."
                )

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        traces_path = dir_path / "traces"
        return cls(suite_data=data, traces_dir=traces_path if traces_path.exists() else None)

    def _resolve_trace(self, scenario_id: str, policy_key: str, p_data: Dict[str, Any]) -> str:
        """Retrieves raw execution trace text from JSON or traces directory on disk."""
        if "execution_trace" in p_data and p_data["execution_trace"]:
            return p_data["execution_trace"]

        if self.traces_dir and self.traces_dir.exists():
            trace_file = self.traces_dir / f"{scenario_id}_{policy_key}.trace.txt"
            if trace_file.exists():
                with open(trace_file, "r", encoding="utf-8") as f:
                    return f.read()

        return ""

    def generate_all_plots(
        self,
        output_dir: Union[str, Path],
        scenario_id: Optional[str] = None,
        dpi: int = 300,
    ) -> Dict[str, str]:
        """Generates all 10 verified evaluation plots from experiment outputs.
        
        Args:
            output_dir: Directory where plot images will be saved.
            scenario_id: Optional scenario identifier (e.g. 'scenario_1' or 'scenario_10').
                         If omitted, chooses a representative scenario (e.g. 'scenario_10' or first).
            dpi: Figure resolution in dots per inch (default: 300).
            
        Returns:
            Dictionary mapping plot keys to saved file paths.
        """
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        saved_plots: Dict[str, str] = {}

        scenarios = self.suite_data.get("scenarios", {})
        if not scenarios:
            logger.warning("No scenarios found in experiment results to plot.")
            return saved_plots

        # Pick target scenario for scenario-specific plots
        if scenario_id and scenario_id in scenarios:
            target_sc_id = scenario_id
        elif "scenario_10" in scenarios:
            target_sc_id = "scenario_10"  # Excellent bimodal scenario with preemption and concurrency
        else:
            target_sc_id = list(scenarios.keys())[0]

        target_sc_data = scenarios[target_sc_id]
        sc_name = target_sc_data.get("scenario_name", target_sc_id)
        policies_data = target_sc_data.get("policies", {})

        # Extract traces for all policies in target scenario
        policy_traces = {}
        for p_key, p_dict in policies_data.items():
            tr = self._resolve_trace(target_sc_id, p_key, p_dict)
            if tr:
                policy_traces[p_key] = tr

        # -------------------------------------------------------------
        # Plot 1: GPU Execution Timeline (Gantt-style timeline)
        # -------------------------------------------------------------
        # 1a. Multi-scheduler stacked comparison Gantt
        gantt_multi_file = out_p / "01_gpu_execution_timeline.png"
        plot_multi_scheduler_gantt(
            policy_traces=policy_traces,
            scenario_name=f"{sc_name} ({target_sc_id})",
            output_path=str(gantt_multi_file),
            dpi=dpi,
        )
        saved_plots["01_gpu_execution_timeline"] = str(gantt_multi_file)

        # 1b. High-detail single-policy Gantt (GPU-Aware Fair Scheduler)
        fair_trace = policy_traces.get("gpu_fair", "")
        if fair_trace:
            slices_fair = parse_trace_to_slices(fair_trace)
            gantt_fair_file = out_p / "01b_gpu_fair_detailed_timeline.png"
            plot_gpu_gantt(
                slices=slices_fair,
                title=f"GPU-Aware Fair Scheduler Timeline: {sc_name}",
                output_path=str(gantt_fair_file),
                dpi=dpi,
            )
            saved_plots["01b_gpu_fair_detailed_timeline"] = str(gantt_fair_file)

        # -------------------------------------------------------------
        # Plot 2: Tenant GPU Allocation
        # -------------------------------------------------------------
        alloc_file = out_p / "02_tenant_gpu_allocation.png"
        plot_tenant_gpu_allocation(
            scenario_data=target_sc_data,
            output_path=str(alloc_file),
            dpi=dpi,
        )
        saved_plots["02_tenant_gpu_allocation"] = str(alloc_file)

        # -------------------------------------------------------------
        # Plot 3: GPU Utilization
        # -------------------------------------------------------------
        util_file = out_p / "03_gpu_utilization.png"
        plot_gpu_utilization(
            suite_data=self.suite_data,
            output_path=str(util_file),
            scenario_id=target_sc_id,
            dpi=dpi,
        )
        saved_plots["03_gpu_utilization"] = str(util_file)

        # -------------------------------------------------------------
        # Plot 4: Jain Fairness Index
        # -------------------------------------------------------------
        jain_file = out_p / "04_jain_fairness_index.png"
        plot_jain_fairness(
            suite_data=self.suite_data,
            output_path=str(jain_file),
            scenario_id=target_sc_id,
            dpi=dpi,
        )
        saved_plots["04_jain_fairness_index"] = str(jain_file)

        # -------------------------------------------------------------
        # Plot 5: Average Waiting Time
        # -------------------------------------------------------------
        wait_file = out_p / "05_average_waiting_time.png"
        plot_waiting_time(
            suite_data=self.suite_data,
            output_path=str(wait_file),
            scenario_id=target_sc_id,
            dpi=dpi,
        )
        saved_plots["05_average_waiting_time"] = str(wait_file)

        # -------------------------------------------------------------
        # Plot 6: Response Time
        # -------------------------------------------------------------
        resp_file = out_p / "06_response_time.png"
        plot_response_time(
            suite_data=self.suite_data,
            output_path=str(resp_file),
            scenario_id=target_sc_id,
            dpi=dpi,
        )
        saved_plots["06_response_time"] = str(resp_file)

        # -------------------------------------------------------------
        # Plot 7: Throughput
        # -------------------------------------------------------------
        tput_file = out_p / "07_throughput.png"
        plot_throughput(
            suite_data=self.suite_data,
            output_path=str(tput_file),
            scenario_id=target_sc_id,
            dpi=dpi,
        )
        saved_plots["07_throughput"] = str(tput_file)

        # -------------------------------------------------------------
        # Plot 8: Preemption Count
        # -------------------------------------------------------------
        preempt_file = out_p / "08_preemption_count.png"
        plot_preemption_count(
            suite_data=self.suite_data,
            output_path=str(preempt_file),
            scenario_id=target_sc_id,
            dpi=dpi,
        )
        saved_plots["08_preemption_count"] = str(preempt_file)

        # -------------------------------------------------------------
        # Plot 9: Per-Tenant GPU Service
        # -------------------------------------------------------------
        service_file = out_p / "09_per_tenant_gpu_service.png"
        plot_per_tenant_service(
            scenario_data=target_sc_data,
            output_path=str(service_file),
            dpi=dpi,
        )
        saved_plots["09_per_tenant_gpu_service"] = str(service_file)

        # -------------------------------------------------------------
        # Plot 10: Comparison Between Scheduling Algorithms
        # -------------------------------------------------------------
        comp_file = out_p / "10_algorithm_comparison.png"
        plot_algorithm_comparison(
            scenario_data=target_sc_data,
            output_path=str(comp_file),
            dpi=dpi,
        )
        saved_plots["10_algorithm_comparison"] = str(comp_file)

        # Close figures to avoid memory accumulation
        import matplotlib.pyplot as plt
        plt.close("all")

        logger.info(f"Generated {len(saved_plots)} publication plots in {output_dir}")
        return saved_plots
