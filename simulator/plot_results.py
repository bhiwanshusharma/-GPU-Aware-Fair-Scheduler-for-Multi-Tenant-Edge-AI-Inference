"""CLI command to generate all evaluation plots from experiment results.

Usage:
    python -m simulator.plot_results --results-dir results/benchmarks --output-dir results/plots
    python -m simulator.plot_results --scenario scenario_10 --dpi 300
"""

import sys
import argparse
import logging
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.plotter import ResultsVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("simulator.plot_results")


def build_arg_parser() -> argparse.ArgumentParser:
    """Constructs command line argument parser for visualization generation."""
    parser = argparse.ArgumentParser(
        prog="python -m simulator.plot_results",
        description="Generate All 10 Scheduler Behavior and Performance Evaluation Plots",
    )
    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default="results/benchmarks",
        help="Directory containing benchmark_results.json and execution traces (default: results/benchmarks)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results/plots",
        help="Directory to save generated plot figures (default: results/plots)",
    )
    parser.add_argument(
        "--scenario", "-s",
        type=str,
        default=None,
        help="Specific scenario ID to visualize (e.g. scenario_10, scenario_1). Defaults to scenario_10.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure resolution in DPI (default: 300)",
    )
    return parser


def run_plotting(args: argparse.Namespace) -> int:
    """Loads experiment results and generates all 10 verified plots."""
    results_path = Path(args.results_dir)
    output_path = Path(args.output_dir)

    print("=" * 80)
    print("  GPU-AWARE FAIR SCHEDULER: EVALUATION PLOT GENERATOR")
    print("=" * 80)
    print(f"Results Directory : {results_path.resolve()}")
    print(f"Output Directory  : {output_path.resolve()}")
    if args.scenario:
        print(f"Target Scenario   : {args.scenario}")
    print(f"Figure Resolution : {args.dpi} DPI")
    print("=" * 80)

    try:
        visualizer = ResultsVisualizer.from_results_dir(results_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    saved = visualizer.generate_all_plots(
        output_dir=output_path,
        scenario_id=args.scenario,
        dpi=args.dpi,
    )

    print("\n" + "=" * 80)
    print("  SUCCESSFULLY GENERATED 10 EVALUATION PLOTS")
    print("=" * 80)
    for plot_name, file_path in sorted(saved.items()):
        print(f"  ✓ {plot_name:<34} : {file_path}")
    print("=" * 80)
    return 0


def main() -> None:
    """CLI execution entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()
    sys.exit(run_plotting(args))


if __name__ == "__main__":
    main()
