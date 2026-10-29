"""CLI entry point for running reproducible benchmarking experiments.

Usage:
    python -m simulator.run_experiment --config configs/example.yaml
    python -m simulator.run_experiment --scenario scenario_1 --policy fifo round_robin pdwfq
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.runner import (
    BenchmarkRunner,
    CANONICAL_POLICIES,
    normalize_policy_name,
)
from src.benchmarks.scenarios import (
    SCENARIO_FACTORIES,
    get_scenario,
    get_all_scenarios,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("simulator.run_experiment")


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Loads benchmark configuration from a YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def build_arg_parser() -> argparse.ArgumentParser:
    """Constructs command line argument parser for benchmark experiments."""
    parser = argparse.ArgumentParser(
        prog="python -m simulator.run_experiment",
        description="Reproducible Multi-Tenant GPU Scheduling Benchmarking Framework",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to benchmark configuration YAML (e.g. configs/example.yaml)",
    )
    parser.add_argument(
        "--scenario", "-s",
        nargs="+",
        default=None,
        help="Specific scenario ID(s) to evaluate (e.g. scenario_1, or 'all')",
    )
    parser.add_argument(
        "--policy", "-p",
        nargs="+",
        default=None,
        help=f"Scheduling policies to compare. Options: {', '.join(CANONICAL_POLICIES)}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Deterministic pseudo-random seed for workload generation (e.g. 42)",
    )
    parser.add_argument(
        "--quantum",
        type=float,
        default=None,
        help="Base GPU time quantum in ms (default: 20.0)",
    )
    parser.add_argument(
        "--overhead",
        type=float,
        default=None,
        help="Context switch overhead in ms (default: 0.5)",
    )
    parser.add_argument(
        "--starvation-threshold",
        type=float,
        default=None,
        help="Starvation latency threshold in ms (default: 100.0)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directory to save raw results and traces (default: results/benchmarks)",
    )
    parser.add_argument(
        "--no-traces",
        action="store_true",
        help="Disable saving individual discrete execution trace text files",
    )
    parser.add_argument(
        "--use-predictor",
        action="store_true",
        help="Enable AI workload predictor for GPU-Aware Fair Scheduler",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/workload_predictor.joblib",
        help="Path to trained AI workload predictor model",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress intermediate scenario comparison tables and print only summary",
    )
    return parser


def run_experiment(args: argparse.Namespace) -> int:
    """Executes the benchmarking experiment according to configuration and CLI flags."""
    # 1. Base defaults
    bench_cfg: Dict[str, Any] = {
        "random_seed": 42,
        "base_quantum_ms": 20.0,
        "min_quantum_ms": 5.0,
        "max_quantum_ms": 100.0,
        "context_switch_overhead_ms": 0.5,
        "aging_rate": 0.15,
        "starvation_threshold_ms": 100.0,
        "output_dir": "results/benchmarks",
        "save_traces": True,
    }
    configured_policies = list(CANONICAL_POLICIES)
    configured_scenarios = list(SCENARIO_FACTORIES.keys())

    # 2. Load from YAML config if supplied or default file exists
    config_file: Optional[Path] = None
    if args.config:
        config_file = Path(args.config)
    elif Path("configs/example.yaml").exists():
        config_file = Path("configs/example.yaml")

    if config_file and config_file.exists():
        logger.info(f"Loading configuration from {config_file}...")
        raw_cfg = load_yaml_config(config_file)
        if "benchmark" in raw_cfg:
            bench_cfg.update(raw_cfg["benchmark"])
        if "policies" in raw_cfg and raw_cfg["policies"]:
            configured_policies = raw_cfg["policies"]
        if "scenarios" in raw_cfg and raw_cfg["scenarios"]:
            configured_scenarios = raw_cfg["scenarios"]

    # 3. CLI arguments override config file values
    if args.seed is not None:
        bench_cfg["random_seed"] = args.seed
    if args.quantum is not None:
        bench_cfg["base_quantum_ms"] = args.quantum
    if args.overhead is not None:
        bench_cfg["context_switch_overhead_ms"] = args.overhead
    if args.starvation_threshold is not None:
        bench_cfg["starvation_threshold_ms"] = args.starvation_threshold
    if args.output_dir is not None:
        bench_cfg["output_dir"] = args.output_dir
    if args.no_traces:
        bench_cfg["save_traces"] = False

    if args.policy is not None:
        configured_policies = args.policy
    if args.scenario is not None:
        configured_scenarios = args.scenario

    # 4. Optional AI Workload Predictor
    predictor = None
    if args.use_predictor:
        try:
            from src.predictor.inference import WorkloadInferenceEngine
            model_p = Path(args.model_path)
            if model_p.exists():
                predictor = WorkloadInferenceEngine.from_model_file(str(model_p))
                logger.info(f"Loaded AI Workload Predictor from {model_p}")
            else:
                logger.warning(
                    f"Model file {model_p} not found; running with heuristic predictor"
                )
        except Exception as e:
            logger.warning(f"Could not load AI Workload Predictor ({e}); proceeding without ML")

    # 5. Initialize BenchmarkRunner
    runner = BenchmarkRunner(
        base_quantum_ms=bench_cfg["base_quantum_ms"],
        min_quantum_ms=bench_cfg.get("min_quantum_ms", 5.0),
        max_quantum_ms=bench_cfg.get("max_quantum_ms", 100.0),
        context_switch_overhead_ms=bench_cfg["context_switch_overhead_ms"],
        aging_rate=bench_cfg.get("aging_rate", 0.15),
        starvation_threshold_ms=bench_cfg["starvation_threshold_ms"],
        random_seed=bench_cfg["random_seed"],
        policies=configured_policies,
        predictor=predictor,
    )

    print("=" * 80)
    print("  GPU-AWARE FAIR SCHEDULER: REPRODUCIBLE BENCHMARK EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"Random Seed           : {bench_cfg['random_seed']}")
    print(f"Base Quantum          : {bench_cfg['base_quantum_ms']} ms")
    print(f"Context Switch Delay  : {bench_cfg['context_switch_overhead_ms']} ms")
    print(f"Starvation Threshold  : {bench_cfg['starvation_threshold_ms']} ms")
    print(f"Evaluated Policies    : {', '.join(runner.policies)}")
    print(f"Output Directory      : {bench_cfg['output_dir']}")
    print("=" * 80)

    # 6. Execute Benchmark Suite
    suite_result = runner.run_all(
        scenario_ids=configured_scenarios,
        random_seed=bench_cfg["random_seed"],
    )

    # 7. Print Result Comparison Tables
    if not args.quiet:
        for sc_id, sc_res in suite_result.scenarios.items():
            print("\n" + sc_res.format_comparison_table())

    # 8. Export Raw Artifacts (JSON, CSVs, Traces)
    saved_files = suite_result.export_results(
        output_dir=bench_cfg["output_dir"],
        save_traces=bench_cfg.get("save_traces", True),
    )

    print("\n" + "=" * 80)
    print("  EXPERIMENT COMPLETE - ARTIFACTS PERSISTED")
    print("=" * 80)
    print(f"  ✓ Raw JSON Results      : {saved_files.get('json')}")
    print(f"  ✓ Summary Metrics CSV   : {saved_files.get('summary_csv')}")
    print(f"  ✓ Tenant Service CSV    : {saved_files.get('tenant_csv')}")
    if "traces_dir" in saved_files:
        print(f"  ✓ Execution Traces      : {saved_files.get('traces_dir')} ({saved_files.get('trace_count')} files)")
    print("=" * 80)

    return 0


def main() -> None:
    """CLI execution entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        sys.exit(run_experiment(args))
    except Exception as e:
        logger.error(f"Experiment execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
