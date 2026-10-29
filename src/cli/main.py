"""Command Line Interface (CLI) entry point for the GPU-Aware Fair Scheduler."""

import argparse
import sys
import os
import logging
from pathlib import Path

from src import __version__
from src.core.config import load_tenants_config, load_workloads_config, load_simulation_config
from src.core.types import PolicyType
from src.core.queues import TenantQueueManager
from src.core.job import GPUJob, Job

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gpu_sched")


def cmd_info(args: argparse.Namespace) -> int:
    """Displays system status, supported policies, and runtime information."""
    print("=" * 65)
    print(f"  GPU-Aware Fair Scheduler for Multi-Tenant Edge AI (v{__version__})")
    print("=" * 65)
    print(f"Target Platforms : NVIDIA Jetson (UMA) & Cross-Platform DES")
    print(f"Scheduling Level : Application / Job-Dispatch Arbiter")
    print(f"Supported Policies:")
    for p in PolicyType:
        print(f"  - {p.value.upper():<16} : {p.name}")
    print(f"Python Runtime   : {sys.version.split()[0]}")
    print("=" * 65)
    return 0


def cmd_validate_config(args: argparse.Namespace) -> int:
    """Validates configuration YAML files."""
    config_dir = Path(args.config_dir)
    print(f"[Config Validation] Checking directory: {config_dir.resolve()}")

    tenants_path = config_dir / "tenants.yaml"
    workloads_path = config_dir / "workloads.yaml"
    sim_path = config_dir / "default_simulation.yaml"

    try:
        tenants = load_tenants_config(tenants_path)
        print(f"  ✓ Successfully loaded {len(tenants)} tenants from {tenants_path.name}")
        for t in tenants:
            print(f"    - Tenant '{t.tenant_id}': weight={t.weight}, priority={t.priority}, quota={t.gpu_quota_pct}%")

        workloads = load_workloads_config(workloads_path)
        print(f"  ✓ Successfully loaded {len(workloads)} AI workload profiles from {workloads_path.name}")
        for w in workloads.values():
            print(f"    - Model '{w.model_name}': category={w.category}, latency={w.calibrated_base_latency_ms}ms")

        sim_cfg = load_simulation_config(sim_path)
        print(f"  ✓ Successfully loaded simulation config from {sim_path.name}")
        print(f"    - Policy: {sim_cfg.policy.value}, Duration: {sim_cfg.duration_ms}ms, Seed: {sim_cfg.random_seed}")

        print("\nAll configurations validated successfully with zero errors.")
        return 0
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return 1


def cmd_inspect_tenants(args: argparse.Namespace) -> int:
    """Prints a tabular summary of registered tenants."""
    tenants_path = Path(args.config)
    try:
        tenants = load_tenants_config(tenants_path)
        print(f"\n{'TENANT ID':<22} | {'WEIGHT':<6} | {'PRIO':<4} | {'QUOTA %':<8} | {'SLA (ms)':<8} | {'MAX QUEUE':<9}")
        print("-" * 75)
        for t in tenants:
            print(f"{t.tenant_id:<22} | {t.weight:<6.1f} | {t.priority:<4} | {t.gpu_quota_pct:<8.1f} | {t.sla_latency_target_ms:<8.1f} | {t.max_queue_capacity:<9}")
        return 0
    except Exception as e:
        logger.error(f"Failed to inspect tenants: {e}")
        return 1


def cmd_smoke_test(args: argparse.Namespace) -> int:
    """Executes a smoke test of the core queuing and lifecycle mechanics."""
    print("[Smoke Test] Testing multi-tenant isolated queues and job states...")
    qm = TenantQueueManager()
    
    # Load default tenants
    tenants = load_tenants_config("configs/tenants.yaml")
    for t in tenants:
        qm.register_tenant(t)

    # Submit jobs
    t0 = tenants[0]
    for i in range(5):
        job = Job(
            job_id=f"smoke_job_{i}",
            tenant_id=t0.tenant_id,
            model_name="yolov8n_detection",
            input_shape=(3, 640, 640),
            batch_size=1,
            arrival_time_ms=10.0 * i,
        )
        qm.enqueue(job)

    assert qm.total_queued_jobs == 5, "Queue job count mismatch"
    popped = qm.dequeue(t0.tenant_id)
    assert popped is not None and popped.job_id == "smoke_job_0", "Dequeue head mismatch"
    popped.mark_started(50.0)
    popped.mark_completed(70.0, actual_gpu_time_ms=20.0)
    
    report = popped.to_completion_report()
    assert report.waiting_time_ms == 50.0, "Waiting time mismatch"
    assert report.actual_gpu_time_ms == 20.0, "Execution time mismatch"
    
    print(f"  ✓ Smoke test passed: 5 jobs enqueued, 1 completed (Wait: {report.waiting_time_ms}ms, Exec: {report.actual_gpu_time_ms}ms)")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    """Runs a multi-tenant event-driven discrete simulation and displays the execution trace and metrics."""
    from src.scheduler import (
        FIFOScheduler,
        RoundRobinScheduler,
        PriorityScheduler,
        WeightedRoundRobinScheduler,
        GPUFairScheduler,
        run_scheduler_benchmark,
    )
    from src.core.types import SchedulingQuantum

    print(f"[GPU Simulator] Running scheduler simulation (Quantum={args.quantum}ms, Overhead={args.overhead}ms, Policy={args.policy})...")
    tenants = load_tenants_config(args.tenants_config)
    tenant_map = {t.tenant_id: t for t in tenants}

    sq = SchedulingQuantum(base_quantum_ms=args.quantum)
    policy_name = args.policy.lower()
    if policy_name in ("fifo",):
        sched = FIFOScheduler(tenants=tenant_map, quantum=sq)
    elif policy_name in ("round_robin", "rr"):
        sched = RoundRobinScheduler(tenants=tenant_map, quantum=sq)
    elif policy_name in ("priority", "prio"):
        sched = PriorityScheduler(tenants=tenant_map, quantum=sq)
    elif policy_name in ("weighted_round_robin", "wrr"):
        sched = WeightedRoundRobinScheduler(tenants=tenant_map, quantum=sq)
    elif policy_name in ("fair", "pdwfq", "gpu_fair"):
        sched = GPUFairScheduler(tenants=tenant_map, quantum=sq)
    else:
        raise ValueError(f"Unknown policy '{args.policy}'. Supported: fifo, round_robin, priority, weighted_round_robin, pdwfq")

    if getattr(args, "use_predictor", False) and hasattr(sched, "set_predictor"):
        from src.predictor.inference import WorkloadInferenceEngine
        engine = WorkloadInferenceEngine.from_model_file(args.model_path)
        sched.set_predictor(engine)
        print(f"[GPU Simulator] AI Workload Predictor attached from {args.model_path}.")

    # Sample representative multi-tenant workload jobs
    jobs = [
        GPUJob("A1", "tenant_robotics", "yolov8n", arrival_time=0.0, estimated_gpu_time=25.0, remaining_gpu_time=25.0, priority=10),
        GPUJob("B1", "tenant_surveillance", "mobilenet", arrival_time=0.0, estimated_gpu_time=15.0, remaining_gpu_time=15.0, priority=5),
        GPUJob("C1", "tenant_analytics", "audio", arrival_time=5.0, estimated_gpu_time=8.0, remaining_gpu_time=8.0, priority=1),
    ]

    result = run_scheduler_benchmark(
        scheduler=sched,
        jobs=jobs,
        quantum_ms=args.quantum,
        context_switch_overhead_ms=args.overhead,
    )

    print("\n--- Execution Trace ---")
    print(result.execution_trace)
    print("------------------------\n")

    from src.telemetry.metrics_engine import MetricsEngine
    engine = MetricsEngine()
    metrics_report = engine.compute(
        jobs=result.completed_jobs,
        total_simulation_time_ms=result.total_simulation_time_ms,
        tenants=list(sched.tenants.keys()),
        context_switch_overhead_ms=args.overhead,
    )
    print(metrics_report.format_summary_table())

    if getattr(args, "export_json", None):
        metrics_report.save_json(args.export_json)
        print(f"[+] Exported metrics to JSON: {args.export_json}")
    if getattr(args, "export_csv", None):
        metrics_report.save_csv(args.export_csv)
        print(f"[+] Exported summary metrics to CSV: {args.export_csv}")

    return 0


def cmd_train_predictor(args: argparse.Namespace) -> int:
    """Trains the AI workload prediction model."""
    from src.predictor.trainer import train_workload_predictor

    print(f"[*] Training AI workload predictor...")
    _, metrics = train_workload_predictor(
        dataset_path=args.dataset,
        output_model_path=args.output,
        test_size=args.test_size,
        random_seed=args.seed,
        num_samples=getattr(args, "samples", 1200),
    )
    print("\n" + metrics.summary())
    return 0


def cmd_evaluate_predictor(args: argparse.Namespace) -> int:
    """Evaluates the serialized AI workload predictor on test data."""
    from src.predictor.model import LightweightWorkloadPredictor
    from src.predictor.dataset import load_dataset_from_csv, train_test_split_dataset
    from src.predictor.evaluator import evaluate_workload_predictor

    if not os.path.exists(args.model):
        print(f"[-] Model file not found: {args.model}")
        return 1
    if not os.path.exists(args.dataset):
        print(f"[-] Dataset file not found: {args.dataset}")
        return 1

    predictor = LightweightWorkloadPredictor.load(args.model)
    X, y_class, y_latency = load_dataset_from_csv(args.dataset)
    _, X_test, _, y_class_test, _, y_lat_test = train_test_split_dataset(
        X, y_class, y_latency, test_size=args.test_size, random_seed=args.seed
    )
    pred_classes, confs, pred_lats = predictor.predict(X_test)
    metrics = evaluate_workload_predictor(
        y_class_true=y_class_test,
        y_class_pred=pred_classes,
        y_lat_true=y_lat_test,
        y_lat_pred=pred_lats,
    )
    print("\n" + metrics.summary())
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Executes multi-tenant benchmark suite comparing all 5 scheduling policies."""
    from simulator.run_experiment import run_experiment
    return run_experiment(args)


def cmd_visualize(args: argparse.Namespace) -> int:
    """Generates all 10 evaluation plots from benchmark results."""
    from simulator.plot_results import run_plotting
    return run_plotting(args)


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Launches the lightweight monitoring dashboard web server."""
    from simulator.dashboard import run_dashboard
    return run_dashboard(args)


def cmd_hardware(args: argparse.Namespace) -> int:
    """Inspects real hardware availability, Jetson/CUDA status, and telemetry."""
    from src.execution import create_executor

    print("=" * 65)
    print("  Hardware Execution Layer & Device Inspection")
    print("=" * 65)
    print("Architectural Boundary: Application-level workload dispatch.")
    print("No hardware-level SM/register context preemption is claimed.\n")

    backend = getattr(args, "backend", "auto")
    executor = create_executor(backend=backend)
    telemetry = executor.get_hardware_telemetry()

    print(f"Requested Backend : {backend}")
    print(f"Active Executor   : {executor.__class__.__name__}")
    print(f"Hardware Available: {'YES' if telemetry.hardware_available and telemetry.backend_type == 'jetson_cuda' else 'NO (Simulation Fallback)'}")
    print(f"Device Name       : {telemetry.device_name}")
    print(f"Driver / Runtime  : {telemetry.cuda_driver_version}")
    print(f"VRAM Memory       : {telemetry.memory_used_mb:.1f} MB / {telemetry.memory_total_mb:.1f} MB")

    if telemetry.gpu_utilization_pct is not None:
        print(f"GPU Utilization   : {telemetry.gpu_utilization_pct:.1f}%")
    if telemetry.temperature_celsius is not None:
        print(f"Temperature       : {telemetry.temperature_celsius:.1f} °C")
    if telemetry.power_watts is not None:
        print(f"Power Draw        : {telemetry.power_watts:.2f} W")

    if not (telemetry.hardware_available and telemetry.backend_type == "jetson_cuda"):
        print("\nNote: Physical CUDA/Jetson device was not detected on this host environment.")
        print("Discrete Event Simulation (DES) executor is active and fully functional.")

    print("=" * 65)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Constructs the root argument parser."""
    parser = argparse.ArgumentParser(
        prog="gpu-sched",
        description="GPU-Aware Fair Scheduler for Multi-Tenant Edge AI Inference",
    )
    parser.add_argument("-v", "--version", action="version", version=f"gpu-sched {__version__}")
    
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # info
    subparsers.add_parser("info", help="Display system and scheduler configuration info")

    # validate-config
    val_parser = subparsers.add_parser("validate-config", help="Validate YAML configuration files")
    val_parser.add_argument("--config-dir", type=str, default="configs", help="Directory containing YAML configs")

    # inspect-tenants
    insp_parser = subparsers.add_parser("inspect-tenants", help="Inspect registered tenants catalog")
    insp_parser.add_argument("--config", type=str, default="configs/tenants.yaml", help="Path to tenants.yaml")

    # smoke-test
    subparsers.add_parser("smoke-test", help="Execute Phase 1 core smoke test")

    # simulate
    sim_parser = subparsers.add_parser("simulate", help="Run event-driven GPU simulation demo")
    sim_parser.add_argument("--quantum", type=float, default=10.0, help="GPU time quantum in ms")
    sim_parser.add_argument("--overhead", type=float, default=0.5, help="Context switch overhead in ms")
    sim_parser.add_argument("--policy", type=str, default="round_robin", help="Scheduling policy (fifo, round_robin, priority, weighted_round_robin, pdwfq)")
    sim_parser.add_argument("--tenants-config", type=str, default="configs/tenants.yaml", help="Path to tenants.yaml")
    sim_parser.add_argument("--use-predictor", action="store_true", help="Attach AI workload predictor to scheduler")
    sim_parser.add_argument("--model-path", type=str, default="models/workload_predictor.joblib", help="Path to predictor model")
    sim_parser.add_argument("--export-json", type=str, default=None, help="Path to export simulation metrics in JSON format")
    sim_parser.add_argument("--export-csv", type=str, default=None, help="Path to export simulation metrics in CSV format")

    # train-predictor
    train_parser = subparsers.add_parser("train-predictor", help="Train AI workload prediction model")
    train_parser.add_argument("--dataset", type=str, default="data/workload_dataset.csv", help="Dataset path")
    train_parser.add_argument("--output", type=str, default="models/workload_predictor.joblib", help="Output model path")
    train_parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    train_parser.add_argument("--samples", type=int, default=1200, help="Number of synthetic samples to generate")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # evaluate-predictor
    eval_parser = subparsers.add_parser("evaluate-predictor", help="Evaluate AI workload prediction model")
    eval_parser.add_argument("--model", type=str, default="models/workload_predictor.joblib", help="Model path")
    eval_parser.add_argument("--dataset", type=str, default="data/workload_dataset.csv", help="Dataset path")
    eval_parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    eval_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run reproducible multi-tenant benchmark suite")
    bench_parser.add_argument("--config", "-c", type=str, default="configs/example.yaml", help="Path to benchmark config YAML")
    bench_parser.add_argument("--scenario", "-s", nargs="+", default=None, help="Scenario ID(s) to run (e.g. scenario_1)")
    bench_parser.add_argument("--policy", "-p", nargs="+", default=None, help="Scheduling policies to compare")
    bench_parser.add_argument("--seed", type=int, default=None, help="Deterministic random seed")
    bench_parser.add_argument("--quantum", type=float, default=None, help="Base time quantum in ms")
    bench_parser.add_argument("--overhead", type=float, default=None, help="Context switch overhead in ms")
    bench_parser.add_argument("--starvation-threshold", type=float, default=None, help="Starvation threshold in ms")
    bench_parser.add_argument("--output-dir", "-o", type=str, default=None, help="Output directory")
    bench_parser.add_argument("--no-traces", action="store_true", help="Disable saving trace files")
    bench_parser.add_argument("--use-predictor", action="store_true", help="Enable AI workload predictor")
    bench_parser.add_argument("--model-path", type=str, default="models/workload_predictor.joblib", help="Predictor model path")
    bench_parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")

    # visualize
    vis_parser = subparsers.add_parser("visualize", help="Generate all 10 evaluation plots from benchmark results")
    vis_parser.add_argument("--results-dir", "-r", type=str, default="results/benchmarks", help="Results directory containing benchmark_results.json")
    vis_parser.add_argument("--output-dir", "-o", type=str, default="results/plots", help="Output directory for generated plots")
    vis_parser.add_argument("--scenario", "-s", type=str, default=None, help="Specific scenario ID to visualize (default: scenario_10)")
    vis_parser.add_argument("--dpi", type=int, default=300, help="Figure resolution in DPI")

    # dashboard
    dash_parser = subparsers.add_parser("dashboard", help="Launch interactive monitoring dashboard")
    dash_parser.add_argument("--port", "-p", type=int, default=8501, help="Port to serve dashboard on")
    dash_parser.add_argument("--host", type=str, default="localhost", help="Host address to bind to")
    dash_parser.add_argument("--headless", action="store_true", help="Run without automatically opening browser")

    # hardware
    hw_parser = subparsers.add_parser("hardware", help="Inspect real hardware availability, Jetson/CUDA status, and telemetry")
    hw_parser.add_argument("--backend", "-b", type=str, default="auto", choices=["auto", "jetson", "simulation"], help="Execution backend to inspect")

    return parser


def main() -> None:
    """Main entrypoint function."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        sys.exit(0)

    handlers = {
        "info": cmd_info,
        "validate-config": cmd_validate_config,
        "inspect-tenants": cmd_inspect_tenants,
        "smoke-test": cmd_smoke_test,
        "simulate": cmd_simulate,
        "train-predictor": cmd_train_predictor,
        "evaluate-predictor": cmd_evaluate_predictor,
        "benchmark": cmd_benchmark,
        "visualize": cmd_visualize,
        "dashboard": cmd_dashboard,
        "hardware": cmd_hardware,
    }

    handler = handlers.get(args.subcommand)
    if handler:
        sys.exit(handler(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
