"""Configuration loading and validation for tenants, workloads, and simulation parameters."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml

from src.core.types import SchedulingQuantum, PolicyType
from src.core.tenant import Tenant
from src.core.workload import Workload, WorkloadModelProfile


@dataclass
class SimulationConfig:
    """Runtime parameters for the Discrete Event Simulation engine."""
    duration_ms: float = 60000.0
    random_seed: int = 42
    clock_step_min_ms: float = 0.1
    policy: PolicyType = PolicyType.PDWFQ
    quantum: SchedulingQuantum = field(default_factory=SchedulingQuantum)
    traffic_pattern: str = "poisson"
    total_load_target_util: float = 0.85
    arrival_rates: Dict[str, float] = field(default_factory=dict)
    log_level: str = "INFO"
    save_raw_traces: bool = True
    results_directory: str = "results"


import logging
from src.core.exceptions import ConfigurationCorruptedError

logger = logging.getLogger(__name__)


def load_tenants_config(config_path: str | Path) -> List[Tenant]:
    """Loads and validates tenant configurations from a YAML file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Tenant configuration file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Corrupted YAML in tenants configuration {path}: {e}")
        raise ConfigurationCorruptedError(f"Corrupted YAML syntax in tenant config {path}: {e}")

    if not data or not isinstance(data, dict) or "tenants" not in data or not isinstance(data["tenants"], list):
        logger.error(f"Invalid tenant config schema in {path}: missing or invalid 'tenants' root list")
        raise ConfigurationCorruptedError(f"Invalid tenant config: missing or malformed 'tenants' root list in {path}")

    tenants: List[Tenant] = []
    total_quota = 0.0

    for idx, item in enumerate(data["tenants"]):
        if not isinstance(item, dict):
            raise ConfigurationCorruptedError(f"Malformed tenant record at index {idx} in {path}")
        try:
            tenant = Tenant(
                tenant_id=item["tenant_id"],
                name=item["name"],
                weight=float(item["weight"]),
                priority=int(item["priority"]),
                gpu_quota_pct=float(item["gpu_quota_pct"]),
                quota_window_ms=float(item.get("quota_window_ms", 1000.0)),
                sla_latency_target_ms=float(item.get("sla_latency_target_ms", 100.0)),
                max_queue_capacity=int(item.get("max_queue_capacity", 100)),
                workload_type=item.get("workload_type", "GENERIC"),
            )
        except (KeyError, TypeError) as e:
            raise ConfigurationCorruptedError(f"Corrupted tenant configuration entry in {path}: {e}")
        total_quota += tenant.gpu_quota_pct
        tenants.append(tenant)

    return tenants


def load_workloads_config(config_path: str | Path) -> Dict[str, WorkloadModelProfile]:
    """Loads and validates workload definitions from a YAML file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Workload configuration file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Corrupted YAML in workloads configuration {path}: {e}")
        raise ConfigurationCorruptedError(f"Corrupted YAML syntax in workload config {path}: {e}")

    if not data or not isinstance(data, dict) or "workloads" not in data or not isinstance(data["workloads"], list):
        logger.error(f"Invalid workload config schema in {path}: missing or invalid 'workloads' root list")
        raise ConfigurationCorruptedError(f"Invalid workload config: missing or malformed 'workloads' root list in {path}")

    workloads: Dict[str, WorkloadModelProfile] = {}
    for idx, item in enumerate(data["workloads"]):
        if not isinstance(item, dict):
            raise ConfigurationCorruptedError(f"Malformed workload record at index {idx} in {path}")
        try:
            profile = WorkloadModelProfile(
                model_name=item["model_name"],
                domain=item["domain"],
                category=item["category"],
                input_shape=tuple(item["input_shape"]),
                default_batch_size=int(item.get("default_batch_size", 1)),
                max_batch_size=int(item.get("max_batch_size", 16)),
                estimated_flops_g=float(item.get("estimated_flops_g", 1.0)),
                memory_footprint_mb=float(item.get("memory_footprint_mb", 10.0)),
                calibrated_base_latency_ms=float(item["calibrated_base_latency_ms"]),
                latency_per_batch_item_ms=float(item.get("latency_per_batch_item_ms", 1.0)),
            )
        except (KeyError, TypeError) as e:
            raise ConfigurationCorruptedError(f"Corrupted workload configuration entry in {path}: {e}")
        workloads[profile.model_name] = profile

    return workloads


def load_simulation_config(config_path: str | Path) -> SimulationConfig:
    """Loads runtime simulation parameters from a YAML file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Simulation configuration file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Corrupted YAML in simulation configuration {path}: {e}")
        raise ConfigurationCorruptedError(f"Corrupted YAML syntax in simulation config {path}: {e}")

    if not data or not isinstance(data, dict):
        raise ConfigurationCorruptedError(f"Corrupted simulation configuration root in {path}")

    sim_data = data.get("simulation", {})
    sched_data = data.get("scheduler", {})
    workload_data = data.get("workload_generation", {})
    out_data = data.get("output", {})

    try:
        quantum = SchedulingQuantum(
            base_quantum_ms=float(sched_data.get("base_quantum_ms", 25.0)),
            min_quantum_ms=float(sched_data.get("min_quantum_ms", 5.0)),
            max_quantum_ms=float(sched_data.get("max_quantum_ms", 100.0)),
            aging_alpha=float(sched_data.get("aging_alpha", 0.15)),
            starvation_threshold_ms=float(sched_data.get("starvation_threshold_ms", 150.0)),
        )

        policy_str = sched_data.get("policy", "pdwfq").lower()
        policy = PolicyType(policy_str)

        duration_ms = float(sim_data.get("duration_ms", 60000.0))
        if duration_ms <= 0.0:
            raise ConfigurationCorruptedError(f"simulation duration_ms must be positive, got {duration_ms}")

        return SimulationConfig(
            duration_ms=duration_ms,
            random_seed=int(sim_data.get("random_seed", 42)),
            clock_step_min_ms=float(sim_data.get("clock_step_min_ms", 0.1)),
            policy=policy,
            quantum=quantum,
            traffic_pattern=workload_data.get("traffic_pattern", "poisson"),
            total_load_target_util=float(workload_data.get("total_load_target_util", 0.85)),
            arrival_rates=workload_data.get("arrival_rates", {}),
            log_level=out_data.get("log_level", "INFO"),
            save_raw_traces=bool(out_data.get("save_raw_traces", True)),
            results_directory=out_data.get("results_directory", "results"),
        )
    except (ValueError, KeyError, TypeError) as e:
        if isinstance(e, ConfigurationCorruptedError):
            raise
        raise ConfigurationCorruptedError(f"Corrupted simulation config parameters in {path}: {e}")
