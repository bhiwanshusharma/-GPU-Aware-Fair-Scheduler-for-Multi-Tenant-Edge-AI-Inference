"""Unit tests for YAML configuration loaders."""

from pathlib import Path
import pytest
from src.core.config import (
    load_tenants_config,
    load_workloads_config,
    load_simulation_config,
)
from src.core.types import PolicyType


def test_load_default_tenants():
    path = Path("configs/tenants.yaml")
    assert path.is_file(), "configs/tenants.yaml missing"

    tenants = load_tenants_config(path)
    assert len(tenants) == 3
    tenant_ids = [t.tenant_id for t in tenants]
    assert "tenant_robotics" in tenant_ids
    assert "tenant_surveillance" in tenant_ids
    assert "tenant_analytics" in tenant_ids

    robotics = next(t for t in tenants if t.tenant_id == "tenant_robotics")
    assert robotics.weight == 4.0
    assert robotics.priority == 10
    assert robotics.gpu_quota_pct == 40.0


def test_load_default_workloads():
    path = Path("configs/workloads.yaml")
    assert path.is_file(), "configs/workloads.yaml missing"

    workloads = load_workloads_config(path)
    assert len(workloads) == 4
    assert "yolov8n_detection" in workloads
    assert "mobilenetv3_classification" in workloads
    assert "keyword_spotting_audio" in workloads
    assert "sensor_anomaly_mlp" in workloads

    yolo = workloads["yolov8n_detection"]
    assert yolo.category == "GPU_INTENSIVE"
    assert yolo.calibrated_base_latency_ms > 0.0


def test_load_default_simulation_config():
    path = Path("configs/default_simulation.yaml")
    assert path.is_file(), "configs/default_simulation.yaml missing"

    sim_cfg = load_simulation_config(path)
    assert sim_cfg.duration_ms == 60000.0
    assert sim_cfg.random_seed == 42
    assert sim_cfg.policy == PolicyType.PDWFQ
    assert sim_cfg.quantum.base_quantum_ms == 25.0
    assert "tenant_robotics" in sim_cfg.arrival_rates


def test_missing_config_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_tenants_config("configs/non_existent.yaml")
