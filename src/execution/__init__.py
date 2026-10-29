"""Execution backends for simulated and physical NVIDIA Jetson GPU workload dispatch."""

from src.execution.base import (
    HardwareExecutor,
    IGPUExecutor,
    ExecutionResult,
    HardwareTelemetry,
    BoundedUnitType,
)
from src.execution.simulation_executor import SimulationExecutor
from src.execution.jetson_executor import JetsonExecutor
from src.execution.factory import create_executor

__all__ = [
    "HardwareExecutor",
    "IGPUExecutor",
    "ExecutionResult",
    "HardwareTelemetry",
    "BoundedUnitType",
    "SimulationExecutor",
    "JetsonExecutor",
    "create_executor",
]
