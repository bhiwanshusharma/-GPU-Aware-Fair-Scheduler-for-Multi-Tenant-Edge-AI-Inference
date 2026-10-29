"""Executor Factory and Discovery Utilities."""

import logging
from typing import Optional

from src.execution.base import HardwareExecutor
from src.execution.simulation_executor import SimulationExecutor
from src.execution.jetson_executor import JetsonExecutor

logger = logging.getLogger(__name__)


from src.core.exceptions import CUDAUnavailableError


def create_executor(
    backend: str = "auto",
    context_switch_overhead_ms: float = 0.5,
    device_index: int = 0,
    strict: bool = False,
) -> HardwareExecutor:
    """Instantiates a HardwareExecutor backend according to requested mode and hardware availability.
    
    Args:
        backend: Execution mode ('auto', 'jetson', 'cuda', 'simulation').
                 'auto' probes for physical NVIDIA CUDA/Jetson device and falls back
                 to SimulationExecutor if not detected.
        context_switch_overhead_ms: Hardware/scheduling delay penalty in ms.
        device_index: CUDA device index (default: 0).
        strict: If True, raises CUDAUnavailableError when physical hardware is unavailable.
        
    Returns:
        HardwareExecutor implementation (JetsonExecutor or SimulationExecutor).
    """
    mode = backend.strip().lower()

    if mode not in ("auto", "jetson", "cuda", "simulation", "sim"):
        raise ValueError(
            f"Unknown execution backend '{backend}'. Supported backends: 'auto', 'jetson', 'cuda', 'simulation'"
        )

    if mode in ("auto", "jetson", "cuda"):
        jetson = JetsonExecutor(
            device_index=device_index,
            context_switch_overhead_ms=context_switch_overhead_ms,
        )
        if jetson.is_available():
            logger.info("Hardware integration active: using JetsonExecutor on physical CUDA GPU.")
            return jetson

        if mode in ("jetson", "cuda"):
            if strict:
                raise CUDAUnavailableError(
                    "Physical CUDA/Jetson device is unavailable on this host and strict mode was requested."
                )
            logger.warning(
                "Requested 'jetson' / 'cuda' backend but CUDA runtime was not detected. "
                "Returning JetsonExecutor in unavailable state."
            )
            return jetson

        logger.info(
            "CUDA hardware runtime not detected on host. "
            "Falling back to SimulationExecutor (software simulation mode)."
        )
        return SimulationExecutor(context_switch_overhead_ms=context_switch_overhead_ms)

    # Explicit simulation backend
    return SimulationExecutor(context_switch_overhead_ms=context_switch_overhead_ms)
