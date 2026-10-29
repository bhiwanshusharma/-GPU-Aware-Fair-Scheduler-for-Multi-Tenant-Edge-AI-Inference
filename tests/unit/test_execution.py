"""Unit tests for the hardware execution layer and NVIDIA Jetson dispatch backend.

Verifies:
1. SimulationExecutor contracts, time-slicing, and telemetry collection.
2. JetsonExecutor safe detection and graceful degradation when CUDA is unavailable.
3. Hardware execution tests with CUDA event timing (skipped when physical GPU is not present).
4. Executor factory instantiation and automatic fallback.
"""

import pytest

try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available() and (torch.cuda.device_count() > 0)
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

from src.core.job import GPUJob
from src.execution.base import (
    HardwareExecutor,
    ExecutionResult,
    HardwareTelemetry,
    BoundedUnitType,
)
from src.execution.simulation_executor import SimulationExecutor
from src.execution.jetson_executor import JetsonExecutor
from src.execution.factory import create_executor


class TestSimulationExecutor:
    """Verifies the SimulationExecutor backend."""

    def test_simulation_executor_is_always_available(self):
        executor = SimulationExecutor()
        assert executor.is_available() is True
        telemetry = executor.get_hardware_telemetry()
        assert telemetry.hardware_available is True
        assert telemetry.backend_type == "simulation"
        assert telemetry.memory_total_mb > 0

    def test_simulation_dispatch_partial_slice_preemption(self):
        """Job has 30ms remaining, slice is 10ms -> remains 20ms, is_completed=False."""
        executor = SimulationExecutor(context_switch_overhead_ms=0.5)
        job = GPUJob(
            job_id="J1",
            tenant_id="tenant_A",
            model_name="yolov8",
            arrival_time=0.0,
            estimated_gpu_time=30.0,
            remaining_gpu_time=30.0,
        )

        result = executor.dispatch_slice(
            job=job,
            quantum_slice_ms=10.0,
            bounded_unit=BoundedUnitType.MICRO_BATCH,
        )

        assert isinstance(result, ExecutionResult)
        assert result.job_id == "J1"
        assert result.tenant_id == "tenant_A"
        assert result.actual_gpu_execution_time_ms == pytest.approx(10.0, 1e-4)
        assert result.is_completed is False
        assert result.remaining_gpu_time_ms == pytest.approx(20.0, 1e-4)
        assert job.remaining_gpu_time == pytest.approx(20.0, 1e-4)
        assert job.accumulated_execution_time == pytest.approx(10.0, 1e-4)
        assert result.scheduling_overhead_ms >= 0.5
        assert result.bounded_unit_type == BoundedUnitType.MICRO_BATCH
        assert result.memory_used_bytes > 0

    def test_simulation_dispatch_full_completion(self):
        """Job has 8ms remaining, slice is 15ms -> completes, remains 0ms, is_completed=True."""
        executor = SimulationExecutor()
        job = GPUJob(
            job_id="J2",
            tenant_id="tenant_B",
            model_name="mobilenet",
            arrival_time=0.0,
            estimated_gpu_time=8.0,
            remaining_gpu_time=8.0,
        )

        result = executor.dispatch_slice(job=job, quantum_slice_ms=15.0)

        assert result.actual_gpu_execution_time_ms == pytest.approx(8.0, 1e-4)
        assert result.is_completed is True
        assert result.remaining_gpu_time_ms == 0.0
        assert job.remaining_gpu_time == 0.0

    def test_invalid_quantum_slice_raises_value_error(self):
        executor = SimulationExecutor()
        job = GPUJob("J3", "t1", "m", arrival_time=0.0, estimated_gpu_time=10.0, remaining_gpu_time=10.0)
        with pytest.raises(ValueError):
            executor.dispatch_slice(job, quantum_slice_ms=0.0)
        with pytest.raises(ValueError):
            executor.dispatch_slice(job, quantum_slice_ms=-5.0)

    def test_simulation_executor_reset(self):
        executor = SimulationExecutor()
        job = GPUJob("J4", "t1", "m", arrival_time=0.0, estimated_gpu_time=10.0, remaining_gpu_time=10.0)
        executor.dispatch_slice(job, 10.0)
        executor.reset()
        assert executor._dispatches_count == 0


class TestJetsonExecutorGracefulDegradation:
    """Verifies that JetsonExecutor behaves safely on hosts without physical CUDA hardware."""

    def test_jetson_executor_reports_availability_status(self):
        executor = JetsonExecutor()
        # When running on host without CUDA (e.g. macOS CPU):
        if not CUDA_AVAILABLE:
            assert executor.is_available() is False
            telemetry = executor.get_hardware_telemetry()
            assert telemetry.hardware_available is False
            assert "Unavailable" in telemetry.device_name or "unavailable" in telemetry.device_name.lower()
        else:
            assert executor.is_available() is True
            telemetry = executor.get_hardware_telemetry()
            assert telemetry.hardware_available is True

    def test_dispatch_slice_without_cuda_raises_informative_error(self):
        executor = JetsonExecutor()
        if not CUDA_AVAILABLE:
            job = GPUJob("J5", "t1", "m", arrival_time=0.0, estimated_gpu_time=10.0, remaining_gpu_time=10.0)
            with pytest.raises(RuntimeError, match="JetsonExecutor is unavailable"):
                executor.dispatch_slice(job, quantum_slice_ms=10.0)

    def test_jetson_executor_reset_safe_without_cuda(self):
        executor = JetsonExecutor()
        # Should execute safely without crashing
        executor.reset()
        assert executor._dispatches_count == 0


class TestJetsonExecutorPhysicalHardware:
    """Tests executing real CUDA kernels and event timing when physical GPU is available."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires physical NVIDIA CUDA / Jetson GPU")
    def test_physical_cuda_dispatch_and_event_timing(self):
        executor = JetsonExecutor(enable_real_gemm_payload=True)
        assert executor.is_available() is True

        job = GPUJob(
            job_id="JETSON_JOB_1",
            tenant_id="tenant_orin",
            model_name="yolov8_nano",
            estimated_gpu_time=15.0,
            remaining_gpu_time=15.0,
        )

        result = executor.dispatch_slice(
            job=job,
            quantum_slice_ms=10.0,
            bounded_unit=BoundedUnitType.MICRO_BATCH,
        )

        assert result.actual_gpu_execution_time_ms == pytest.approx(10.0, 1e-4)
        assert result.is_completed is False
        assert result.remaining_gpu_time_ms == pytest.approx(5.0, 1e-4)
        assert "cuda_measured_kernel_ms" in result.metadata
        assert result.metadata["cuda_measured_kernel_ms"] > 0.0
        assert result.memory_total_bytes > 0


class TestExecutorFactory:
    """Verifies create_executor factory behavior and automatic fallback."""

    def test_create_simulation_executor_explicit(self):
        executor = create_executor(backend="simulation")
        assert isinstance(executor, SimulationExecutor)
        assert executor.is_available() is True

    def test_create_executor_auto_mode(self):
        executor = create_executor(backend="auto")
        if CUDA_AVAILABLE:
            assert isinstance(executor, JetsonExecutor)
        else:
            # Safely falls back to SimulationExecutor
            assert isinstance(executor, SimulationExecutor)
            assert executor.is_available() is True

    def test_create_executor_jetson_mode(self):
        executor = create_executor(backend="jetson")
        assert isinstance(executor, JetsonExecutor)
        if not CUDA_AVAILABLE:
            assert executor.is_available() is False
