"""Comprehensive Failure-Mode Analysis Test Suite.

Verifies robust, fail-safe handling across all 18 identified failure modes:
1. Invalid tenant configuration
2. Invalid GPU quantum
3. Zero GPU execution time
4. Negative execution time
5. Empty queues
6. Tenant quota exhaustion
7. Scheduler starvation
8. Predictor unavailable
9. Predictor returns invalid output
10. Missing ML model
11. Corrupted experiment configuration
12. CUDA unavailable
13. TensorRT unavailable
14. GPU memory allocation failure
15. Hardware executor failure
16. Job cancellation
17. Duplicate job IDs
18. Unknown tenant IDs
"""

import math
import os
import tempfile
from unittest.mock import MagicMock, patch
import pytest

from src.core.types import JobState, WorkloadClass, SchedulingQuantum, PredictionOutput
from src.core.job import GPUJob, Job
from src.core.tenant import Tenant
from src.core.queues import TenantJobQueue, TenantQueueManager
from src.core.scheduler_config import SchedulerConfiguration
from src.core.config import (
    load_tenants_config,
    load_workloads_config,
    load_simulation_config,
)
from src.core.exceptions import (
    GPUSchedulerError,
    InvalidTenantConfigError,
    InvalidQuantumError,
    InvalidExecutionTimeError,
    EmptyQueueError,
    TenantQuotaExhaustedError,
    SchedulerStarvationError,
    PredictorUnavailableError,
    InvalidPredictorOutputError,
    ModelNotFoundError,
    ConfigurationCorruptedError,
    CUDAUnavailableError,
    TensorRTUnavailableError,
    GPUMemoryAllocationError,
    HardwareExecutionError,
    JobCancelledError,
    DuplicateJobError,
    UnknownTenantError,
)
from src.scheduler.fifo import FIFOScheduler
from src.scheduler.round_robin import RoundRobinScheduler
from src.scheduler.priority import PriorityScheduler
from src.scheduler.weighted_round_robin import WeightedRoundRobinScheduler
from src.scheduler.fair_scheduler import GPUFairScheduler
from src.simulation.simulator import GPUSimulator
from src.predictor.model import LightweightWorkloadPredictor
from src.predictor.inference import WorkloadInferenceEngine
from src.execution.simulation_executor import SimulationExecutor
from src.execution.jetson_executor import JetsonExecutor
from src.execution.factory import create_executor


# ============================================================================
# Failure Mode 1: Invalid Tenant Configuration
# ============================================================================
class TestFailureMode1InvalidTenantConfig:
    """Verifies that invalid tenant definitions fail fast with InvalidTenantConfigError."""

    def test_empty_tenant_id(self):
        with pytest.raises(InvalidTenantConfigError):
            Tenant(tenant_id="", priority=5, weight=1.0, gpu_quota=50.0)

    def test_whitespace_tenant_id(self):
        with pytest.raises(InvalidTenantConfigError):
            Tenant(tenant_id="   ", priority=5, weight=1.0, gpu_quota=50.0)

    def test_priority_below_minimum(self):
        with pytest.raises(InvalidTenantConfigError):
            Tenant(tenant_id="t1", priority=0, weight=1.0, gpu_quota=50.0)

    def test_priority_above_maximum(self):
        with pytest.raises(InvalidTenantConfigError):
            Tenant(tenant_id="t1", priority=11, weight=1.0, gpu_quota=50.0)

    def test_negative_weight(self):
        with pytest.raises(InvalidTenantConfigError):
            Tenant(tenant_id="t1", priority=5, weight=-2.0, gpu_quota=50.0)

    def test_zero_weight(self):
        with pytest.raises(InvalidTenantConfigError):
            Tenant(tenant_id="t1", priority=5, weight=0.0, gpu_quota=50.0)

    def test_nan_or_inf_weight(self):
        with pytest.raises(InvalidTenantConfigError):
            Tenant(tenant_id="t1", priority=5, weight=float("nan"), gpu_quota=50.0)
        with pytest.raises(InvalidTenantConfigError):
            Tenant(tenant_id="t1", priority=5, weight=float("inf"), gpu_quota=50.0)

    def test_negative_or_zero_quota(self):
        with pytest.raises(InvalidTenantConfigError):
            Tenant(tenant_id="t1", priority=5, weight=1.0, gpu_quota=0.0)
        with pytest.raises(InvalidTenantConfigError):
            Tenant(tenant_id="t1", priority=5, weight=1.0, gpu_quota=-10.0)

    def test_quota_exceeding_100_percent(self):
        with pytest.raises(InvalidTenantConfigError):
            Tenant(tenant_id="t1", priority=5, weight=1.0, gpu_quota=105.0)

    def test_invalid_window_or_sla(self):
        with pytest.raises(InvalidTenantConfigError):
            Tenant(tenant_id="t1", priority=5, weight=1.0, gpu_quota=50.0, quota_window_ms=-500.0)
        with pytest.raises(InvalidTenantConfigError):
            Tenant(tenant_id="t1", priority=5, weight=1.0, gpu_quota=50.0, sla_latency_target_ms=0.0)
        with pytest.raises(InvalidTenantConfigError):
            Tenant(tenant_id="t1", priority=5, weight=1.0, gpu_quota=50.0, max_queue_capacity=0)


# ============================================================================
# Failure Mode 2: Invalid GPU Quantum
# ============================================================================
class TestFailureMode2InvalidGPUQuantum:
    """Verifies that non-positive or inverted quanta raise InvalidQuantumError."""

    def test_negative_base_quantum(self):
        with pytest.raises(InvalidQuantumError):
            SchedulingQuantum(base_quantum_ms=-10.0)

    def test_zero_base_quantum(self):
        with pytest.raises(InvalidQuantumError):
            SchedulingQuantum(base_quantum_ms=0.0)

    def test_min_exceeds_base_quantum(self):
        with pytest.raises(InvalidQuantumError):
            SchedulingQuantum(min_quantum_ms=30.0, base_quantum_ms=20.0)

    def test_base_exceeds_max_quantum(self):
        with pytest.raises(InvalidQuantumError):
            SchedulingQuantum(base_quantum_ms=120.0, max_quantum_ms=100.0)

    def test_scheduler_configuration_invalid_quantum(self):
        with pytest.raises(InvalidQuantumError):
            SchedulerConfiguration(base_quantum_ms=-5.0)
        with pytest.raises(InvalidQuantumError):
            SchedulerConfiguration(min_quantum_ms=50.0, base_quantum_ms=25.0)

    def test_simulator_invalid_quantum(self):
        with pytest.raises(InvalidQuantumError):
            GPUSimulator(quantum_ms=-1.0)
        with pytest.raises(InvalidQuantumError):
            GPUSimulator(quantum_ms=0.0)

    def test_executors_invalid_quantum(self):
        exec_sim = SimulationExecutor()
        job = GPUJob(job_id="j1", tenant_id="t1", model_name="m1", arrival_time=0.0, estimated_gpu_time=10.0)
        with pytest.raises(InvalidQuantumError):
            exec_sim.dispatch_slice(job, quantum_slice_ms=0.0)
        with pytest.raises(InvalidQuantumError):
            exec_sim.dispatch_slice(job, quantum_slice_ms=-10.0)


# ============================================================================
# Failure Mode 3: Zero GPU Execution Time
# ============================================================================
class TestFailureMode3ZeroGPUExecutionTime:
    """Verifies that zero execution time jobs are rejected safely."""

    def test_simulator_rejects_zero_estimated_time(self):
        sim = GPUSimulator(quantum_ms=10.0)
        job = GPUJob(job_id="j_zero", tenant_id="t1", model_name="m", arrival_time=0.0, estimated_gpu_time=0.0)
        with pytest.raises(InvalidExecutionTimeError):
            sim.submit_job(job)

    def test_executor_rejects_zero_remaining_time(self):
        exec_sim = SimulationExecutor()
        job = GPUJob(job_id="j_zero", tenant_id="t1", model_name="m", arrival_time=0.0, estimated_gpu_time=10.0)
        job.remaining_gpu_time = 0.0
        with pytest.raises(InvalidExecutionTimeError):
            exec_sim.dispatch_slice(job, quantum_slice_ms=10.0)


# ============================================================================
# Failure Mode 4: Negative Execution Time
# ============================================================================
class TestFailureMode4NegativeExecutionTime:
    """Verifies that negative execution time is caught and raises InvalidExecutionTimeError."""

    def test_job_constructor_negative_estimated_time(self):
        with pytest.raises(InvalidExecutionTimeError):
            GPUJob(job_id="j_neg", tenant_id="t1", model_name="m", arrival_time=0.0, estimated_gpu_time=-5.0)

    def test_job_constructor_negative_remaining_time(self):
        with pytest.raises(InvalidExecutionTimeError):
            GPUJob(
                job_id="j_neg",
                tenant_id="t1",
                model_name="m",
                arrival_time=0.0,
                estimated_gpu_time=10.0,
                remaining_gpu_time=-2.0,
            )

    def test_execute_slice_negative_duration(self):
        job = GPUJob(job_id="j1", tenant_id="t1", model_name="m", arrival_time=0.0, estimated_gpu_time=10.0)
        with pytest.raises(InvalidExecutionTimeError):
            job.execute_slice(actual_gpu_time=-3.0, current_time=0.0)


# ============================================================================
# Failure Mode 5: Empty Queues
# ============================================================================
class TestFailureMode5EmptyQueues:
    """Verifies safe operation with empty queues without crashing or busy-waiting."""

    def test_tenant_job_queue_empty_dequeue(self):
        q = TenantJobQueue(tenant_id="t1")
        assert q.dequeue() is None
        assert q.peek() is None
        assert q.is_empty is True
        with pytest.raises(EmptyQueueError):
            q.dequeue_strict()

    def test_all_schedulers_safe_on_empty_queues(self):
        qm = TenantQueueManager()
        t1 = Tenant("t1", priority=1, weight=1.0, gpu_quota=100.0)
        qm.register_tenant(t1)

        schedulers = [
            FIFOScheduler(tenants={"t1": t1}, queue_manager=qm),
            RoundRobinScheduler(tenants={"t1": t1}, queue_manager=qm),
            PriorityScheduler(tenants={"t1": t1}, queue_manager=qm),
            WeightedRoundRobinScheduler(tenants={"t1": t1}, queue_manager=qm),
            GPUFairScheduler(tenants={"t1": t1}, queue_manager=qm),
        ]

        for s in schedulers:
            winner = s.schedule(current_time_ms=0.0)
            assert winner is None, f"Scheduler {s.name} should return None on empty queue"

    def test_simulator_idle_with_empty_queues(self):
        sim = GPUSimulator(quantum_ms=10.0)
        trace = sim.run(until_time_ms=100.0)
        assert len(sim.completed_jobs) == 0
        assert sim.gpu.is_busy is False


# ============================================================================
# Failure Mode 6: Tenant Quota Exhaustion
# ============================================================================
class TestFailureMode6TenantQuotaExhaustion:
    """Verifies quota enforcement and safe throttling upon exhaustion."""

    def test_strict_quota_enforcement_raises_exception(self):
        t = Tenant("t1", priority=5, weight=1.0, gpu_quota=20.0, quota_window_ms=1000.0)
        # 20% of 1000ms = 200ms allowed
        t.record_gpu_execution(gpu_time_ms=190.0, current_time_ms=10.0)
        assert t.is_quota_exceeded(current_time_ms=20.0) is False

        t.record_gpu_execution(gpu_time_ms=20.0, current_time_ms=30.0)
        # Consumed 210ms >= 200ms
        assert t.is_quota_exceeded(current_time_ms=40.0) is True

        with pytest.raises(TenantQuotaExhaustedError):
            t.enforce_quota_strict(current_time_ms=40.0)

    def test_fair_scheduler_throttles_exhausted_tenant(self):
        qm = TenantQueueManager()
        t1 = Tenant("t1", priority=5, weight=1.0, gpu_quota=10.0, quota_window_ms=1000.0)
        qm.register_tenant(t1)
        sched = GPUFairScheduler(tenants={"t1": t1}, queue_manager=qm, enable_quota_enforcement=True)

        # Enqueue job
        job = GPUJob("j1", "t1", "m", arrival_time=0.0, estimated_gpu_time=150.0)
        qm.enqueue(job)

        # Consume full quota
        t1.record_gpu_execution(105.0, 10.0)
        assert t1.is_quota_exceeded(10.0) is True

        # Scheduler must skip throttled tenant safely and return None
        winner = sched.schedule(current_time_ms=20.0)
        assert winner is None


# ============================================================================
# Failure Mode 7: Scheduler Starvation
# ============================================================================
class TestFailureMode7SchedulerStarvation:
    """Verifies that prolonged waiting triggers starvation warnings and aging escalation."""

    def test_starvation_violation_detection_and_strict_error(self):
        qm = TenantQueueManager()
        t1 = Tenant("t1", priority=1, weight=1.0, gpu_quota=100.0)
        qm.register_tenant(t1)
        sched = GPUFairScheduler(
            tenants={"t1": t1},
            queue_manager=qm,
            starvation_threshold_ms=50.0,
        )

        job = GPUJob("j_starve", "t1", "m", arrival_time=0.0, estimated_gpu_time=20.0)
        qm.enqueue(job)

        # Check starvation at t=500ms (10x threshold)
        violations = sched.check_starvation_violations(current_time_ms=550.0, critical_multiplier=10.0)
        assert "j_starve" in violations

        with pytest.raises(SchedulerStarvationError):
            sched.check_starvation_violations(current_time_ms=550.0, critical_multiplier=10.0, strict=True)


# ============================================================================
# Failure Mode 8: Predictor Unavailable
# ============================================================================
class TestFailureMode8PredictorUnavailable:
    """Verifies fallback heuristics when ML model is disabled or uninitialized."""

    def test_inference_engine_fallback_when_disabled(self):
        engine = WorkloadInferenceEngine(predictor=None, enabled=False)
        job = GPUJob("j1", "t1", "yolov8n", arrival_time=0.0, estimated_gpu_time=25.0)

        # Safe fallback produces valid classification
        out = engine.predict_job(job)
        assert isinstance(out, PredictionOutput)
        assert out.classification in list(WorkloadClass)

    def test_strict_prediction_raises_when_unavailable(self):
        engine = WorkloadInferenceEngine(predictor=None, enabled=True)
        job = GPUJob("j1", "t1", "m", arrival_time=0.0, estimated_gpu_time=25.0)
        with pytest.raises(PredictorUnavailableError):
            engine.predict_strict(job)


# ============================================================================
# Failure Mode 9: Predictor Returns Invalid Output
# ============================================================================
class TestFailureMode9PredictorInvalidOutput:
    """Verifies sanitization and validation of corrupt/invalid ML outputs."""

    def test_prediction_output_rejects_nan_and_inf(self):
        with pytest.raises(InvalidPredictorOutputError):
            PredictionOutput(
                estimated_latency_ms=float("nan"),
                estimated_memory_mb=100.0,
                classification=WorkloadClass.BALANCED,
            )
        with pytest.raises(InvalidPredictorOutputError):
            PredictionOutput(
                estimated_latency_ms=10.0,
                estimated_memory_mb=float("inf"),
                classification=WorkloadClass.BALANCED,
            )
        with pytest.raises(InvalidPredictorOutputError):
            PredictionOutput(
                estimated_latency_ms=10.0,
                estimated_memory_mb=50.0,
                classification=WorkloadClass.BALANCED,
                confidence_score=1.5,
            )

    def test_scheduler_sanitizes_nan_latency_from_mock_predictor(self):
        qm = TenantQueueManager()
        t1 = Tenant("t1", priority=5, weight=1.0, gpu_quota=100.0)
        qm.register_tenant(t1)

        # Mock predictor that returns invalid NaN latency
        mock_pred = MagicMock()
        mock_pred.predict.return_value = MagicMock(
            estimated_latency_ms=float("nan"),
            confidence_score=0.9,
            classification=WorkloadClass.GPU_INTENSIVE,
        )

        sched = GPUFairScheduler(tenants={"t1": t1}, queue_manager=qm, predictor=mock_pred)
        job = GPUJob("j1", "t1", "m", arrival_time=0.0, estimated_gpu_time=20.0)
        qm.enqueue(job)

        # Must not crash or produce NaN total_score
        breakdown = sched.calculate_score(job, current_time_ms=0.0, v_bar=0.0)
        assert not math.isnan(breakdown.total_score)
        assert not math.isinf(breakdown.total_score)


# ============================================================================
# Failure Mode 10: Missing ML Model
# ============================================================================
class TestFailureMode10MissingMLModel:
    """Verifies handling of non-existent or corrupted model artifacts."""

    def test_load_nonexistent_model_raises_model_not_found(self):
        with pytest.raises(ModelNotFoundError):
            LightweightWorkloadPredictor.load("/tmp/definitely_missing_model.joblib")

    def test_load_corrupted_model_file_raises_model_not_found(self):
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp.write(b"NOT_A_VALID_JOBLIB_PICKLE_GARBAGE_BYTES")
            tmp_path = tmp.name

        try:
            with pytest.raises(ModelNotFoundError):
                LightweightWorkloadPredictor.load(tmp_path)
        finally:
            os.remove(tmp_path)

    def test_inference_engine_from_model_file_strict_vs_fallback(self):
        # Strict mode raises
        with pytest.raises(ModelNotFoundError):
            WorkloadInferenceEngine.from_model_file("/tmp/missing.joblib", strict=True)

        # Non-strict mode logs warning and degrades gracefully
        engine = WorkloadInferenceEngine.from_model_file("/tmp/missing.joblib", strict=False)
        assert engine.predictor is None


# ============================================================================
# Failure Mode 11: Corrupted Experiment Configuration
# ============================================================================
class TestFailureMode11CorruptedConfig:
    """Verifies that invalid YAML syntax or malformed schemas raise ConfigurationCorruptedError."""

    def test_corrupted_yaml_syntax(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as tmp:
            tmp.write("tenants: [unbalanced brackets: {bad syntax")
            tmp_path = tmp.name

        try:
            with pytest.raises(ConfigurationCorruptedError):
                load_tenants_config(tmp_path)
            with pytest.raises(ConfigurationCorruptedError):
                load_workloads_config(tmp_path)
            with pytest.raises(ConfigurationCorruptedError):
                load_simulation_config(tmp_path)
        finally:
            os.remove(tmp_path)

    def test_missing_root_key_raises_corrupted_error(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as tmp:
            tmp.write("unrelated_root: 123\n")
            tmp_path = tmp.name

        try:
            with pytest.raises(ConfigurationCorruptedError):
                load_tenants_config(tmp_path)
            with pytest.raises(ConfigurationCorruptedError):
                load_workloads_config(tmp_path)
        finally:
            os.remove(tmp_path)


# ============================================================================
# Failure Mode 12: CUDA Unavailable
# ============================================================================
class TestFailureMode12CUDAUnavailable:
    """Verifies graceful handling when CUDA runtime is absent."""

    def test_jetson_executor_dispatch_without_cuda_raises_cuda_unavailable_error(self):
        executor = JetsonExecutor()
        job = GPUJob("j1", "t1", "m", arrival_time=0.0, estimated_gpu_time=10.0)
        if not executor.is_available():
            with pytest.raises(CUDAUnavailableError):
                executor.dispatch_slice(job, quantum_slice_ms=10.0)

    def test_create_executor_strict_cuda_mode(self):
        if not JetsonExecutor().is_available():
            with pytest.raises(CUDAUnavailableError):
                create_executor(backend="jetson", strict=True)


# ============================================================================
# Failure Mode 13: TensorRT Unavailable
# ============================================================================
class TestFailureMode13TensorRTUnavailable:
    """Verifies handling when TensorRT Python package is missing."""

    def test_enable_tensorrt_strict_mode(self):
        executor = JetsonExecutor()
        # TensorRT is not installed in standard test host
        with patch("src.execution.jetson_executor.TENSORRT_AVAILABLE", False):
            with pytest.raises(TensorRTUnavailableError):
                executor.enable_tensorrt(strict=True)

            # Non-strict mode returns False without crashing
            assert executor.enable_tensorrt(strict=False) is False


# ============================================================================
# Failure Mode 14: GPU Memory Allocation Failure
# ============================================================================
class TestFailureMode14GPUMemoryAllocationFailure:
    """Verifies that out-of-memory conditions raise GPUMemoryAllocationError."""

    def test_simulation_executor_vram_limit_exceeded(self):
        # 1024 MB total memory
        executor = SimulationExecutor(total_memory_mb=1024.0)
        # Giant batch size demanding > 1024 MB VRAM
        job = GPUJob(
            "j_giant",
            "t1",
            "resnet50",
            arrival_time=0.0,
            estimated_gpu_time=50.0,
            batch_size=16,  # 800MB * 16 = 12800 MB >> 1024 MB
        )
        with pytest.raises(GPUMemoryAllocationError):
            executor.dispatch_slice(job, quantum_slice_ms=25.0)


# ============================================================================
# Failure Mode 15: Hardware Executor Failure
# ============================================================================
class TestFailureMode15HardwareExecutorFailure:
    """Verifies that unexpected hardware exceptions raise HardwareExecutionError."""

    def test_hardware_execution_error_wrapped(self):
        executor = JetsonExecutor()
        job = GPUJob("j1", "t1", "m", arrival_time=0.0, estimated_gpu_time=10.0)

        with patch.object(executor, "is_available", return_value=True):
            with patch("src.execution.jetson_executor.torch") as mock_torch:
                # Force hardware exception on stream or event
                mock_torch.cuda.Event.side_effect = RuntimeError("CUDA driver kernel launch failure")
                mock_torch.cuda.stream.side_effect = RuntimeError("Stream synchronization fault")
                with pytest.raises(HardwareExecutionError):
                    executor.dispatch_slice(job, quantum_slice_ms=10.0)


# ============================================================================
# Failure Mode 16: Job Cancellation
# ============================================================================
class TestFailureMode16JobCancellation:
    """Verifies safe lifecycle transition and scheduling guard for cancelled jobs."""

    def test_job_cancel_state_and_forbidden_execution(self):
        job = GPUJob("j_cancel", "t1", "m", arrival_time=0.0, estimated_gpu_time=20.0)
        job.cancel(reason="Client revoked request")
        assert job.status == JobState.CANCELLED
        assert job.metadata["cancellation_reason"] == "Client revoked request"

        # Forbidden operations on cancelled jobs
        with pytest.raises(JobCancelledError):
            job.mark_started(start_time=1.0)
        with pytest.raises(JobCancelledError):
            job.execute_slice(actual_gpu_time=5.0, current_time=1.0)
        with pytest.raises(JobCancelledError):
            job.transition_to(JobState.RUNNING)

    def test_queue_and_simulator_cancel_job(self):
        q = TenantJobQueue("t1")
        job = GPUJob("j1", "t1", "m", arrival_time=0.0, estimated_gpu_time=20.0)
        q.enqueue(job)
        assert q.size == 1

        cancelled = q.cancel_job("j1", reason="User cancel")
        assert cancelled is not None
        assert cancelled.status == JobState.CANCELLED
        assert q.size == 0
        assert q.is_empty is True

        # Enqueueing already cancelled job is rejected
        with pytest.raises(JobCancelledError):
            q.enqueue(cancelled)


# ============================================================================
# Failure Mode 17: Duplicate Job IDs
# ============================================================================
class TestFailureMode17DuplicateJobIDs:
    """Verifies rejection of duplicate job IDs across queues and simulator."""

    def test_queue_rejects_duplicate_job_id(self):
        q = TenantJobQueue("t1")
        j1 = GPUJob("dup_id", "t1", "m", arrival_time=0.0, estimated_gpu_time=10.0)
        j2 = GPUJob("dup_id", "t1", "m", arrival_time=5.0, estimated_gpu_time=15.0)

        assert q.enqueue(j1) is True
        with pytest.raises(DuplicateJobError):
            q.enqueue(j2)

    def test_simulator_rejects_duplicate_job_submission(self):
        sim = GPUSimulator(quantum_ms=10.0)
        j1 = GPUJob("dup_id", "t1", "m", arrival_time=0.0, estimated_gpu_time=10.0)
        j2 = GPUJob("dup_id", "t1", "m", arrival_time=5.0, estimated_gpu_time=15.0)

        sim.submit_job(j1)
        with pytest.raises(DuplicateJobError):
            sim.submit_job(j2)


# ============================================================================
# Failure Mode 18: Unknown Tenant IDs
# ============================================================================
class TestFailureMode18UnknownTenantIDs:
    """Verifies that referencing unregistered tenant IDs raises UnknownTenantError."""

    def test_queue_manager_raises_unknown_tenant_error(self):
        qm = TenantQueueManager()
        t1 = Tenant("t1", priority=1, weight=1.0, gpu_quota=100.0)
        qm.register_tenant(t1)

        job_unknown = GPUJob("j1", "unknown_tenant", "m", arrival_time=0.0, estimated_gpu_time=10.0)
        with pytest.raises(UnknownTenantError):
            qm.enqueue(job_unknown)
        with pytest.raises(UnknownTenantError):
            qm.dequeue("unknown_tenant")
        with pytest.raises(UnknownTenantError):
            qm.peek("unknown_tenant")
        with pytest.raises(UnknownTenantError):
            qm.get_queue("unknown_tenant")

    def test_fair_scheduler_raises_unknown_tenant_error(self):
        qm = TenantQueueManager()
        sched = GPUFairScheduler(tenants={}, queue_manager=qm)
        job_unknown = GPUJob("j1", "unregistered_tenant", "m", arrival_time=0.0, estimated_gpu_time=10.0)
        with pytest.raises(UnknownTenantError):
            sched.calculate_score(job_unknown, current_time_ms=0.0, v_bar=0.0)
