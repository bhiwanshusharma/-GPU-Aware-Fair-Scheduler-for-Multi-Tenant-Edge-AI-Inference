"""Comprehensive unit tests for the Workload, Tenant, and GPUJob data models."""

import pytest
from src.core.types import WorkloadClass, JobState, PolicyType, SchedulingQuantum
from src.core.job import GPUJob, Job
from src.core.tenant import Tenant, WaitingStatistics
from src.core.workload import Workload
from src.core.execution_record import GPUExecutionRecord
from src.core.decision import SchedulingDecision
from src.core.scheduler_config import SchedulerConfiguration, SchedulerConfig


# ============================================================================
# 1. GPUJob Data Model Tests
# ============================================================================

def test_gpujob_all_required_fields_and_defaults():
    job = GPUJob(
        job_id="job_full_001",
        tenant_id="tenant_alpha",
        model_name="yolov8n",
        arrival_time=100.0,
        estimated_gpu_time=25.0,
        remaining_gpu_time=25.0,
        priority=8,
        deadline=180.0,
        workload_type=WorkloadClass.GPU_INTENSIVE,
        batch_size=2,
    )
    # Check all requested fields
    assert job.job_id == "job_full_001"
    assert job.tenant_id == "tenant_alpha"
    assert job.arrival_time == 100.0
    assert job.estimated_gpu_time == 25.0
    assert job.remaining_gpu_time == 25.0
    assert job.priority == 8
    assert job.deadline == 180.0
    assert job.workload_type == WorkloadClass.GPU_INTENSIVE
    assert job.batch_size == 2
    assert job.model_name == "yolov8n"
    assert job.status == JobState.QUEUED
    assert job.start_time is None
    assert job.completion_time is None
    assert job.accumulated_waiting_time == 0.0
    assert job.accumulated_execution_time == 0.0
    assert job.number_of_preemptions == 0


def test_gpujob_validation_errors():
    # Empty job_id
    with pytest.raises(ValueError, match="job_id"):
        GPUJob(job_id="", tenant_id="t1", model_name="m1", arrival_time=0.0)

    # Empty tenant_id
    with pytest.raises(ValueError, match="tenant_id"):
        GPUJob(job_id="j1", tenant_id="   ", model_name="m1", arrival_time=0.0)

    # Empty model_name
    with pytest.raises(ValueError, match="model_name"):
        GPUJob(job_id="j1", tenant_id="t1", model_name="", arrival_time=0.0)

    # Negative arrival_time
    with pytest.raises(ValueError, match="arrival_time"):
        GPUJob(job_id="j1", tenant_id="t1", model_name="m1", arrival_time=-1.0)

    # Negative estimated_gpu_time
    with pytest.raises(ValueError, match="estimated_gpu_time"):
        GPUJob(job_id="j1", tenant_id="t1", model_name="m1", arrival_time=0.0, estimated_gpu_time=-5.0)

    # Negative remaining_gpu_time
    with pytest.raises(ValueError, match="remaining_gpu_time"):
        GPUJob(job_id="j1", tenant_id="t1", model_name="m1", arrival_time=0.0, remaining_gpu_time=-2.0)

    # Invalid priority
    with pytest.raises(ValueError, match="priority"):
        GPUJob(job_id="j1", tenant_id="t1", model_name="m1", arrival_time=0.0, priority=11)
    with pytest.raises(ValueError, match="priority"):
        GPUJob(job_id="j1", tenant_id="t1", model_name="m1", arrival_time=0.0, priority=0)

    # Deadline earlier than arrival_time
    with pytest.raises(ValueError, match="deadline"):
        GPUJob(job_id="j1", tenant_id="t1", model_name="m1", arrival_time=50.0, deadline=40.0)

    # Invalid batch_size
    with pytest.raises(ValueError, match="batch_size"):
        GPUJob(job_id="j1", tenant_id="t1", model_name="m1", arrival_time=0.0, batch_size=0)

    # Invalid workload_type
    with pytest.raises(ValueError, match="workload_type"):
        GPUJob(job_id="j1", tenant_id="t1", model_name="m1", arrival_time=0.0, workload_type="INVALID_TYPE")


def test_gpujob_multi_slice_execution_and_preemption():
    job = GPUJob(
        job_id="job_slice_001",
        tenant_id="tenant_beta",
        model_name="transformer",
        arrival_time=0.0,
        estimated_gpu_time=30.0,
        remaining_gpu_time=30.0,
    )
    # Execute slice 1: 10ms at current_time=5.0 -> Not finished, should preempt
    finished = job.execute_slice(actual_gpu_time=10.0, current_time=5.0)
    assert not finished
    assert job.status == JobState.PREEMPTED
    assert job.number_of_preemptions == 1
    assert job.remaining_gpu_time == 20.0
    assert job.accumulated_execution_time == 10.0
    assert job.accumulated_waiting_time == 5.0

    # Resume slice 2: 15ms at current_time=25.0 -> Not finished, preempt again
    job.transition_to(JobState.RUNNING)
    finished2 = job.execute_slice(actual_gpu_time=15.0, current_time=25.0)
    assert not finished2
    assert job.status == JobState.PREEMPTED
    assert job.number_of_preemptions == 2
    assert job.remaining_gpu_time == 5.0
    assert job.accumulated_execution_time == 25.0

    # Final slice 3: 5ms at current_time=50.0 -> Finished!
    job.transition_to(JobState.RUNNING)
    finished3 = job.execute_slice(actual_gpu_time=5.0, current_time=50.0)
    assert finished3
    assert job.status == JobState.COMPLETED
    assert job.remaining_gpu_time == 0.0
    assert job.accumulated_execution_time == 30.0
    assert job.completion_time == 55.0


# ============================================================================
# 2. Tenant Data Model Tests
# ============================================================================

def test_tenant_all_required_fields():
    stats = WaitingStatistics()
    tenant = Tenant(
        tenant_id="tenant_robotics_x",
        priority=10,
        weight=4.5,
        gpu_quota=40.0,
        max_concurrent_jobs=8,
        accumulated_gpu_service=120.0,
        waiting_statistics=stats,
    )
    assert tenant.tenant_id == "tenant_robotics_x"
    assert tenant.priority == 10
    assert tenant.weight == 4.5
    assert tenant.gpu_quota == 40.0
    assert tenant.max_concurrent_jobs == 8
    assert tenant.jobs == []
    assert tenant.accumulated_gpu_service == 120.0
    assert tenant.waiting_statistics is stats


def test_tenant_validation_errors():
    # Empty tenant_id
    with pytest.raises(ValueError, match="tenant_id"):
        Tenant(tenant_id="", priority=5, weight=1.0, gpu_quota=50.0)

    # Invalid priority
    with pytest.raises(ValueError, match="priority"):
        Tenant(tenant_id="t1", priority=0, weight=1.0, gpu_quota=50.0)
    with pytest.raises(ValueError, match="priority"):
        Tenant(tenant_id="t1", priority=11, weight=1.0, gpu_quota=50.0)

    # Invalid weight
    with pytest.raises(ValueError, match="weight"):
        Tenant(tenant_id="t1", priority=5, weight=0.0, gpu_quota=50.0)
    with pytest.raises(ValueError, match="weight"):
        Tenant(tenant_id="t1", priority=5, weight=-2.0, gpu_quota=50.0)

    # Invalid GPU quota
    with pytest.raises(ValueError, match="gpu_quota"):
        Tenant(tenant_id="t1", priority=5, weight=1.0, gpu_quota=0.0)
    with pytest.raises(ValueError, match="gpu_quota"):
        Tenant(tenant_id="t1", priority=5, weight=1.0, gpu_quota=105.0)

    # Invalid max_concurrent_jobs
    with pytest.raises(ValueError, match="max_concurrent_jobs"):
        Tenant(tenant_id="t1", priority=5, weight=1.0, gpu_quota=50.0, max_concurrent_jobs=0)


def test_tenant_job_concurrency_and_management():
    tenant = Tenant(
        tenant_id="t_conc",
        priority=6,
        weight=2.0,
        gpu_quota=30.0,
        max_concurrent_jobs=2,
    )
    assert tenant.can_admit_concurrent_job()
    assert tenant.active_jobs_count == 0

    j1 = GPUJob("j1", "t_conc", "m", 0.0, 10.0, 10.0)
    j2 = GPUJob("j2", "t_conc", "m", 1.0, 10.0, 10.0)
    j3 = GPUJob("j3", "t_conc", "m", 2.0, 10.0, 10.0)

    tenant.add_job(j1)
    tenant.add_job(j2)
    assert tenant.active_jobs_count == 2
    assert not tenant.can_admit_concurrent_job()

    # Reject foreign tenant job
    j_foreign = GPUJob("j_foreign", "other_tenant", "m", 0.0, 10.0, 10.0)
    with pytest.raises(ValueError, match="Job tenant"):
        tenant.add_job(j_foreign)

    # Complete j1
    j1.mark_started(10.0)
    j1.mark_completed(20.0, 10.0)
    tenant.record_job_completed(j1)

    assert tenant.active_jobs_count == 1
    assert tenant.can_admit_concurrent_job()
    assert tenant.waiting_statistics.jobs_count == 1
    assert tenant.waiting_statistics.mean_waiting_time_ms == 10.0


def test_waiting_statistics():
    ws = WaitingStatistics()
    assert ws.jobs_count == 0
    assert ws.mean_waiting_time_ms == 0.0

    ws.record_wait(10.0)
    ws.record_wait(30.0)
    ws.record_wait(20.0)

    assert ws.jobs_count == 3
    assert ws.total_waiting_time_ms == 60.0
    assert ws.mean_waiting_time_ms == 20.0
    assert ws.min_waiting_time_ms == 10.0
    assert ws.max_waiting_time_ms == 30.0

    with pytest.raises(ValueError):
        ws.record_wait(-1.0)

    ws.reset()
    assert ws.jobs_count == 0
    assert ws.total_waiting_time_ms == 0.0


# ============================================================================
# 3. Workload Data Model Tests
# ============================================================================

def test_workload_creation_and_latency_estimation():
    wl = Workload(
        model_name="mobilenet_v3",
        domain="VISION",
        workload_type=WorkloadClass.BALANCED,
        input_shape=(3, 224, 224),
        default_batch_size=1,
        max_batch_size=8,
        calibrated_base_latency_ms=6.0,
        latency_per_batch_item_ms=2.0,
        memory_footprint_mb=20.0,
    )
    assert wl.model_name == "mobilenet_v3"
    assert wl.estimate_execution_time(batch_size=1) == 6.0
    assert wl.estimate_execution_time(batch_size=4) == 6.0 + 3 * 2.0  # 12.0ms
    assert wl.estimate_memory_mb(batch_size=1) == 20.0

    # Batch size validation
    with pytest.raises(ValueError):
        wl.estimate_execution_time(batch_size=0)
    with pytest.raises(ValueError):
        wl.estimate_execution_time(batch_size=9)  # > max_batch_size


def test_workload_factory_create_job():
    wl = Workload(
        model_name="yolov8n",
        domain="DETECTION",
        workload_type=WorkloadClass.GPU_INTENSIVE,
        input_shape=(3, 640, 640),
        calibrated_base_latency_ms=20.0,
        latency_per_batch_item_ms=5.0,
    )
    job = wl.create_job(
        job_id="fact_001",
        tenant_id="tenant_cam",
        arrival_time=15.0,
        batch_size=2,
        priority=7,
        deadline=100.0,
    )
    assert job.job_id == "fact_001"
    assert job.tenant_id == "tenant_cam"
    assert job.model_name == "yolov8n"
    assert job.arrival_time == 15.0
    assert job.estimated_gpu_time == 25.0
    assert job.remaining_gpu_time == 25.0
    assert job.priority == 7
    assert job.deadline == 100.0
    assert job.workload_type == WorkloadClass.GPU_INTENSIVE


# ============================================================================
# 4. GPUExecutionRecord Data Model Tests
# ============================================================================

def test_gpu_execution_record():
    rec = GPUExecutionRecord(
        record_id="rec_001",
        job_id="j_rec_1",
        tenant_id="t1",
        model_name="yolo",
        start_time=10.0,
        end_time=25.0,
        allocated_quantum=20.0,
        actual_execution_time=15.0,
        remaining_gpu_time_after=0.0,
        was_preempted=False,
        is_completed=True,
        vram_allocated_mb=32.0,
    )
    assert rec.elapsed_wall_time == 15.0
    assert rec.actual_execution_time == 15.0
    assert not rec.was_preempted
    assert rec.is_completed

    # Validation: end_time < start_time
    with pytest.raises(ValueError, match="end_time"):
        GPUExecutionRecord(
            record_id="r_bad",
            job_id="j1",
            tenant_id="t1",
            model_name="m",
            start_time=30.0,
            end_time=20.0,
            allocated_quantum=10.0,
            actual_execution_time=5.0,
            remaining_gpu_time_after=0.0,
            was_preempted=False,
            is_completed=True,
        )


# ============================================================================
# 5. SchedulingDecision Data Model Tests
# ============================================================================

def test_scheduling_decision():
    job = GPUJob("j_dec", "t_dec", "model", 0.0, 10.0, 10.0)
    decision = SchedulingDecision(
        decision_id="dec_001",
        timestamp=50.0,
        selected_job=job,
        selected_tenant_id="t_dec",
        allocated_quantum_ms=25.0,
        reason="PDWFQ_DEFICIT_MATCH",
        active_tenants_count=3,
        deficit_snapshot={"t_dec": 25.0},
    )
    assert not decision.is_idle
    assert decision.selected_job == job
    assert decision.allocated_quantum_ms == 25.0

    # Idle decision
    idle_dec = SchedulingDecision(
        decision_id="dec_idle",
        timestamp=100.0,
        selected_job=None,
        selected_tenant_id=None,
        allocated_quantum_ms=0.0,
        reason="NO_ELIGIBLE_JOBS",
    )
    assert idle_dec.is_idle

    # Mismatch between selected_job and selected_tenant_id
    with pytest.raises(ValueError, match="Tenant mismatch"):
        SchedulingDecision(
            decision_id="dec_err",
            timestamp=10.0,
            selected_job=job,
            selected_tenant_id="wrong_tenant",
            allocated_quantum_ms=10.0,
            reason="ERROR",
        )


# ============================================================================
# 6. SchedulerConfiguration Data Model Tests
# ============================================================================

def test_scheduler_configuration():
    config = SchedulerConfiguration(
        policy=PolicyType.PDWFQ,
        base_quantum_ms=30.0,
        min_quantum_ms=10.0,
        max_quantum_ms=100.0,
        aging_alpha=0.2,
        starvation_threshold_ms=200.0,
    )
    assert config.policy == PolicyType.PDWFQ
    assert config.base_quantum_ms == 30.0

    sq = config.to_scheduling_quantum()
    assert isinstance(sq, SchedulingQuantum)
    assert sq.base_quantum_ms == 30.0

    # From dictionary
    raw_dict = {
        "policy": "round_robin",
        "base_quantum_ms": 20.0,
        "min_quantum_ms": 5.0,
        "max_quantum_ms": 50.0,
    }
    loaded = SchedulerConfiguration.from_dict(raw_dict)
    assert loaded.policy == PolicyType.ROUND_ROBIN
    assert loaded.base_quantum_ms == 20.0

    # Validation: min > base
    with pytest.raises(ValueError, match="min_quantum_ms"):
        SchedulerConfiguration(base_quantum_ms=10.0, min_quantum_ms=20.0)

    # Validation: max < base
    with pytest.raises(ValueError, match="max_quantum_ms"):
        SchedulerConfiguration(base_quantum_ms=50.0, max_quantum_ms=30.0)
