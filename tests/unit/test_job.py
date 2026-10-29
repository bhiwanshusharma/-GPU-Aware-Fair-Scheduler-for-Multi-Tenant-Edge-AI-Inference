"""Unit tests for Job data model and lifecycle transitions."""

import pytest
from src.core.types import JobState, WorkloadClass
from src.core.job import Job


def test_job_initialization_and_validation():
    job = Job(
        job_id="job_001",
        tenant_id="tenant_a",
        model_name="yolov8n",
        input_shape=(3, 640, 640),
        batch_size=1,
        arrival_time_ms=10.0,
    )
    assert job.job_id == "job_001"
    assert job.state == JobState.QUEUED
    assert job.waiting_time_ms == 0.0

    # Negative arrival time
    with pytest.raises(ValueError):
        Job(
            job_id="job_bad",
            tenant_id="tenant_a",
            model_name="yolov8n",
            input_shape=(3, 640, 640),
            batch_size=1,
            arrival_time_ms=-5.0,
        )

    # Non-positive batch size
    with pytest.raises(ValueError):
        Job(
            job_id="job_bad",
            tenant_id="tenant_a",
            model_name="yolov8n",
            input_shape=(3, 640, 640),
            batch_size=0,
            arrival_time_ms=0.0,
        )


def test_job_lifecycle_state_transitions():
    job = Job(
        job_id="job_002",
        tenant_id="tenant_b",
        model_name="mobilenetv3",
        input_shape=(3, 224, 224),
        batch_size=2,
        arrival_time_ms=20.0,
        deadline_ms=100.0,
    )
    assert job.state == JobState.QUEUED

    # Transition to PREDICTED
    job.transition_to(JobState.PREDICTED)
    assert job.state == JobState.PREDICTED

    # Transition to DISPATCHED
    job.transition_to(JobState.DISPATCHED)
    assert job.state == JobState.DISPATCHED

    # Mark started -> RUNNING
    job.mark_started(start_time_ms=35.0)
    assert job.state == JobState.RUNNING
    assert job.waiting_time_ms == 15.0

    # Mark completed -> COMPLETED
    job.mark_completed(completion_time_ms=45.0, actual_gpu_time_ms=10.0)
    assert job.state == JobState.COMPLETED
    assert job.turnaround_time_ms == 25.0
    assert not job.is_sla_violated

    # Generate telemetry report
    report = job.to_completion_report()
    assert report.job_id == "job_002"
    assert report.waiting_time_ms == 15.0
    assert report.actual_gpu_time_ms == 10.0
    assert report.turnaround_time_ms == 25.0
    assert not report.sla_violated


def test_job_sla_violation():
    job = Job(
        job_id="job_003",
        tenant_id="tenant_c",
        model_name="resnet18",
        input_shape=(3, 224, 224),
        batch_size=1,
        arrival_time_ms=0.0,
        deadline_ms=50.0,
    )
    job.mark_started(start_time_ms=40.0)
    job.mark_completed(completion_time_ms=65.0, actual_gpu_time_ms=25.0)
    assert job.is_sla_violated
    report = job.to_completion_report()
    assert report.sla_violated


def test_invalid_state_transitions():
    job = Job(
        job_id="job_004",
        tenant_id="tenant_d",
        model_name="audio_kws",
        input_shape=(1, 16000),
        batch_size=1,
        arrival_time_ms=0.0,
    )
    # Cannot jump from QUEUED directly to COMPLETED
    with pytest.raises(ValueError):
        job.transition_to(JobState.COMPLETED)
