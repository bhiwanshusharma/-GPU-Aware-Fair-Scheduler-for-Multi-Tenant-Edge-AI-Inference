"""Unit tests for Tenant data model, aging mechanics, and quota tracking."""

import pytest
from src.core.tenant import Tenant
from src.core.job import Job


def test_tenant_validation():
    t = Tenant(
        tenant_id="t1",
        name="Tenant One",
        weight=2.0,
        priority=5,
        gpu_quota_pct=30.0,
    )
    assert t.effective_priority == 5.0

    # Negative weight
    with pytest.raises(ValueError):
        Tenant(tenant_id="t_bad", name="Bad", weight=-1.0, priority=5, gpu_quota_pct=30.0)

    # Invalid priority
    with pytest.raises(ValueError):
        Tenant(tenant_id="t_bad", name="Bad", weight=1.0, priority=11, gpu_quota_pct=30.0)

    # Invalid quota percentage
    with pytest.raises(ValueError):
        Tenant(tenant_id="t_bad", name="Bad", weight=1.0, priority=5, gpu_quota_pct=150.0)


def test_tenant_aging_mechanics():
    t = Tenant(
        tenant_id="t_aging",
        name="Aging Tenant",
        weight=1.0,
        priority=3,
        gpu_quota_pct=25.0,
    )
    assert t.effective_priority == 3.0

    # At t=0, no aging
    p0 = t.apply_aging(current_time_ms=0.0, alpha=0.1, starvation_threshold_ms=100.0)
    assert p0 == 3.0

    # At t=200ms without service, effective priority must increase
    p1 = t.apply_aging(current_time_ms=200.0, alpha=0.1, starvation_threshold_ms=100.0)
    assert p1 == pytest.approx(3.0 + 0.1 * (200.0 / 100.0))  # 3.2
    assert p1 > 3.0

    # After service at t=200ms, time_since_service resets
    t.record_gpu_execution(gpu_time_ms=20.0, current_time_ms=200.0)
    p2 = t.apply_aging(current_time_ms=200.0, alpha=0.1, starvation_threshold_ms=100.0)
    assert p2 == 3.0


def test_tenant_quota_tracking_and_window_slide():
    t = Tenant(
        tenant_id="t_quota",
        name="Quota Tenant",
        weight=1.0,
        priority=5,
        gpu_quota_pct=20.0,       # 20% quota
        quota_window_ms=1000.0,   # Max 200ms per 1000ms window
    )
    assert not t.is_quota_exceeded(current_time_ms=100.0)

    # Consume 150ms
    t.record_gpu_execution(gpu_time_ms=150.0, current_time_ms=100.0)
    assert not t.is_quota_exceeded(current_time_ms=200.0)

    # Consume another 60ms (total 210ms > 200ms quota)
    t.record_gpu_execution(gpu_time_ms=60.0, current_time_ms=300.0)
    assert t.is_quota_exceeded(current_time_ms=300.0)

    # Slide window past 1000ms -> Quota must refresh
    assert not t.is_quota_exceeded(current_time_ms=1200.0)


def test_deficit_counter_arithmetic():
    t = Tenant(
        tenant_id="t_drr",
        name="DRR Tenant",
        weight=2.0,
        priority=5,
        gpu_quota_pct=50.0,
    )
    assert t.deficit_counter_ms == 0.0

    # Replenish with quantum 25ms
    t.replenish_deficit(25.0)
    assert t.deficit_counter_ms == 25.0

    # Deduct 18ms
    t.deduct_deficit(18.0)
    assert t.deficit_counter_ms == pytest.approx(7.0)


def test_tenant_completion_telemetry():
    t = Tenant(
        tenant_id="t_telemetry",
        name="Telemetry Tenant",
        weight=1.0,
        priority=5,
        gpu_quota_pct=50.0,
    )
    job = Job(
        job_id="j1",
        tenant_id="t_telemetry",
        model_name="yolo",
        input_shape=(3, 640, 640),
        batch_size=1,
        arrival_time_ms=0.0,
        deadline_ms=20.0,
    )
    job.mark_started(10.0)
    job.mark_completed(25.0, actual_gpu_time_ms=15.0)  # Total 25ms > 20ms deadline -> SLA violated

    t.record_job_completed(job)
    assert t.total_jobs_completed == 1
    assert t.total_waiting_time_ms == 10.0
    assert t.total_execution_time_ms == 15.0
    assert t.total_sla_violations == 1
    assert t.sla_violation_rate == 1.0
    assert t.mean_waiting_time_ms == 10.0

    t.reset_dynamic_state()
    assert t.total_jobs_completed == 0
    assert t.deficit_counter_ms == 0.0
