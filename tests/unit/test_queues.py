"""Unit tests for isolated per-tenant queues and queue manager."""

import pytest
from src.core.types import JobState
from src.core.job import Job
from src.core.tenant import Tenant
from src.core.queues import TenantJobQueue, TenantQueueManager


def test_tenant_job_queue_fifo_order():
    q = TenantJobQueue(tenant_id="t1", max_capacity=10)
    assert q.is_empty
    assert len(q) == 0

    j1 = Job("j1", "t1", "m", (3, 224, 224), 1, 0.0)
    j2 = Job("j2", "t1", "m", (3, 224, 224), 1, 5.0)

    assert q.enqueue(j1)
    assert q.enqueue(j2)
    assert len(q) == 2
    assert q.peek() == j1

    popped1 = q.dequeue()
    assert popped1 == j1
    popped2 = q.dequeue()
    assert popped2 == j2
    assert q.is_empty
    assert q.dequeue() is None


def test_queue_overflow_and_drop_policy():
    q = TenantJobQueue(tenant_id="t_capped", max_capacity=2)
    j1 = Job("j1", "t_capped", "m", (3, 224, 224), 1, 0.0)
    j2 = Job("j2", "t_capped", "m", (3, 224, 224), 1, 1.0)
    j3 = Job("j3", "t_capped", "m", (3, 224, 224), 1, 2.0)

    assert q.enqueue(j1) is True
    assert q.enqueue(j2) is True
    assert q.is_full

    # 3rd job must be dropped
    assert q.enqueue(j3) is False
    assert j3.state == JobState.DROPPED
    assert q.total_dropped == 1
    assert len(q) == 2


def test_cross_tenant_pollution_prevention():
    q = TenantJobQueue(tenant_id="tenant_alpha", max_capacity=10)
    j_wrong = Job("j_alien", "tenant_beta", "m", (3, 224, 224), 1, 0.0)

    with pytest.raises(ValueError, match="Cross-tenant pollution error"):
        q.enqueue(j_wrong)


def test_tenant_queue_manager_routing():
    qm = TenantQueueManager()
    t1 = Tenant("t1", "T1", 1.0, 5, 50.0, max_queue_capacity=5)
    t2 = Tenant("t2", "T2", 2.0, 8, 50.0, max_queue_capacity=5)

    qm.register_tenant(t1)
    qm.register_tenant(t2)

    assert qm.registered_tenant_ids == ["t1", "t2"]
    assert qm.get_active_tenant_ids() == []

    j1 = Job("j1", "t1", "m", (3, 224, 224), 1, 0.0)
    j2 = Job("j2", "t2", "m", (3, 224, 224), 1, 1.0)

    qm.enqueue(j1)
    qm.enqueue(j2)

    assert set(qm.get_active_tenant_ids()) == {"t1", "t2"}
    assert qm.total_queued_jobs == 2

    # Dequeue from t1
    popped = qm.dequeue("t1")
    assert popped == j1
    assert qm.get_active_tenant_ids() == ["t2"]

    # Enqueue to unregistered tenant raises KeyError
    j_unregistered = Job("ju", "t_unknown", "m", (3, 224, 224), 1, 0.0)
    with pytest.raises(KeyError):
        qm.enqueue(j_unregistered)
