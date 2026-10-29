"""Unit tests for the event-driven GPU simulation engine."""

import pytest
from src.core.types import WorkloadClass, JobState
from src.core.job import GPUJob
from src.core.tenant import Tenant
from src.simulation.simulator import GPUSimulator
from src.simulation.trace import TraceEventType


# ============================================================================
# Test 1: One Tenant
# ============================================================================

def test_simulator_one_tenant_single_job():
    sim = GPUSimulator(quantum_ms=10.0, context_switch_overhead_ms=0.0)
    job = GPUJob(
        job_id="job_t1_1",
        tenant_id="tenant_A",
        model_name="yolo",
        arrival_time=0.0,
        estimated_gpu_time=8.0,
        remaining_gpu_time=8.0,
    )
    sim.submit_job(job)
    sim.run()

    assert len(sim.completed_jobs) == 1
    assert job.status == JobState.COMPLETED
    assert job.start_time == 0.0
    assert job.completion_time == 8.0
    assert job.number_of_preemptions == 0
    assert sim.gpu_utilization_pct == 100.0


# ============================================================================
# Test 2: Two Tenants (Trace formatting check)
# ============================================================================

def test_simulator_two_tenants_trace_formatting():
    # Both jobs take 15ms with quantum=10ms
    # Tenant A job A1: 15ms -> slice 1 (10ms), yields, slice 2 (5ms)
    # Tenant B job B1: 15ms -> slice 1 (10ms), yields, slice 2 (5ms)
    sim = GPUSimulator(quantum_ms=10.0, context_switch_overhead_ms=0.0, policy="round_robin")
    
    t_a = Tenant("A", priority=5, weight=1.0, gpu_quota=50.0)
    t_b = Tenant("B", priority=5, weight=1.0, gpu_quota=50.0)
    sim.register_tenant(t_a)
    sim.register_tenant(t_b)

    job_a1 = GPUJob("A1", "A", "m", arrival_time=0.0, estimated_gpu_time=15.0, remaining_gpu_time=15.0)
    job_b1 = GPUJob("B1", "B", "m", arrival_time=0.0, estimated_gpu_time=15.0, remaining_gpu_time=15.0)

    sim.submit_job(job_a1)
    sim.submit_job(job_b1)
    sim.run()

    assert len(sim.completed_jobs) == 2
    assert job_a1.status == JobState.COMPLETED
    assert job_b1.status == JobState.COMPLETED
    assert job_a1.number_of_preemptions == 1
    assert job_b1.number_of_preemptions == 1

    trace_text = sim.get_execution_trace_text()
    # Check key trace lines
    assert "tenant=A job=A1 dispatched" in trace_text
    assert "A1 quantum expired" in trace_text
    assert "tenant=B job=B1 dispatched" in trace_text
    assert "B1 quantum expired" in trace_text
    assert "tenant=A job=A1 resumed" in trace_text


# ============================================================================
# Test 3: Multiple Tenants (3+ Tenants sharing GPU)
# ============================================================================

def test_simulator_multiple_tenants():
    sim = GPUSimulator(quantum_ms=10.0, context_switch_overhead_ms=0.0, policy="round_robin")
    
    for t_id in ["T1", "T2", "T3"]:
        sim.register_tenant(Tenant(t_id, priority=5, weight=1.0, gpu_quota=33.3))

    j1 = GPUJob("j1", "T1", "m", 0.0, 10.0, 10.0)
    j2 = GPUJob("j2", "T2", "m", 0.0, 10.0, 10.0)
    j3 = GPUJob("j3", "T3", "m", 0.0, 10.0, 10.0)

    sim.submit_job(j1)
    sim.submit_job(j2)
    sim.submit_job(j3)
    sim.run()

    assert len(sim.completed_jobs) == 3
    assert sim.current_time_ms == 30.0
    assert sim.gpu_utilization_pct == 100.0


# ============================================================================
# Test 4: Empty Queue
# ============================================================================

def test_simulator_empty_queue():
    sim = GPUSimulator(quantum_ms=10.0)
    # Run with zero jobs
    sim.run(until_time_ms=50.0)

    assert len(sim.completed_jobs) == 0
    assert sim.gpu.is_idle
    assert sim.gpu_utilization_pct == 0.0


# ============================================================================
# Test 5: Job Shorter than Quantum
# ============================================================================

def test_simulator_job_shorter_than_quantum():
    # Quantum is 20ms, Job is 6ms
    sim = GPUSimulator(quantum_ms=20.0, context_switch_overhead_ms=0.0)
    job = GPUJob("j_short", "t_single", "m", arrival_time=0.0, estimated_gpu_time=6.0, remaining_gpu_time=6.0)
    sim.submit_job(job)
    sim.run()

    assert job.status == JobState.COMPLETED
    assert job.completion_time == 6.0
    assert job.number_of_preemptions == 0
    assert len(sim.trace.preemption_events) == 0


# ============================================================================
# Test 6: Job Longer than Quantum
# ============================================================================

def test_simulator_job_longer_than_quantum():
    # Quantum is 10ms, Job is 25ms
    sim = GPUSimulator(quantum_ms=10.0, context_switch_overhead_ms=0.0)
    job = GPUJob("j_long", "t_long", "m", arrival_time=0.0, estimated_gpu_time=25.0, remaining_gpu_time=25.0)
    sim.submit_job(job)
    sim.run()

    assert job.status == JobState.COMPLETED
    assert job.completion_time == 25.0
    # Preempted at 10ms and 20ms -> exactly 2 preemptions
    assert job.number_of_preemptions == 2
    assert len(sim.trace.preemption_events) == 2


# ============================================================================
# Test 7: Jobs Arriving During Execution
# ============================================================================

def test_simulator_job_arriving_during_execution():
    sim = GPUSimulator(quantum_ms=10.0, context_switch_overhead_ms=0.0)
    
    # J1 arrives at t=0, executes 10ms slice from t=0 to t=10
    j1 = GPUJob("j1", "t1", "m", arrival_time=0.0, estimated_gpu_time=10.0, remaining_gpu_time=10.0)
    # J2 arrives at t=4 while J1 is computing
    j2 = GPUJob("j2", "t2", "m", arrival_time=4.0, estimated_gpu_time=5.0, remaining_gpu_time=5.0)

    sim.submit_job(j1)
    sim.submit_job(j2)
    sim.run()

    assert len(sim.completed_jobs) == 2
    assert j1.completion_time == 10.0
    assert j2.start_time == 10.0
    assert j2.completion_time == 15.0
    assert j2.waiting_time_ms == 6.0  # Arrived at 4, started at 10


# ============================================================================
# Test 8: Multiple Jobs From One Tenant
# ============================================================================

def test_simulator_multiple_jobs_from_one_tenant():
    sim = GPUSimulator(quantum_ms=10.0, context_switch_overhead_ms=0.0)
    t = Tenant("t_fifo", priority=5, weight=1.0, gpu_quota=100.0)
    sim.register_tenant(t)

    j1 = GPUJob("job_1", "t_fifo", "m", arrival_time=0.0, estimated_gpu_time=5.0, remaining_gpu_time=5.0)
    j2 = GPUJob("job_2", "t_fifo", "m", arrival_time=1.0, estimated_gpu_time=8.0, remaining_gpu_time=8.0)

    sim.submit_job(j1)
    sim.submit_job(j2)
    sim.run()

    assert sim.completed_jobs[0].job_id == "job_1"
    assert sim.completed_jobs[1].job_id == "job_2"
    assert j1.completion_time == 5.0
    assert j2.start_time == 5.0
    assert j2.completion_time == 13.0


# ============================================================================
# Test 9: Different Priorities
# ============================================================================

def test_simulator_different_priorities():
    sim = GPUSimulator(quantum_ms=10.0, context_switch_overhead_ms=0.0, policy="priority")
    
    t_low = Tenant("t_low", priority=2, weight=1.0, gpu_quota=50.0)
    t_high = Tenant("t_high", priority=9, weight=1.0, gpu_quota=50.0)
    sim.register_tenant(t_low)
    sim.register_tenant(t_high)

    # Both arrive at t=0
    j_low = GPUJob("j_low", "t_low", "m", arrival_time=0.0, estimated_gpu_time=10.0, remaining_gpu_time=10.0)
    j_high = GPUJob("j_high", "t_high", "m", arrival_time=0.0, estimated_gpu_time=10.0, remaining_gpu_time=10.0)

    sim.submit_job(j_low)
    sim.submit_job(j_high)
    sim.run()

    # High priority tenant job must complete first!
    assert sim.completed_jobs[0].job_id == "j_high"
    assert sim.completed_jobs[1].job_id == "j_low"
    assert j_high.completion_time == 10.0
    assert j_low.completion_time == 20.0


# ============================================================================
# Test 10: Zero or Invalid Execution Time Rejection
# ============================================================================

def test_simulator_zero_or_negative_execution_time_rejected():
    sim = GPUSimulator(quantum_ms=10.0)

    # Job with zero estimated time
    j_zero = GPUJob("j_zero", "t1", "m", arrival_time=0.0, estimated_gpu_time=0.0, remaining_gpu_time=0.0)
    with pytest.raises(ValueError, match="positive execution time"):
        sim.submit_job(j_zero)

    # Job with negative remaining time
    j_neg = GPUJob("j_neg", "t1", "m", arrival_time=0.0, estimated_gpu_time=10.0, remaining_gpu_time=10.0)
    j_neg.remaining_gpu_time = -5.0
    with pytest.raises(ValueError, match="positive execution time"):
        sim.submit_job(j_neg)


# ============================================================================
# Additional Verification: Context Switch Overhead & Idle Accounting
# ============================================================================

def test_simulator_context_switch_overhead():
    # Context switch overhead = 1.0ms
    sim = GPUSimulator(quantum_ms=10.0, context_switch_overhead_ms=1.0)
    
    j1 = GPUJob("j1", "A", "m", 0.0, 5.0, 5.0)
    j2 = GPUJob("j2", "B", "m", 0.0, 5.0, 5.0)

    sim.submit_job(j1)
    sim.submit_job(j2)
    sim.run()

    # j1 runs from 0 to 5 (switch overhead applies if switching, here first job runs at 0)
    # j2 switches from A -> overhead of 1.0ms added!
    # Total time = 5.0 (j1) + 1.0 (cs) + 5.0 (j2) = 11.0ms
    assert sim.current_time_ms == 11.0
    assert sim.gpu.total_context_switches >= 1
    assert sim.gpu.total_switching_time_ms == 1.0


def test_simulator_idle_periods_and_queue_snapshots():
    sim = GPUSimulator(quantum_ms=10.0)
    # Job 1 arrives at 0, runs for 10ms (0 to 10)
    j1 = GPUJob("j1", "t1", "m", 0.0, 10.0, 10.0)
    # Job 2 arrives at 30 (GPU idle from 10 to 30)
    j2 = GPUJob("j2", "t1", "m", 30.0, 10.0, 10.0)

    sim.submit_job(j1)
    sim.submit_job(j2)
    sim.run()

    idle_periods = sim.get_idle_periods()
    assert len(idle_periods) >= 1
    # Check that idle period duration is 20ms (from 10 to 30)
    assert any(dur == pytest.approx(20.0) for _, _, dur in idle_periods)
    assert len(sim.get_queue_states()) >= 2
