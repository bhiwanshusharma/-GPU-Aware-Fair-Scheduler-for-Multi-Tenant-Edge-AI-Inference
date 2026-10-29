"""OS Scheduler Scalability Evaluation and Profiling Suite.

Evaluates the multi-tenant GPU scheduler as an OS scheduling subsystem across:
- 10 tenants
- 25 tenants
- 50 tenants
- 100 tenants
- 500 tenants

Generates thousands of synthetic GPU inference requests across diverse arrival
patterns, measuring:
1. Decision latency (microsecond distribution: mean, p50, p95, p99)
2. Simulation throughput (jobs/sec, decisions/sec)
3. Memory footprint (peak RAM delta via tracemalloc in MB)
4. Scheduling decisions count
5. Queue operations count (enqueues, dequeues, peeks)
6. Scheduling overhead
7. Detailed function call profiling via cProfile
"""

import sys
import os
import time
import random
import tracemalloc
import cProfile
import pstats
import io
import math
from typing import Dict, List, Tuple, Any

from src.core.types import SchedulingQuantum, WorkloadClass
from src.core.job import GPUJob
from src.core.tenant import Tenant
from src.core.queues import TenantQueueManager
from src.scheduler.fair_scheduler import GPUFairScheduler
from src.simulation.simulator import GPUSimulator
from src.telemetry.metrics_engine import MetricsEngine


def generate_benchmark_workload(
    tenants: Dict[str, Tenant],
    jobs_per_tenant: int = 10,
    seed: int = 42,
) -> List[GPUJob]:
    """Generates synthetic multi-tenant inference workloads."""
    rng = random.Random(seed)
    jobs: List[GPUJob] = []
    tenant_ids = list(tenants.keys())
    job_counter = 0

    for t_id in tenant_ids:
        tenant = tenants[t_id]
        curr_time = 0.0
        for _ in range(jobs_per_tenant):
            job_counter += 1
            # Arrival jitter
            curr_time += rng.expovariate(1.0 / 15.0)  # Poisson arrivals, mean 15ms interval
            # Execution duration
            if tenant.priority >= 8:
                dur = rng.uniform(5.0, 15.0)  # Short interactive
                w_type = WorkloadClass.CPU_BOUND if rng.random() < 0.3 else WorkloadClass.BALANCED
            elif tenant.priority <= 3:
                dur = rng.uniform(25.0, 60.0)  # Heavy batch
                w_type = WorkloadClass.GPU_INTENSIVE
            else:
                dur = rng.uniform(10.0, 30.0)  # Standard
                w_type = WorkloadClass.BALANCED

            job = GPUJob(
                job_id=f"job_{t_id}_{job_counter}",
                tenant_id=t_id,
                model_name=f"model_{t_id}",
                arrival_time=curr_time,
                estimated_gpu_time=dur,
                remaining_gpu_time=dur,
                priority=tenant.priority,
                workload_type=w_type,
            )
            jobs.append(job)

    # Sort globally by arrival timestamp
    jobs.sort(key=lambda j: j.arrival_time)
    return jobs


def run_scaling_benchmark(
    num_tenants: int,
    jobs_per_tenant: int = 10,
    profile_top_n: int = 0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs a complete OS evaluation benchmark for a given tenant count."""
    rng = random.Random(seed)

    # 1. Tenant Creation
    tenants = {}
    for i in range(num_tenants):
        t_id = f"tenant_{i:04d}"
        weight = rng.choice([1.0, 2.0, 3.0, 5.0, 8.0])
        priority = rng.randint(1, 10)
        quota = rng.choice([25.0, 50.0, 75.0, 100.0])
        t = Tenant(
            tenant_id=t_id,
            name=f"Tenant {i}",
            weight=weight,
            priority=priority,
            gpu_quota=quota,
            max_queue_capacity=500,
        )
        tenants[t_id] = t

    total_jobs_target = num_tenants * jobs_per_tenant
    workload = generate_benchmark_workload(tenants, jobs_per_tenant=jobs_per_tenant, seed=seed)

    # 2. Setup Scheduler and Simulator
    qm = TenantQueueManager()
    quantum = SchedulingQuantum(base_quantum_ms=10.0, min_quantum_ms=2.0, max_quantum_ms=25.0)
    scheduler = GPUFairScheduler(
        tenants=tenants,
        queue_manager=qm,
        quantum=quantum,
        starvation_threshold_ms=100.0,
    )
    sim = GPUSimulator(
        tenants=tenants,
        quantum_ms=10.0,
        context_switch_overhead_ms=0.5,
        scheduler=scheduler,
    )

    for job in workload:
        sim.submit_job(job)

    # Instrument scheduler decision latency
    decision_latencies_us: List[float] = []
    original_schedule = scheduler.schedule

    def instrumented_schedule(current_time_ms: float):
        t0 = time.perf_counter()
        result = original_schedule(current_time_ms)
        t1 = time.perf_counter()
        decision_latencies_us.append((t1 - t0) * 1e6)
        return result

    scheduler.schedule = instrumented_schedule

    # 3. Profiling & Memory Tracking
    tracemalloc.start()
    pr = cProfile.Profile() if profile_top_n > 0 else None

    wall_start = time.perf_counter()
    if pr:
        pr.enable()

    sim.run()

    if pr:
        pr.disable()
    wall_duration_sec = time.perf_counter() - wall_start

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mem_mb = peak_mem / (1024 * 1024)

    # 4. Latency Distribution Analysis
    decision_latencies_us.sort()
    n_decisions = len(decision_latencies_us)

    if n_decisions > 0:
        lat_min = decision_latencies_us[0]
        lat_max = decision_latencies_us[-1]
        lat_mean = sum(decision_latencies_us) / n_decisions
        lat_p50 = decision_latencies_us[int(0.50 * (n_decisions - 1))]
        lat_p95 = decision_latencies_us[int(0.95 * (n_decisions - 1))]
        lat_p99 = decision_latencies_us[int(0.99 * (n_decisions - 1))]
    else:
        lat_min = lat_max = lat_mean = lat_p50 = lat_p95 = lat_p99 = 0.0

    # 5. Throughput & Queue Counters
    completed_count = len(sim.completed_jobs)
    sim_throughput_jobs_sec = completed_count / wall_duration_sec if wall_duration_sec > 0 else 0.0
    decisions_per_sec = n_decisions / wall_duration_sec if wall_duration_sec > 0 else 0.0

    total_enqueued = sum(q.total_enqueued for q in qm._queues.values())
    total_dequeued = sum(q.total_dequeued for q in qm._queues.values())

    profile_output = ""
    if pr and profile_top_n > 0:
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(profile_top_n)
        profile_output = s.getvalue()

    return {
        "num_tenants": num_tenants,
        "total_jobs": len(workload),
        "completed_jobs": completed_count,
        "n_decisions": n_decisions,
        "wall_duration_sec": wall_duration_sec,
        "sim_throughput_jobs_sec": sim_throughput_jobs_sec,
        "decisions_per_sec": decisions_per_sec,
        "peak_mem_mb": peak_mem_mb,
        "total_enqueued": total_enqueued,
        "total_dequeued": total_dequeued,
        "lat_min_us": lat_min,
        "lat_mean_us": lat_mean,
        "lat_p50_us": lat_p50,
        "lat_p95_us": lat_p95,
        "lat_p99_us": lat_p99,
        "lat_max_us": lat_max,
        "sim_duration_ms": sim.current_time_ms,
        "profile_output": profile_output,
    }


def main():
    print("=" * 80)
    print("  OS SCHEDULING SCALABILITY EVALUATION: 10 TO 500 TENANTS")
    print("=" * 80)
    print(f"{'Tenants':<8} | {'Jobs':<6} | {'Decisions':<9} | {'Throughput':<12} | {'Lat Mean':<10} | {'Lat p95':<9} | {'Lat p99':<9} | {'Peak RAM':<9}")
    print(f"{'':<8} | {'':<6} | {'':<9} | {'(jobs/sec)':<12} | {'(μs)':<10} | {'(μs)':<9} | {'(μs)':<9} | {'(MB)':<9}")
    print("-" * 80)

    # 10, 25, 50, 100, 500 tenants
    tenant_scales = [10, 25, 50, 100, 500]
    jobs_per_tenant = {
        10: 100,   # 1,000 jobs
        25: 60,    # 1,500 jobs
        50: 40,    # 2,000 jobs
        100: 25,   # 2,500 jobs
        500: 10,   # 5,000 jobs
    }

    results = []
    for n in tenant_scales:
        jpt = jobs_per_tenant[n]
        # Profile top 15 calls for 100 and 500 tenants to spot bottlenecks
        top_n = 12 if n in (100, 500) else 0
        res = run_scaling_benchmark(n, jobs_per_tenant=jpt, profile_top_n=top_n)
        results.append(res)
        print(
            f"{res['num_tenants']:<8} | "
            f"{res['total_jobs']:<6} | "
            f"{res['n_decisions']:<9} | "
            f"{res['sim_throughput_jobs_sec']:<12.1f} | "
            f"{res['lat_mean_us']:<10.2f} | "
            f"{res['lat_p95_us']:<9.2f} | "
            f"{res['lat_p99_us']:<9.2f} | "
            f"{res['peak_mem_mb']:<9.2f}"
        )

    print("=" * 80)

    # Print top bottlenecks from 500-tenant run
    for res in results:
        if res["profile_output"]:
            print(f"\n--- cProfile Top Bottlenecks for {res['num_tenants']} Tenants ---")
            print(res["profile_output"])


if __name__ == "__main__":
    main()
