"""Reproducible multi-tenant benchmark scenarios across edge AI workload profiles.

Defines the 10 required evaluation scenarios:
Scenario 1:  2 equal tenants
Scenario 2:  4 equal tenants
Scenario 3:  unequal tenant weights
Scenario 4:  one heavy tenant + several light tenants
Scenario 5:  bursty workloads
Scenario 6:  mixed CPU/GPU workloads
Scenario 7:  high-priority tenant
Scenario 8:  quota-constrained tenant
Scenario 9:  many tenants (8+ concurrent tenants)
Scenario 10: short jobs mixed with long jobs
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional

from src.core.types import WorkloadClass
from src.core.job import GPUJob
from src.core.tenant import Tenant


@dataclass
class BenchmarkScenario:
    """Represents a standardized, reproducible multi-tenant evaluation scenario."""
    scenario_id: str
    name: str
    description: str
    tenants: Dict[str, Tenant]
    workload_generator: Callable[[int], List[GPUJob]]
    recommended_duration_ms: float = 500.0

    def generate_jobs(self, random_seed: int = 42, seed: Optional[int] = None) -> List[GPUJob]:
        """Generates deterministic, bit-for-bit reproducible workloads."""
        actual_seed = seed if seed is not None else random_seed
        return self.workload_generator(actual_seed)


# =========================================================================
# Scenario 1: 2 Equal Tenants
# =========================================================================
def _generate_scenario_1_jobs(seed: int) -> List[GPUJob]:
    rng = random.Random(seed)
    jobs: List[GPUJob] = []
    # 10 jobs per tenant, interleaved arrivals over [0, 100ms]
    for i in range(10):
        t_arr = i * 10.0 + rng.uniform(0.0, 2.0)
        jobs.append(GPUJob(f"T1_J{i}", "tenant_1", "mobilenet_v3", arrival_time=t_arr, estimated_gpu_time=12.0, remaining_gpu_time=12.0, priority=5, deadline=t_arr + 80.0, workload_type=WorkloadClass.BALANCED))
        jobs.append(GPUJob(f"T2_J{i}", "tenant_2", "mobilenet_v3", arrival_time=t_arr + 1.0, estimated_gpu_time=12.0, remaining_gpu_time=12.0, priority=5, deadline=t_arr + 80.0, workload_type=WorkloadClass.BALANCED))
    jobs.sort(key=lambda j: j.arrival_time)
    return jobs


def create_scenario_1() -> BenchmarkScenario:
    t1 = Tenant("tenant_1", name="Tenant 1", weight=1.0, priority=5, gpu_quota=100.0)
    t2 = Tenant("tenant_2", name="Tenant 2", weight=1.0, priority=5, gpu_quota=100.0)
    return BenchmarkScenario(
        scenario_id="scenario_1",
        name="2 Equal Tenants",
        description="Two symmetric tenants with identical weights (1:1), priorities, and unconstrained quotas under interleaved load.",
        tenants={t1.tenant_id: t1, t2.tenant_id: t2},
        workload_generator=_generate_scenario_1_jobs,
        recommended_duration_ms=300.0,
    )


# =========================================================================
# Scenario 2: 4 Equal Tenants
# =========================================================================
def _generate_scenario_2_jobs(seed: int) -> List[GPUJob]:
    rng = random.Random(seed)
    jobs: List[GPUJob] = []
    # 5 jobs per tenant across 4 tenants (20 jobs total)
    for i in range(5):
        base_t = i * 20.0
        for t_idx in range(1, 5):
            t_arr = base_t + (t_idx - 1) * 3.0 + rng.uniform(0.0, 1.0)
            jobs.append(GPUJob(f"T{t_idx}_J{i}", f"tenant_{t_idx}", "resnet18", arrival_time=t_arr, estimated_gpu_time=10.0, remaining_gpu_time=10.0, priority=5, deadline=t_arr + 100.0, workload_type=WorkloadClass.BALANCED))
    jobs.sort(key=lambda j: j.arrival_time)
    return jobs


def create_scenario_2() -> BenchmarkScenario:
    tenants = {
        f"tenant_{i}": Tenant(f"tenant_{i}", name=f"Tenant {i}", weight=1.0, priority=5, gpu_quota=100.0)
        for i in range(1, 5)
    }
    return BenchmarkScenario(
        scenario_id="scenario_2",
        name="4 Equal Tenants",
        description="Four symmetric tenants competing with equal weights (1:1:1:1) and staggered arrivals.",
        tenants=tenants,
        workload_generator=_generate_scenario_2_jobs,
        recommended_duration_ms=300.0,
    )


# =========================================================================
# Scenario 3: Unequal Tenant Weights
# =========================================================================
def _generate_scenario_3_jobs(seed: int) -> List[GPUJob]:
    rng = random.Random(seed)
    jobs: List[GPUJob] = []
    # Heavy (weight 4.0): 16 jobs; Medium (weight 2.0): 8 jobs; Light (weight 1.0): 4 jobs
    # Arriving under contention in the first 40ms
    for i in range(16):
        jobs.append(GPUJob(f"H_J{i}", "tenant_heavy", "yolov8n", arrival_time=i * 2.0 + rng.uniform(0.0, 0.5), estimated_gpu_time=10.0, remaining_gpu_time=10.0, priority=5, deadline=200.0))
    for i in range(8):
        jobs.append(GPUJob(f"M_J{i}", "tenant_medium", "mobilenet_v3", arrival_time=i * 4.0 + rng.uniform(0.0, 0.5), estimated_gpu_time=10.0, remaining_gpu_time=10.0, priority=5, deadline=200.0))
    for i in range(4):
        jobs.append(GPUJob(f"L_J{i}", "tenant_light", "sensor_mlp", arrival_time=i * 8.0 + rng.uniform(0.0, 0.5), estimated_gpu_time=10.0, remaining_gpu_time=10.0, priority=5, deadline=200.0))
    jobs.sort(key=lambda j: j.arrival_time)
    return jobs


def create_scenario_3() -> BenchmarkScenario:
    t_heavy = Tenant("tenant_heavy", name="Heavy Tenant", weight=4.0, priority=5, gpu_quota=100.0)
    t_med = Tenant("tenant_medium", name="Medium Tenant", weight=2.0, priority=5, gpu_quota=100.0)
    t_light = Tenant("tenant_light", name="Light Tenant", weight=1.0, priority=5, gpu_quota=100.0)
    return BenchmarkScenario(
        scenario_id="scenario_3",
        name="Unequal Tenant Weights",
        description="Three tenants with heterogeneous weights (4.0 : 2.0 : 1.0) competing under high workload contention.",
        tenants={t_heavy.tenant_id: t_heavy, t_med.tenant_id: t_med, t_light.tenant_id: t_light},
        workload_generator=_generate_scenario_3_jobs,
        recommended_duration_ms=350.0,
    )


# =========================================================================
# Scenario 4: One Heavy Tenant + Several Light Tenants
# =========================================================================
def _generate_scenario_4_jobs(seed: int) -> List[GPUJob]:
    rng = random.Random(seed)
    jobs: List[GPUJob] = []
    # 1 heavy rogue tenant submits 4 massive vision jobs (50ms each = 200ms total)
    for i in range(4):
        jobs.append(GPUJob(f"HEAVY_J{i}", "tenant_heavy", "yolov8_large", arrival_time=i * 15.0, estimated_gpu_time=50.0, remaining_gpu_time=50.0, priority=5, deadline=300.0, workload_type=WorkloadClass.GPU_INTENSIVE))
    # 3 light tenants submit rapid 2.5ms audio/sensor requests
    for t_idx in (1, 2, 3):
        t_id = f"tenant_light_{t_idx}"
        for i in range(5):
            t_arr = i * 8.0 + rng.uniform(0.0, 1.0)
            jobs.append(GPUJob(f"L{t_idx}_J{i}", t_id, "audio_kws", arrival_time=t_arr, estimated_gpu_time=2.5, remaining_gpu_time=2.5, priority=5, deadline=t_arr + 40.0, workload_type=WorkloadClass.IO_BOUND))
    jobs.sort(key=lambda j: j.arrival_time)
    return jobs


def create_scenario_4() -> BenchmarkScenario:
    t_heavy = Tenant("tenant_heavy", name="Heavy Vision Tenant", weight=1.0, priority=5, gpu_quota=100.0)
    t_l1 = Tenant("tenant_light_1", name="Light Audio 1", weight=1.0, priority=5, gpu_quota=100.0)
    t_l2 = Tenant("tenant_light_2", name="Light Audio 2", weight=1.0, priority=5, gpu_quota=100.0)
    t_l3 = Tenant("tenant_light_3", name="Light Sensor 3", weight=1.0, priority=5, gpu_quota=100.0)
    return BenchmarkScenario(
        scenario_id="scenario_4",
        name="One Heavy Tenant + Several Light Tenants",
        description="Evaluates Head-of-Line (HoL) blocking when one tenant submits 50ms kernels while 3 tenants submit fast 2.5ms requests.",
        tenants={t_heavy.tenant_id: t_heavy, t_l1.tenant_id: t_l1, t_l2.tenant_id: t_l2, t_l3.tenant_id: t_l3},
        workload_generator=_generate_scenario_4_jobs,
        recommended_duration_ms=300.0,
    )


# =========================================================================
# Scenario 5: Bursty Workloads
# =========================================================================
def _generate_scenario_5_jobs(seed: int) -> List[GPUJob]:
    rng = random.Random(seed)
    jobs: List[GPUJob] = []
    # Bursts arrive at t=0ms, t=80ms, t=160ms
    burst_centers = [0.0, 80.0, 160.0]
    for b_idx, center in enumerate(burst_centers):
        # 5 jobs per tenant per burst (30 jobs total)
        for i in range(5):
            arr1 = center + rng.uniform(0.0, 2.0)
            arr2 = center + rng.uniform(0.0, 2.0)
            jobs.append(GPUJob(f"B{b_idx}_A{i}", "tenant_burst_A", "model_burst", arrival_time=arr1, estimated_gpu_time=6.0, remaining_gpu_time=6.0, priority=5, deadline=center + 50.0))
            jobs.append(GPUJob(f"B{b_idx}_B{i}", "tenant_burst_B", "model_burst", arrival_time=arr2, estimated_gpu_time=6.0, remaining_gpu_time=6.0, priority=5, deadline=center + 50.0))
    jobs.sort(key=lambda j: j.arrival_time)
    return jobs


def create_scenario_5() -> BenchmarkScenario:
    t_a = Tenant("tenant_burst_A", name="Burst Tenant A", weight=1.0, priority=5, gpu_quota=100.0)
    t_b = Tenant("tenant_burst_B", name="Burst Tenant B", weight=1.0, priority=5, gpu_quota=100.0)
    return BenchmarkScenario(
        scenario_id="scenario_5",
        name="Bursty Workloads",
        description="High-frequency arrival surges separated by idle intervals evaluating queue recovery and burst absorption.",
        tenants={t_a.tenant_id: t_a, t_b.tenant_id: t_b},
        workload_generator=_generate_scenario_5_jobs,
        recommended_duration_ms=250.0,
    )


# =========================================================================
# Scenario 6: Mixed CPU/GPU Workloads
# =========================================================================
def _generate_scenario_6_jobs(seed: int) -> List[GPUJob]:
    rng = random.Random(seed)
    jobs: List[GPUJob] = []
    # 4 distinct workload categories
    for i in range(5):
        t_base = i * 20.0
        # GPU Intensive: heavy 30ms matrix operations
        jobs.append(GPUJob(f"GPU_{i}", "tenant_gpu", "yolov8n", arrival_time=t_base + rng.uniform(0.0, 1.0), estimated_gpu_time=30.0, remaining_gpu_time=30.0, priority=5, workload_type=WorkloadClass.GPU_INTENSIVE))
        # CPU Bound: light 1.5ms operations
        jobs.append(GPUJob(f"CPU_{i}", "tenant_cpu", "sensor_anomaly", arrival_time=t_base + 2.0 + rng.uniform(0.0, 1.0), estimated_gpu_time=1.5, remaining_gpu_time=1.5, priority=5, workload_type=WorkloadClass.CPU_BOUND))
        # IO Bound: 3.0ms audio streams
        jobs.append(GPUJob(f"IO_{i}", "tenant_io", "audio_kws", arrival_time=t_base + 4.0 + rng.uniform(0.0, 1.0), estimated_gpu_time=3.0, remaining_gpu_time=3.0, priority=5, workload_type=WorkloadClass.IO_BOUND))
        # Balanced: 10.0ms classification
        jobs.append(GPUJob(f"BAL_{i}", "tenant_balanced", "mobilenet", arrival_time=t_base + 6.0 + rng.uniform(0.0, 1.0), estimated_gpu_time=10.0, remaining_gpu_time=10.0, priority=5, workload_type=WorkloadClass.BALANCED))
    jobs.sort(key=lambda j: j.arrival_time)
    return jobs


def create_scenario_6() -> BenchmarkScenario:
    t_gpu = Tenant("tenant_gpu", name="GPU Intensive Tenant", weight=1.0, priority=5, gpu_quota=100.0)
    t_cpu = Tenant("tenant_cpu", name="CPU Bound Tenant", weight=1.0, priority=5, gpu_quota=100.0)
    t_io = Tenant("tenant_io", name="I/O Bound Tenant", weight=1.0, priority=5, gpu_quota=100.0)
    t_bal = Tenant("tenant_balanced", name="Balanced Tenant", weight=1.0, priority=5, gpu_quota=100.0)
    return BenchmarkScenario(
        scenario_id="scenario_6",
        name="Mixed CPU/GPU Workloads",
        description="Heterogeneous execution profiles across GPU-intensive, CPU-bound, I/O-bound, and balanced AI workloads.",
        tenants={t_gpu.tenant_id: t_gpu, t_cpu.tenant_id: t_cpu, t_io.tenant_id: t_io, t_bal.tenant_id: t_bal},
        workload_generator=_generate_scenario_6_jobs,
        recommended_duration_ms=300.0,
    )


# =========================================================================
# Scenario 7: High-Priority Tenant
# =========================================================================
def _generate_scenario_7_jobs(seed: int) -> List[GPUJob]:
    rng = random.Random(seed)
    jobs: List[GPUJob] = []
    # Low priority batch tenant (Priority 1): 10 jobs arriving at t=0
    for i in range(10):
        jobs.append(GPUJob(f"BATCH_{i}", "tenant_batch", "indexing_resnet", arrival_time=0.0, estimated_gpu_time=15.0, remaining_gpu_time=15.0, priority=1))
    # Medium priority tenant (Priority 5): 5 jobs arriving over time
    for i in range(5):
        t_arr = 15.0 + i * 25.0
        jobs.append(GPUJob(f"MED_{i}", "tenant_normal", "surveillance_cnn", arrival_time=t_arr, estimated_gpu_time=12.0, remaining_gpu_time=12.0, priority=5))
    # Critical tenant (Priority 10): 4 periodic urgent jobs with strict deadlines
    for i in range(4):
        t_arr = 20.0 + i * 35.0
        jobs.append(GPUJob(f"CRIT_{i}", "tenant_critical", "safety_trigger", arrival_time=t_arr, estimated_gpu_time=8.0, remaining_gpu_time=8.0, priority=10, deadline=t_arr + 25.0))
    jobs.sort(key=lambda j: j.arrival_time)
    return jobs


def create_scenario_7() -> BenchmarkScenario:
    t_crit = Tenant("tenant_critical", name="Critical Safety Tenant", weight=1.0, priority=10, gpu_quota=100.0)
    t_norm = Tenant("tenant_normal", name="Normal Interactive Tenant", weight=1.0, priority=5, gpu_quota=100.0)
    t_batch = Tenant("tenant_batch", name="Background Batch Tenant", weight=1.0, priority=1, gpu_quota=100.0)
    return BenchmarkScenario(
        scenario_id="scenario_7",
        name="High-Priority Tenant",
        description="Evaluates priority responsiveness versus starvation: Critical tenant (p=10) vs Background batch (p=1).",
        tenants={t_crit.tenant_id: t_crit, t_norm.tenant_id: t_norm, t_batch.tenant_id: t_batch},
        workload_generator=_generate_scenario_7_jobs,
        recommended_duration_ms=300.0,
    )


# =========================================================================
# Scenario 8: Quota-Constrained Tenant
# =========================================================================
def _generate_scenario_8_jobs(seed: int) -> List[GPUJob]:
    jobs: List[GPUJob] = []
    # Constrained tenant floods system with 14 jobs of 15ms (210ms compute, quota is 25% of 200ms = 50ms)
    for i in range(14):
        jobs.append(GPUJob(f"CONST_{i}", "tenant_constrained", "model_flood", arrival_time=i * 5.0, estimated_gpu_time=15.0, remaining_gpu_time=15.0, priority=5))
    # Unconstrained tenant submits 8 jobs of 15ms
    for i in range(8):
        jobs.append(GPUJob(f"UNCONST_{i}", "tenant_unconstrained", "model_normal", arrival_time=10.0 + i * 15.0, estimated_gpu_time=15.0, remaining_gpu_time=15.0, priority=5))
    jobs.sort(key=lambda j: j.arrival_time)
    return jobs


def create_scenario_8() -> BenchmarkScenario:
    t_const = Tenant("tenant_constrained", name="Constrained Tenant", weight=1.0, priority=5, gpu_quota=25.0, quota_window_ms=200.0)
    t_unconst = Tenant("tenant_unconstrained", name="Unconstrained Tenant", weight=1.0, priority=5, gpu_quota=75.0, quota_window_ms=200.0)
    return BenchmarkScenario(
        scenario_id="scenario_8",
        name="Quota-Constrained Tenant",
        description="Evaluates sliding-window GPU quota throttling: Constrained tenant (25% quota) vs Unconstrained tenant (75%).",
        tenants={t_const.tenant_id: t_const, t_unconst.tenant_id: t_unconst},
        workload_generator=_generate_scenario_8_jobs,
        recommended_duration_ms=350.0,
    )


# =========================================================================
# Scenario 9: Many Tenants (High Concurrency)
# =========================================================================
def _generate_scenario_9_jobs(seed: int) -> List[GPUJob]:
    rng = random.Random(seed)
    jobs: List[GPUJob] = []
    # 8 distinct tenants, 4 jobs each (32 jobs total)
    for t_idx in range(8):
        t_id = f"tenant_{t_idx}"
        for j_idx in range(4):
            t_arr = j_idx * 25.0 + rng.uniform(0.0, 5.0)
            dur = 8.0 + (t_idx % 3) * 4.0
            jobs.append(GPUJob(f"T{t_idx}_J{j_idx}", t_id, "multi_model", arrival_time=t_arr, estimated_gpu_time=dur, remaining_gpu_time=dur, priority=(t_idx % 5) + 1, deadline=t_arr + 150.0))
    jobs.sort(key=lambda j: j.arrival_time)
    return jobs


def create_scenario_9() -> BenchmarkScenario:
    tenants = {
        f"tenant_{i}": Tenant(f"tenant_{i}", name=f"Tenant {i}", weight=1.0 + (i % 3) * 0.5, priority=(i % 5) + 1, gpu_quota=100.0)
        for i in range(8)
    }
    return BenchmarkScenario(
        scenario_id="scenario_9",
        name="Many Tenants (High Concurrency)",
        description="Scalability and multi-tenant isolation under high concurrency with 8 active tenants.",
        tenants=tenants,
        workload_generator=_generate_scenario_9_jobs,
        recommended_duration_ms=400.0,
    )


# =========================================================================
# Scenario 10: Short Jobs Mixed with Long Jobs
# =========================================================================
def _generate_scenario_10_jobs(seed: int) -> List[GPUJob]:
    jobs: List[GPUJob] = []
    # Bulk tenant submits 3 long 50ms jobs at t=0, t=20, t=40
    for i in range(3):
        jobs.append(GPUJob(f"BULK_{i}", "tenant_bulk", "heavy_transformer", arrival_time=i * 20.0, estimated_gpu_time=50.0, remaining_gpu_time=50.0, priority=5, deadline=300.0, workload_type=WorkloadClass.GPU_INTENSIVE))
    # Interactive tenant submits 15 short 2.0ms jobs arriving periodically every 10ms
    for i in range(15):
        t_arr = 5.0 + i * 10.0
        jobs.append(GPUJob(f"INTER_{i}", "tenant_interactive", "fast_sensor", arrival_time=t_arr, estimated_gpu_time=2.0, remaining_gpu_time=2.0, priority=5, deadline=t_arr + 30.0, workload_type=WorkloadClass.CPU_BOUND))
    jobs.sort(key=lambda j: j.arrival_time)
    return jobs


def create_scenario_10() -> BenchmarkScenario:
    t_bulk = Tenant("tenant_bulk", name="Bulk Tenant", weight=1.0, priority=5, gpu_quota=100.0)
    t_inter = Tenant("tenant_interactive", name="Interactive Tenant", weight=1.0, priority=5, gpu_quota=100.0)
    return BenchmarkScenario(
        scenario_id="scenario_10",
        name="Short Jobs Mixed with Long Jobs",
        description="Bimodal execution distribution evaluating short-job responsiveness heuristic and quantum slicing.",
        tenants={t_bulk.tenant_id: t_bulk, t_inter.tenant_id: t_inter},
        workload_generator=_generate_scenario_10_jobs,
        recommended_duration_ms=300.0,
    )


# =========================================================================
# Scenario Registry
# =========================================================================
SCENARIO_FACTORIES = {
    "scenario_1": create_scenario_1,
    "scenario_2": create_scenario_2,
    "scenario_3": create_scenario_3,
    "scenario_4": create_scenario_4,
    "scenario_5": create_scenario_5,
    "scenario_6": create_scenario_6,
    "scenario_7": create_scenario_7,
    "scenario_8": create_scenario_8,
    "scenario_9": create_scenario_9,
    "scenario_10": create_scenario_10,
}


def get_scenario(scenario_id: str) -> BenchmarkScenario:
    """Retrieves a benchmark scenario by its identifier."""
    norm_id = scenario_id.lower().replace("-", "_").strip()
    # 1. Exact match
    if norm_id in SCENARIO_FACTORIES:
        return SCENARIO_FACTORIES[norm_id]()
    # 2. Number shorthand: e.g. '1' -> 'scenario_1', '10' -> 'scenario_10'
    if f"scenario_{norm_id}" in SCENARIO_FACTORIES:
        return SCENARIO_FACTORIES[f"scenario_{norm_id}"]()
    # 3. Descriptive suffix: e.g. 'scenario_1_two_equal_tenants'
    for key, factory in SCENARIO_FACTORIES.items():
        if norm_id.startswith(f"{key}_"):
            return factory()
    raise KeyError(f"Unknown scenario ID '{scenario_id}'. Available: {list(SCENARIO_FACTORIES.keys())}")


def get_all_scenarios() -> List[BenchmarkScenario]:
    """Instantiates all 10 standard benchmark scenarios."""
    return [factory() for factory in SCENARIO_FACTORIES.values()]
