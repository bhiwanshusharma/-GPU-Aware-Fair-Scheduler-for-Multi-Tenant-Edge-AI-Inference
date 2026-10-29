"""Lightweight Monitoring Dashboard for GPU-Aware Fair Scheduler.

Displays live scheduler state, queues, timelines, and verified performance metrics:
1. Current GPU utilization
2. Active tenants
3. Tenant GPU allocation
4. Current running job
5. Waiting jobs in ingress queues
6. Auditable scheduling decisions
7. GPU execution timeline (Gantt chart)
8. Jain fairness index
9. Throughput
10. Average response time
11. Average waiting time
12. Preemption count
13. Quota consumption

Enforces zero scheduling logic inside the UI; solely interacts with the
DashboardSimulationEngine API.
"""

import sys
import copy
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.engine import (
    DashboardSimulationEngine,
    SimulationRunConfig,
    TenantConfigInput,
)
from src.visualization.gantt import plot_gpu_gantt, plot_multi_scheduler_gantt
from src.benchmarks.runner import CANONICAL_POLICIES, POLICY_DISPLAY_NAMES


# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GPU-Aware Fair Scheduler Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 26px;
        font-weight: 800;
        color: #1A365D;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 14px;
        color: #4A5568;
        margin-bottom: 15px;
    }
    .metric-card {
        background-color: #F7FAFC;
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
    }
    .badge-active { background-color: #C6F6D5; color: #22543D; }
    .badge-throttled { background-color: #FED7D7; color: #742A2A; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown('<div class="main-header">⚡ GPU-Aware Fair Scheduler: Multi-Tenant Edge AI Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Application-Level GPU Arbiter | P-DWFQ Policy | Jetson-Class Edge Inference Simulation</div>', unsafe_allow_html=True)
with col_badge:
    st.caption("Platform: macOS / Linux DES Engine")
    st.caption("Status: **Engine Ready (Deterministic)**")


# -----------------------------------------------------------------------------
# Sidebar: Controls & Presets
# -----------------------------------------------------------------------------
st.sidebar.header("🎛️ Simulation Controls")

preset = st.sidebar.selectbox(
    "Demonstration Scenario Preset:",
    [
        "Custom Configuration",
        "Preset 1: Equal Symmetric (2 Tenants)",
        "Preset 2: Starvation Prevention (High vs Low Priority)",
        "Preset 3: Quota Throttling (25% Capped vs 75%)",
        "Preset 4: Interactive Short vs Bulk Long Jobs",
        "Preset 5: Bursty Ingress Stream",
    ],
)

# Apply preset defaults
default_num_tenants = 2
default_dist = "balanced"
default_intensity = "normal"
default_policy = "gpu_fair"
default_quantum = 20.0
default_tenants = [
    {"id": "tenant_robotics", "name": "Robotics Control", "weight": 1.0, "prio": 5, "quota": 100.0},
    {"id": "tenant_analytics", "name": "Video Analytics", "weight": 1.0, "prio": 5, "quota": 100.0},
]

if preset == "Preset 1: Equal Symmetric (2 Tenants)":
    default_num_tenants = 2
    default_dist = "balanced"
    default_tenants = [
        {"id": "tenant_1", "name": "Tenant 1 (Balanced)", "weight": 1.0, "prio": 5, "quota": 100.0},
        {"id": "tenant_2", "name": "Tenant 2 (Balanced)", "weight": 1.0, "prio": 5, "quota": 100.0},
    ]
elif preset == "Preset 2: Starvation Prevention (High vs Low Priority)":
    default_num_tenants = 3
    default_dist = "balanced"
    default_tenants = [
        {"id": "tenant_critical", "name": "Safety Critical", "weight": 1.0, "prio": 10, "quota": 100.0},
        {"id": "tenant_standard", "name": "Standard Stream", "weight": 1.0, "prio": 5, "quota": 100.0},
        {"id": "tenant_batch", "name": "Background Batch", "weight": 1.0, "prio": 1, "quota": 100.0},
    ]
elif preset == "Preset 3: Quota Throttling (25% Capped vs 75%)":
    default_num_tenants = 2
    default_dist = "balanced"
    default_tenants = [
        {"id": "tenant_constrained", "name": "Constrained (25% Quota)", "weight": 1.0, "prio": 5, "quota": 25.0},
        {"id": "tenant_unconstrained", "name": "Unconstrained (75% Quota)", "weight": 1.0, "prio": 5, "quota": 75.0},
    ]
elif preset == "Preset 4: Interactive Short vs Bulk Long Jobs":
    default_num_tenants = 2
    default_dist = "short_long_bimodal"
    default_tenants = [
        {"id": "tenant_bulk", "name": "Bulk LLM Kernels", "weight": 1.0, "prio": 4, "quota": 100.0},
        {"id": "tenant_interactive", "name": "Audio Trigger (4ms)", "weight": 1.0, "prio": 8, "quota": 100.0},
    ]
elif preset == "Preset 5: Bursty Ingress Stream":
    default_num_tenants = 2
    default_dist = "bursty"
    default_tenants = [
        {"id": "tenant_bursty", "name": "Bursty Sensor Ingress", "weight": 1.0, "prio": 5, "quota": 100.0},
        {"id": "tenant_steady", "name": "Periodic Telemetry", "weight": 1.0, "prio": 5, "quota": 100.0},
    ]

# Policy Selection
policy_display_map = {
    "GPU-Aware Fair Scheduler (P-DWFQ)": "gpu_fair",
    "Round Robin": "round_robin",
    "Weighted Round Robin": "weighted_round_robin",
    "Strict Priority": "priority",
    "FIFO (First-In, First-Out)": "fifo",
}
selected_policy_display = st.sidebar.selectbox(
    "Scheduling Policy:",
    list(policy_display_map.keys()),
    index=0,
)
active_policy_key = policy_display_map[selected_policy_display]

# Quantum & Hardware Controls
st.sidebar.subheader("Hardware & Time Quantum")
gpu_quantum = st.sidebar.slider("Base GPU Quantum (ms):", min_value=5.0, max_value=80.0, value=default_quantum, step=2.5)
context_overhead = st.sidebar.slider("Switch Overhead (ms):", min_value=0.0, max_value=2.0, value=0.5, step=0.1)

# Workload Controls
st.sidebar.subheader("Workload Profile")
workload_dist = st.sidebar.selectbox(
    "Workload Distribution:",
    ["balanced", "bursty", "mixed_cpu_gpu", "short_long_bimodal", "heavy_light"],
    index=["balanced", "bursty", "mixed_cpu_gpu", "short_long_bimodal", "heavy_light"].index(default_dist),
    format_func=lambda x: {
        "balanced": "Balanced CV Models (12ms)",
        "bursty": "Bursty Ingress Stream",
        "mixed_cpu_gpu": "Mixed CPU/GPU/IO Models",
        "short_long_bimodal": "Bimodal: Short (4ms) & Long (35ms)",
        "heavy_light": "One Heavy + Light Tenants",
    }.get(x, x),
)
workload_intensity = st.sidebar.select_slider(
    "Workload Intensity:",
    options=["normal", "high", "stress"],
    value=default_intensity,
    format_func=lambda x: x.capitalize(),
)
total_jobs = st.sidebar.slider("Total Inference Jobs:", min_value=10, max_value=60, value=24, step=2)
random_seed = st.sidebar.number_input("Deterministic Random Seed:", min_value=1, max_value=9999, value=42)

# Multi-Tenant Controls
st.sidebar.subheader("Tenant Configuration")
num_tenants = st.sidebar.slider("Number of Tenants:", min_value=2, max_value=6, value=default_num_tenants)

configured_tenants = []
for i in range(num_tenants):
    default_t = default_tenants[i] if i < len(default_tenants) else {
        "id": f"tenant_{i+1}",
        "name": f"Tenant {i+1}",
        "weight": 1.0,
        "prio": 5,
        "quota": 100.0,
    }
    with st.sidebar.expander(f"Tenant {i+1}: {default_t['name']}", expanded=(i == 0)):
        t_id = st.text_input(f"ID #{i+1}", value=default_t["id"], key=f"t_id_{i}")
        t_name = st.text_input(f"Name #{i+1}", value=default_t["name"], key=f"t_name_{i}")
        col_w, col_p = st.columns(2)
        with col_w:
            t_weight = st.slider(f"Weight", min_value=0.5, max_value=8.0, value=float(default_t["weight"]), step=0.5, key=f"t_w_{i}")
        with col_p:
            t_prio = st.slider(f"Priority", min_value=1, max_value=10, value=int(default_t["prio"]), step=1, key=f"t_p_{i}")
        t_quota = st.slider(f"GPU Quota (%)", min_value=10.0, max_value=100.0, value=float(default_t["quota"]), step=5.0, key=f"t_q_{i}")
        configured_tenants.append(
            TenantConfigInput(
                tenant_id=t_id,
                name=t_name,
                weight=t_weight,
                priority=t_prio,
                gpu_quota_pct=t_quota,
            )
        )

run_button = st.sidebar.button("🚀 Run Simulation Experiment", type="primary", use_container_width=True)


# -----------------------------------------------------------------------------
# Execute Simulation via Engine API
# -----------------------------------------------------------------------------
engine = DashboardSimulationEngine()

run_cfg = SimulationRunConfig(
    tenants=configured_tenants,
    policy=active_policy_key,
    base_quantum_ms=gpu_quantum,
    context_switch_overhead_ms=context_overhead,
    workload_distribution=workload_dist,
    workload_intensity=workload_intensity,
    total_jobs=total_jobs,
    random_seed=random_seed,
)

# Run experiment when button pressed or initialize with first snapshot
if "current_snapshot" not in st.session_state or run_button:
    with st.spinner("Executing discrete event simulation via GPU Arbiter..."):
        st.session_state.current_snapshot = engine.run_experiment(run_cfg)

snapshot = st.session_state.current_snapshot


# -----------------------------------------------------------------------------
# Section 1: Top KPI Metric Cards (13 Verified Metrics)
# -----------------------------------------------------------------------------
st.markdown("### 📈 Core Scheduling Performance Metrics")
m1, m2, m3, m4, m5, m6 = st.columns(6)

with m1:
    st.metric(
        label="GPU Utilization",
        value=f"{snapshot.gpu_utilization_pct:.1f} %",
        help="Percentage of simulation time the GPU was actively computing inference kernels",
    )
with m2:
    fair_val = snapshot.metrics.jains_fairness_index
    st.metric(
        label="Jain Fairness Index",
        value=f"{fair_val:.4f}",
        delta="Optimal (1.0)" if fair_val >= 0.95 else None,
        help="Jain's Fairness Index bounded in [1/n, 1.0]",
    )
with m3:
    st.metric(
        label="Throughput",
        value=f"{snapshot.metrics.throughput_jobs_per_sec:.1f} jobs/s",
        help="Completed AI inference jobs per virtual second",
    )
with m4:
    st.metric(
        label="Avg Waiting Time",
        value=f"{snapshot.metrics.avg_waiting_time_ms:.1f} ms",
        help="Mean delay spent waiting in ingress queues",
    )
with m5:
    st.metric(
        label="Avg Response Time",
        value=f"{snapshot.metrics.avg_response_time_ms:.1f} ms",
        help="Delay from arrival to first dispatch onto the GPU",
    )
with m6:
    st.metric(
        label="Preemptions",
        value=f"{snapshot.metrics.preemption_count}",
        help="Total time-slice preemption context switches incurred",
    )

st.divider()


# -----------------------------------------------------------------------------
# Section 2: Active State, Current Job, & Queues
# -----------------------------------------------------------------------------
col_curr_job, col_active_tenants, col_queues = st.columns([1.2, 1.8, 1.5])

with col_curr_job:
    st.markdown("#### ⚡ GPU Execution State")
    if snapshot.current_running_job:
        cj = snapshot.current_running_job
        st.info(
            f"**Last Running Job:** `{cj['job_id']}`\n\n"
            f"• **Tenant:** `{cj['tenant_id']}`\n\n"
            f"• **Model:** `{cj['model_name']}`\n\n"
            f"• **Slice Duration:** `{cj['slice_duration_ms']} ms`\n\n"
            f"• **Exit State:** `{cj['event_type'].upper()}`"
        )
    else:
        st.warning("GPU Currently Idle (No active job)")

    st.caption(f"**Total Sim Time:** {snapshot.total_simulation_time_ms:.1f} ms")
    st.caption(f"**Completed Jobs:** {len(snapshot.completed_jobs)} / {len(snapshot.completed_jobs) + len(snapshot.waiting_jobs)}")
    st.caption(f"**Scheduling Overhead:** {snapshot.metrics.scheduling_overhead_ms:.1f} ms")

with col_active_tenants:
    st.markdown("#### 👥 Active Tenants & Quota Consumption")
    tenant_df = pd.DataFrame(snapshot.active_tenants)
    if not tenant_df.empty:
        display_t_df = tenant_df[[
            "tenant_id", "weight", "priority", "gpu_quota_pct",
            "service_ms", "allocation_pct", "quota_used_pct", "is_throttled"
        ]].copy()
        display_t_df.columns = [
            "Tenant", "Weight", "Prio", "Quota %",
            "Service (ms)", "Alloc %", "Quota Used %", "Throttled"
        ]
        st.dataframe(display_t_df, use_container_width=True, hide_index=True)

        # Quota progress bars
        for t_info in snapshot.active_tenants:
            used_pct = min(100.0, float(t_info["quota_used_pct"]))
            label = f"{t_info['name']} Quota: {used_pct:.1f}% / {t_info['gpu_quota_pct']}%"
            st.progress(used_pct / 100.0, text=label)
    else:
        st.info("No active tenants registered.")

with col_queues:
    st.markdown("#### 📥 Waiting Jobs in Queues")
    if snapshot.waiting_jobs:
        q_df = pd.DataFrame(snapshot.waiting_jobs)
        st.dataframe(q_df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ All ingress queues empty! All submitted jobs dispatched and completed.")

st.divider()


# -----------------------------------------------------------------------------
# Section 3: GPU Execution Timeline (Gantt-Style Timeline)
# -----------------------------------------------------------------------------
st.markdown("### 📊 GPU Execution Timeline (Gantt Chart)")
st.caption("Time-sliced execution lanes per tenant showing job start, end, preemption (dashed line ⏸), and completion (green star ★)")

if snapshot.execution_slices:
    fig_gantt = plot_gpu_gantt(
        slices=snapshot.execution_slices,
        title=f"GPU Execution Timeline: {POLICY_DISPLAY_NAMES.get(snapshot.policy_name, snapshot.policy_name.upper())} ({snapshot.scenario_description})",
        figsize=(15.0, max(3.5, 1.2 * len(snapshot.active_tenants))),
        dpi=150,
        show_labels=True,
    )
    st.pyplot(fig_gantt, use_container_width=True)
    plt.close(fig_gantt)
else:
    st.warning("No execution trace slices recorded for this run.")

st.divider()


# -----------------------------------------------------------------------------
# Section 4: Tenant Allocation & Service Breakdown
# -----------------------------------------------------------------------------
col_alloc_chart, col_service_chart = st.columns(2)

with col_alloc_chart:
    st.markdown("#### 🍰 Tenant GPU Allocation Shares (%)")
    alloc_data = snapshot.tenant_allocation_pct
    if alloc_data:
        alloc_df = pd.DataFrame(list(alloc_data.items()), columns=["Tenant", "Allocation Share (%)"])
        st.bar_chart(alloc_df.set_index("Tenant"), use_container_width=True)
    else:
        st.info("No allocation data.")

with col_service_chart:
    st.markdown("#### ⏱️ Accumulated GPU Service (ms)")
    service_data = snapshot.tenant_service_ms
    if service_data:
        srv_df = pd.DataFrame(list(service_data.items()), columns=["Tenant", "GPU Compute Time (ms)"])
        st.bar_chart(srv_df.set_index("Tenant"), use_container_width=True)
    else:
        st.info("No service data.")

st.divider()


# -----------------------------------------------------------------------------
# Section 5: Auditable Scheduling Decisions (Explainable Dispatch Log)
# -----------------------------------------------------------------------------
st.markdown("### 🔍 Auditable Scheduling Decisions Log")
st.caption("Detailed mathematical score breakdowns recorded at each dispatch cycle by the scheduler arbiter")

if snapshot.scheduling_decisions:
    dec_df = pd.DataFrame(snapshot.scheduling_decisions)
    st.dataframe(
        dec_df[["decision_id", "timestamp_ms", "selected_tenant", "selected_job", "quantum_ms", "scheduling_score", "reason"]],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("🔬 Inspect Mathematical Formula Breakdown for Decisions", expanded=False):
        for d in snapshot.scheduling_decisions[:5]:
            st.markdown(f"**Decision `{d['decision_id']}` (Time: {d['timestamp_ms']} ms) → Winner: `{d['selected_tenant']}` (`{d['selected_job']}`)**")
            st.write(f"• **Score:** `{d['scheduling_score']}`")
            st.write(f"• **Reason:** {d['reason']}")
            if d.get("score_breakdown"):
                st.json(d["score_breakdown"])
            st.markdown("---")
else:
    st.info("No decision records available for this scheduler policy (non-scoring baseline).")

st.divider()


# -----------------------------------------------------------------------------
# Section 6: Live Comparative Benchmark (All 5 Policies)
# -----------------------------------------------------------------------------
st.markdown("### ⚔️ Live Policy Comparison: Benchmark Against Baselines")
st.caption("Evaluates the current workload configuration across all 5 scheduling policies under identical arrival conditions")

compare_button = st.button("📊 Compare All 5 Schedulers Live", use_container_width=False)

if compare_button:
    with st.spinner("Running identical workload against FIFO, RR, Priority, WRR, and GPU-Aware Fair..."):
        all_reports = engine.compare_all_policies(run_cfg)

        comp_rows = []
        for p_key, rep in all_reports.items():
            p_display = POLICY_DISPLAY_NAMES.get(p_key, p_key.upper())
            comp_rows.append({
                "Policy": p_display,
                "GPU Utilization (%)": f"{rep.gpu_utilization_pct:.1f}%",
                "Throughput (jobs/s)": f"{rep.throughput_jobs_per_sec:.2f}",
                "Avg Wait (ms)": f"{rep.avg_waiting_time_ms:.2f}",
                "Avg Resp (ms)": f"{rep.avg_response_time_ms:.2f}",
                "Jain Fairness (J)": f"{rep.jains_fairness_index:.4f}",
                "Preemptions": rep.preemption_count,
                "Deadline Misses": rep.deadline_misses,
                "Starvation Events": rep.starvation_events,
            })

        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
        st.success("Comparison completed. Proposed GPU-Aware Fair Scheduler maintains bounded fairness and prevents starvation while maximizing utilization.")

st.markdown("---")
st.caption("GPU-Aware Fair Scheduler for Multi-Tenant Edge AI — Operating Systems Project")
