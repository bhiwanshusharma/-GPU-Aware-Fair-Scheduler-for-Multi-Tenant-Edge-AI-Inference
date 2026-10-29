"""GPU Gantt-Style Execution Timeline Visualization.

Parses discrete simulation execution traces and renders publication-quality
GPU Gantt timelines showing:
- Time axis (Time →)
- Per-tenant categorical execution lanes
- Individual job time slices [start_time, end_time]
- Slice boundary events: preemption (quantum expired) vs completion
- Context switch overheads and idle gaps
"""

import os
import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence, Any

import matplotlib
matplotlib.use("Agg")  # Headless rendering safe for server/Mac CLI execution
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

logger = logging.getLogger(__name__)

# High-contrast accessible color palette for tenants
TENANT_PALETTE = [
    "#2B6CB0",  # Blue
    "#C53030",  # Crimson Red
    "#2F855A",  # Forest Green
    "#D69E2E",  # Amber / Dark Gold
    "#805AD5",  # Purple
    "#DD6B20",  # Orange
    "#319795",  # Teal
    "#D53F8C",  # Magenta
    "#4A5568",  # Slate
    "#718096",  # Cool Gray
]

# Canonical colors for scheduling algorithms
POLICY_COLORS = {
    "fifo": "#4A5568",
    "round_robin": "#3182CE",
    "priority": "#DD6B20",
    "weighted_round_robin": "#805AD5",
    "gpu_fair": "#2F855A",
}

POLICY_DISPLAY = {
    "fifo": "FIFO",
    "round_robin": "Round Robin",
    "priority": "Priority",
    "weighted_round_robin": "Weighted RR",
    "gpu_fair": "GPU-Aware Fair",
}


@dataclass
class ExecutionSlice:
    """Represents a contiguous GPU execution slice by a single job."""
    tenant_id: str
    job_id: str
    start_time_ms: float
    end_time_ms: float
    duration_ms: float
    event_type: str  # "preemption" | "completion" | "running"


def parse_trace_to_slices(trace_text: str) -> List[ExecutionSlice]:
    """Parses standard discrete simulation execution trace into execution slices.
    
    Extracts:
    - tenant
    - job
    - start time
    - end time
    - preemption (quantum expired)
    - completion
    """
    slices: List[ExecutionSlice] = []
    lines = trace_text.strip().splitlines()

    curr_time = 0.0
    active_tenant: Optional[str] = None
    active_job: Optional[str] = None
    slice_start = 0.0

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        # Time marker: "time=12.5"
        if line.startswith("time="):
            try:
                curr_time = float(line.split("=")[1])
            except (ValueError, IndexError):
                pass
            continue

        # Dispatch or Resume: "tenant=T1 job=J1 dispatched" or "resumed"
        m_disp = re.match(
            r"tenant=([^\s]+)\s+job=([^\s]+)\s+(dispatched|resumed)", line
        )
        if m_disp:
            tenant, job, action = m_disp.groups()
            # If a previous slice was somehow unclosed, record it
            if active_job is not None and curr_time > slice_start:
                slices.append(
                    ExecutionSlice(
                        tenant_id=active_tenant or "unknown",
                        job_id=active_job,
                        start_time_ms=slice_start,
                        end_time_ms=curr_time,
                        duration_ms=curr_time - slice_start,
                        event_type="preemption",
                    )
                )
            active_tenant = tenant
            active_job = job
            slice_start = curr_time
            continue

        # Quantum expiration / Preemption: "J1 quantum expired" or "tenant=T1 job=J1 preempted"
        m_preempt = re.match(r"([^\s]+)\s+quantum expired", line) or re.match(
            r"tenant=([^\s]+)\s+job=([^\s]+)\s+preempted", line
        )
        if m_preempt:
            if active_job is not None and curr_time >= slice_start:
                slices.append(
                    ExecutionSlice(
                        tenant_id=active_tenant or "unknown",
                        job_id=active_job,
                        start_time_ms=slice_start,
                        end_time_ms=curr_time,
                        duration_ms=curr_time - slice_start,
                        event_type="preemption",
                    )
                )
                active_job = None
                active_tenant = None
            continue

        # Job Completion: "tenant=T1 job=J1 completed"
        m_comp = re.match(r"tenant=([^\s]+)\s+job=([^\s]+)\s+completed", line)
        if m_comp:
            tenant, job = m_comp.groups()
            if active_job is not None and curr_time >= slice_start:
                slices.append(
                    ExecutionSlice(
                        tenant_id=tenant,
                        job_id=job,
                        start_time_ms=slice_start,
                        end_time_ms=curr_time,
                        duration_ms=curr_time - slice_start,
                        event_type="completion",
                    )
                )
                active_job = None
                active_tenant = None
            continue

    # Close any trailing active slice at end of trace
    if active_job is not None and curr_time > slice_start:
        slices.append(
            ExecutionSlice(
                tenant_id=active_tenant or "unknown",
                job_id=active_job,
                start_time_ms=slice_start,
                end_time_ms=curr_time,
                duration_ms=curr_time - slice_start,
                event_type="completion",
            )
        )

    return slices


def plot_gpu_gantt(
    slices: List[ExecutionSlice],
    title: str = "GPU Execution Timeline (Gantt)",
    output_path: Optional[str] = None,
    max_time_ms: Optional[float] = None,
    figsize: Tuple[float, float] = (14.0, 6.0),
    dpi: int = 300,
    show_labels: bool = True,
) -> plt.Figure:
    """Renders a single-policy GPU execution timeline in publication-quality Gantt style.
    
    Y-axis: Distinct tenant execution lanes.
    X-axis: Simulation time (ms) →
    Bars  : Job execution duration [start_time, end_time].
    Markers:
    - Preemption (quantum expired): Amber/red boundary + '⏸' tick.
    - Completion: Emerald star '★' marker.
    """
    if not slices:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, "No execution slices in trace", ha="center", va="center")
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        return fig

    # Discover and sort unique tenants consistently
    unique_tenants = sorted(list({s.tenant_id for s in slices}))
    tenant_to_y = {t_id: idx for idx, t_id in enumerate(unique_tenants)}
    tenant_color_map = {
        t_id: TENANT_PALETTE[i % len(TENANT_PALETTE)]
        for i, t_id in enumerate(unique_tenants)
    }

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    bar_height = 0.55
    total_preemptions = 0
    total_completions = 0
    end_times = [s.end_time_ms for s in slices]
    max_trace_time = max(end_times) if end_times else 100.0
    cutoff_time = max_time_ms if max_time_ms is not None else max_trace_time

    for s in slices:
        if s.start_time_ms > cutoff_time:
            continue

        y_pos = tenant_to_y[s.tenant_id]
        dur = min(s.end_time_ms, cutoff_time) - s.start_time_ms
        if dur <= 0:
            continue

        color = tenant_color_map[s.tenant_id]

        # Draw execution interval bar
        ax.barh(
            y=y_pos,
            width=dur,
            left=s.start_time_ms,
            height=bar_height,
            color=color,
            edgecolor="#1A202C",
            linewidth=1.0,
            alpha=0.9,
            zorder=3,
        )

        # Job ID text label inside bar if width allows
        if show_labels and dur >= 2.5:
            # Shorten label if long
            disp_label = s.job_id
            if len(disp_label) > 8:
                disp_label = disp_label.split("_")[-1]
            ax.text(
                s.start_time_ms + dur / 2.0,
                y_pos,
                disp_label,
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
                zorder=4,
            )

        # Event markers at slice completion
        if s.end_time_ms <= cutoff_time:
            if s.event_type == "preemption":
                total_preemptions += 1
                # Red/Amber vertical line and preemption mark
                ax.plot(
                    [s.end_time_ms, s.end_time_ms],
                    [y_pos - bar_height / 2.0, y_pos + bar_height / 2.0],
                    color="#E53E3E",
                    linewidth=2.5,
                    linestyle="--",
                    zorder=5,
                )
                ax.scatter(
                    s.end_time_ms,
                    y_pos + bar_height / 2.0 + 0.08,
                    marker="v",
                    color="#DD6B20",
                    s=45,
                    zorder=6,
                )
            elif s.event_type == "completion":
                total_completions += 1
                # Green star completion mark
                ax.scatter(
                    s.end_time_ms,
                    y_pos,
                    marker="*",
                    color="#48BB78",
                    edgecolor="#22543D",
                    s=120,
                    zorder=6,
                )

    # Lane formatting
    ax.set_yticks(range(len(unique_tenants)))
    ax.set_yticklabels([f"Tenant: {t}" for t in unique_tenants], fontsize=10, fontweight="bold")
    ax.set_ylim(-0.6, len(unique_tenants) - 0.4)
    ax.set_xlim(0, cutoff_time * 1.02)

    # Gridlines and styling
    ax.set_xlabel("Simulation Time (ms) →", fontsize=11, fontweight="bold")
    ax.set_ylabel("Tenant Allocation Lanes", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6, color="#CBD5E0", axis="x")
    ax.set_axisbelow(True)

    # Legend handles
    legend_elements = [
        mpatches.Patch(facecolor=tenant_color_map[t], edgecolor="#1A202C", label=f"Tenant '{t}'")
        for t in unique_tenants
    ]
    legend_elements.append(
        plt.Line2D([0], [0], color="#E53E3E", linestyle="--", linewidth=2, label=f"Preemption ({total_preemptions})")
    )
    legend_elements.append(
        plt.Line2D([0], [0], marker="*", color="#48BB78", markeredgecolor="#22543D", markersize=10, linestyle="None", label=f"Job Completed ({total_completions})")
    )

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.15),
        ncol=min(5, len(legend_elements)),
        fontsize=8.5,
        frameon=True,
        facecolor="#F7FAFC",
        edgecolor="#CBD5E0",
    )

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved Gantt timeline to {output_path}")

    return fig


def plot_multi_scheduler_gantt(
    policy_traces: Dict[str, str],
    scenario_name: str = "Benchmark Workload",
    output_path: Optional[str] = None,
    max_time_ms: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Renders a vertically stacked multi-panel Gantt figure comparing all schedulers.
    
    Directly illustrates OS time-slicing and interleaving across:
    - FIFO (monolithic blocking execution)
    - Round Robin (fixed slicing)
    - Priority (strict tier bias)
    - Weighted RR (weight-proportional slicing)
    - GPU-Aware Fair (dynamic fair aging and responsiveness)
    """
    valid_policies = [p for p in policy_traces.keys() if policy_traces[p].strip()]
    if not valid_policies:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=dpi)
        ax.text(0.5, 0.5, "No traces provided for multi-scheduler Gantt", ha="center", va="center")
        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        return fig

    # Parse all slices first to discover global time bounds and tenants
    parsed_by_policy: Dict[str, List[ExecutionSlice]] = {
        p: parse_trace_to_slices(policy_traces[p]) for p in valid_policies
    }

    all_tenants = sorted(list({
        s.tenant_id
        for p_slices in parsed_by_policy.values()
        for s in p_slices
    }))
    tenant_to_y = {t_id: idx for idx, t_id in enumerate(all_tenants)}
    tenant_colors = {
        t_id: TENANT_PALETTE[i % len(TENANT_PALETTE)]
        for i, t_id in enumerate(all_tenants)
    }

    # Determine global max cutoff time across all traces
    global_max_end = max(
        (max([s.end_time_ms for s in p_slices] or [100.0]) for p_slices in parsed_by_policy.values()),
        default=100.0,
    )
    cutoff = max_time_ms if max_time_ms is not None else global_max_end

    num_policies = len(valid_policies)
    h_per_panel = max(2.2, 0.6 * len(all_tenants) + 1.2)
    fig_h = h_per_panel * num_policies + 1.5
    fig_w = figsize[0] if figsize else 15.0
    fig_size = (fig_w, fig_h)

    fig, axes = plt.subplots(
        nrows=num_policies,
        ncols=1,
        figsize=fig_size,
        dpi=dpi,
        sharex=True,
    )
    if num_policies == 1:
        axes = [axes]

    bar_height = 0.55

    for idx, (p_key, ax) in enumerate(zip(valid_policies, axes)):
        p_slices = parsed_by_policy[p_key]
        p_name = POLICY_DISPLAY.get(p_key, p_key.upper())

        preempt_cnt = 0
        comp_cnt = 0

        for s in p_slices:
            if s.start_time_ms > cutoff:
                continue

            y_pos = tenant_to_y.get(s.tenant_id, 0)
            dur = min(s.end_time_ms, cutoff) - s.start_time_ms
            if dur <= 0:
                continue

            col = tenant_colors.get(s.tenant_id, "#4A5568")

            # Execution slice bar
            ax.barh(
                y=y_pos,
                width=dur,
                left=s.start_time_ms,
                height=bar_height,
                color=col,
                edgecolor="#1A202C",
                linewidth=0.8,
                alpha=0.9,
                zorder=3,
            )

            # Label if long enough
            if dur >= 3.0:
                label_txt = s.job_id.split("_")[-1]
                ax.text(
                    s.start_time_ms + dur / 2.0,
                    y_pos,
                    label_txt,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=7,
                    fontweight="bold",
                    zorder=4,
                )

            # Preemption vs Completion markers
            if s.end_time_ms <= cutoff:
                if s.event_type == "preemption":
                    preempt_cnt += 1
                    ax.plot(
                        [s.end_time_ms, s.end_time_ms],
                        [y_pos - bar_height / 2.0, y_pos + bar_height / 2.0],
                        color="#E53E3E",
                        linewidth=2.0,
                        linestyle="--",
                        zorder=5,
                    )
                elif s.event_type == "completion":
                    comp_cnt += 1
                    ax.scatter(
                        s.end_time_ms,
                        y_pos,
                        marker="*",
                        color="#48BB78",
                        edgecolor="#22543D",
                        s=80,
                        zorder=6,
                    )

        # Format subplot
        ax.set_yticks(range(len(all_tenants)))
        ax.set_yticklabels(all_tenants, fontsize=9, fontweight="bold")
        ax.set_ylim(-0.55, len(all_tenants) - 0.45)
        ax.set_xlim(0, cutoff * 1.01)

        # Policy header banner
        policy_color = POLICY_COLORS.get(p_key, "#2B6CB0")
        ax.set_title(
            f"Policy: {p_name}   |   Completed: {comp_cnt} jobs   |   Preemptions: {preempt_cnt}",
            fontsize=10.5,
            fontweight="bold",
            loc="left",
            color=policy_color,
            pad=4,
        )
        ax.grid(True, linestyle=":", alpha=0.5, color="#CBD5E0", axis="x")
        ax.set_axisbelow(True)

    # Bottom axis label
    axes[-1].set_xlabel("Simulation Time (ms) →", fontsize=11, fontweight="bold")

    # Global Title
    fig.suptitle(
        f"Comparative GPU Execution Timelines: {scenario_name}\n"
        f"(Time-Slicing, Interleaving, and Preemption Comparison across 5 Schedulers)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    # Shared Legend at top
    legend_handles = [
        mpatches.Patch(facecolor=tenant_colors[t], edgecolor="#1A202C", label=f"Tenant '{t}'")
        for t in all_tenants
    ]
    legend_handles.append(
        plt.Line2D([0], [0], color="#E53E3E", linestyle="--", linewidth=2, label="Preemption Boundary")
    )
    legend_handles.append(
        plt.Line2D([0], [0], marker="*", color="#48BB78", markeredgecolor="#22543D", markersize=10, linestyle="None", label="Job Completed")
    )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=min(6, len(legend_handles)),
        fontsize=8.5,
        frameon=True,
        facecolor="#F7FAFC",
        edgecolor="#CBD5E0",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved multi-scheduler Gantt figure to {output_path}")

    return fig
