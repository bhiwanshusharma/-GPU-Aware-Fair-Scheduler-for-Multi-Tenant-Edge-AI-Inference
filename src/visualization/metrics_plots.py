"""Performance Metrics Plotting Engine for Multi-Tenant GPU Scheduling.

Generates plots for:
2. Tenant GPU allocation (share & absolute breakdown)
3. GPU utilization (%)
4. Jain fairness index (J)
5. Average waiting time (ms)
6. Response time (ms)
7. Throughput (jobs/sec)
8. Preemption count
9. Per-tenant GPU service (ms)
10. Comprehensive comparison between scheduling algorithms
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

logger = logging.getLogger(__name__)

POLICY_COLORS = {
    "fifo": "#4A5568",                  # Slate Gray
    "round_robin": "#3182CE",           # Blue
    "priority": "#DD6B20",              # Orange
    "weighted_round_robin": "#805AD5",  # Purple
    "gpu_fair": "#2F855A",              # Green / Emerald
}

POLICY_DISPLAY = {
    "fifo": "FIFO",
    "round_robin": "Round Robin",
    "priority": "Priority",
    "weighted_round_robin": "Weighted RR",
    "gpu_fair": "GPU-Aware Fair",
}

TENANT_COLORS = [
    "#2B6CB0", "#C53030", "#2F855A", "#D69E2E",
    "#805AD5", "#DD6B20", "#319795", "#D53F8C",
]


def _apply_plot_style(ax: plt.Axes) -> None:
    """Applies consistent academic styling to a matplotlib axis."""
    ax.grid(True, linestyle=":", alpha=0.6, color="#CBD5E0", zorder=1)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4A5568")
    ax.spines["bottom"].set_color("#4A5568")


# =========================================================================
# Plot 2: Tenant GPU Allocation
# =========================================================================
def plot_tenant_gpu_allocation(
    scenario_data: Dict[str, Any],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (11.0, 6.0),
    dpi: int = 300,
) -> plt.Figure:
    """Plot 2: Visualizes tenant GPU compute allocation (percentage share & absolute ms).
    
    Renders stacked bar chart showing the breakdown of GPU service received
    by each tenant across the 5 scheduling algorithms.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    _apply_plot_style(ax)

    policies = list(scenario_data.get("policies", {}).keys())
    if not policies:
        ax.text(0.5, 0.5, "No policy data available", ha="center", va="center")
        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        return fig

    # Discover all tenants
    all_tenants = sorted(list(scenario_data.get("tenants", {}).keys()))
    if not all_tenants:
        # Fallback to inspecting per_tenant_gpu_service_ms
        for p in policies:
            srv = scenario_data["policies"][p]["metrics"].get("per_tenant_gpu_service_ms", {})
            for t in srv.keys():
                if t not in all_tenants:
                    all_tenants.append(t)
        all_tenants.sort()

    x_indices = np.arange(len(policies))
    bar_width = 0.55
    bottom_vals = np.zeros(len(policies))

    # Calculate total service per policy for normalization
    totals = []
    for p in policies:
        srv = scenario_data["policies"][p]["metrics"].get("per_tenant_gpu_service_ms", {})
        totals.append(sum(srv.values()) or 1.0)

    for i, t_id in enumerate(all_tenants):
        shares = []
        for p_idx, p in enumerate(policies):
            srv = scenario_data["policies"][p]["metrics"].get("per_tenant_gpu_service_ms", {})
            amt = srv.get(t_id, 0.0)
            share_pct = (amt / totals[p_idx]) * 100.0 if totals[p_idx] > 0 else 0.0
            shares.append(share_pct)

        shares_arr = np.array(shares)
        color = TENANT_COLORS[i % len(TENANT_COLORS)]
        bars = ax.bar(
            x_indices,
            shares_arr,
            bottom=bottom_vals,
            width=bar_width,
            label=f"Tenant '{t_id}'",
            color=color,
            edgecolor="#FFFFFF",
            linewidth=1.2,
            zorder=3,
        )

        # Label inside segments if tall enough
        for p_idx, (b, share) in enumerate(zip(bars, shares_arr)):
            if share >= 8.0:
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    bottom_vals[p_idx] + share / 2.0,
                    f"{share:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8.5,
                    fontweight="bold",
                    zorder=4,
                )

        bottom_vals += shares_arr

    ax.set_xticks(x_indices)
    ax.set_xticklabels([POLICY_DISPLAY.get(p, p.upper()) for p in policies], fontsize=10, fontweight="bold")
    ax.set_ylabel("GPU Allocation Share (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)

    sc_name = scenario_data.get("scenario_name", scenario_data.get("scenario_id", "Scenario"))
    ax.set_title(
        f"Tenant GPU Allocation Share Breakdown: {sc_name}",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.13),
        ncol=min(4, len(all_tenants)),
        frameon=True,
        facecolor="#F7FAFC",
        edgecolor="#CBD5E0",
        fontsize=9,
    )

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved tenant GPU allocation plot to {output_path}")

    return fig


# =========================================================================
# Plot 3: GPU Utilization
# =========================================================================
def plot_gpu_utilization(
    suite_data: Dict[str, Any],
    output_path: Optional[str] = None,
    scenario_id: Optional[str] = None,
    figsize: Tuple[float, float] = (11.0, 5.5),
    dpi: int = 300,
) -> plt.Figure:
    """Plot 3: Compares GPU Utilization (%) across scheduling policies."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    _apply_plot_style(ax)

    scenarios = suite_data.get("scenarios", {})
    if scenario_id and scenario_id in scenarios:
        target_scenarios = {scenario_id: scenarios[scenario_id]}
    else:
        target_scenarios = scenarios

    if not target_scenarios:
        ax.text(0.5, 0.5, "No scenario data available", ha="center", va="center")
        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        return fig

    # If single scenario, simple bar plot
    if len(target_scenarios) == 1:
        s_data = list(target_scenarios.values())[0]
        policies = list(s_data.get("policies", {}).keys())
        utils = [
            s_data["policies"][p]["metrics"]["summary"].get("gpu_utilization_pct", 0.0)
            for p in policies
        ]
        colors = [POLICY_COLORS.get(p, "#3182CE") for p in policies]

        bars = ax.bar(
            range(len(policies)),
            utils,
            width=0.5,
            color=colors,
            edgecolor="#1A202C",
            linewidth=1.0,
            zorder=3,
        )
        for b, val in zip(bars, utils):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                val + 1.2,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )
        ax.set_xticks(range(len(policies)))
        ax.set_xticklabels([POLICY_DISPLAY.get(p, p.upper()) for p in policies], fontsize=10, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.set_ylabel("GPU Utilization (%)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"GPU Compute Utilization Comparison: {s_data.get('scenario_name', 'Scenario')}",
            fontsize=12.5,
            fontweight="bold",
            pad=12,
        )
    else:
        # Multi-scenario grouped bar plot
        sc_keys = list(target_scenarios.keys())[:8]  # up to 8 scenarios for readability
        first_sc = list(target_scenarios.values())[0]
        policies = list(first_sc.get("policies", {}).keys())

        x = np.arange(len(sc_keys))
        n_p = len(policies)
        width = 0.8 / max(1, n_p)

        for i, p in enumerate(policies):
            vals = [
                target_scenarios[s]["policies"].get(p, {}).get("metrics", {}).get("summary", {}).get("gpu_utilization_pct", 0.0)
                for s in sc_keys
            ]
            col = POLICY_COLORS.get(p, "#3182CE")
            ax.bar(
                x + i * width - (n_p - 1) * width / 2.0,
                vals,
                width=width,
                label=POLICY_DISPLAY.get(p, p.upper()),
                color=col,
                edgecolor="#1A202C",
                linewidth=0.7,
                zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("scenario_", "S") for s in sc_keys], fontsize=9.5, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.set_ylabel("GPU Utilization (%)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Benchmark Scenarios", fontsize=11, fontweight="bold")
        ax.set_title("GPU Compute Utilization Across Evaluated Workload Scenarios", fontsize=12.5, fontweight="bold", pad=12)
        ax.legend(loc="upper right", ncol=min(5, n_p), frameon=True, fontsize=8.5)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved GPU utilization plot to {output_path}")

    return fig


# =========================================================================
# Plot 4: Jain Fairness Index
# =========================================================================
def plot_jain_fairness(
    suite_data: Dict[str, Any],
    output_path: Optional[str] = None,
    scenario_id: Optional[str] = None,
    figsize: Tuple[float, float] = (11.0, 5.5),
    dpi: int = 300,
) -> plt.Figure:
    """Plot 4: Compares Jain's Fairness Index across scheduling algorithms."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    _apply_plot_style(ax)

    scenarios = suite_data.get("scenarios", {})
    if scenario_id and scenario_id in scenarios:
        target_scenarios = {scenario_id: scenarios[scenario_id]}
    else:
        target_scenarios = scenarios

    if not target_scenarios:
        ax.text(0.5, 0.5, "No scenario data available", ha="center", va="center")
        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        return fig

    if len(target_scenarios) == 1:
        s_data = list(target_scenarios.values())[0]
        policies = list(s_data.get("policies", {}).keys())
        jains = [
            s_data["policies"][p]["metrics"]["summary"].get("jains_fairness_index", 1.0)
            for p in policies
        ]
        colors = [POLICY_COLORS.get(p, "#3182CE") for p in policies]

        bars = ax.bar(
            range(len(policies)),
            jains,
            width=0.5,
            color=colors,
            edgecolor="#1A202C",
            linewidth=1.0,
            zorder=3,
        )
        for b, val in zip(bars, jains):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                val + 0.02,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )
        # Optimal fairness reference line
        ax.axhline(1.0, color="#E53E3E", linestyle="--", linewidth=1.5, label="Optimal Fairness (J = 1.0)")

        ax.set_xticks(range(len(policies)))
        ax.set_xticklabels([POLICY_DISPLAY.get(p, p.upper()) for p in policies], fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Jain's Fairness Index (J)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"Jain's Fairness Index Comparison: {s_data.get('scenario_name', 'Scenario')}",
            fontsize=12.5,
            fontweight="bold",
            pad=12,
        )
        ax.legend(loc="lower right", frameon=True, fontsize=9)
    else:
        sc_keys = list(target_scenarios.keys())[:8]
        first_sc = list(target_scenarios.values())[0]
        policies = list(first_sc.get("policies", {}).keys())

        x = np.arange(len(sc_keys))
        n_p = len(policies)
        width = 0.8 / max(1, n_p)

        for i, p in enumerate(policies):
            vals = [
                target_scenarios[s]["policies"].get(p, {}).get("metrics", {}).get("summary", {}).get("jains_fairness_index", 1.0)
                for s in sc_keys
            ]
            col = POLICY_COLORS.get(p, "#3182CE")
            ax.bar(
                x + i * width - (n_p - 1) * width / 2.0,
                vals,
                width=width,
                label=POLICY_DISPLAY.get(p, p.upper()),
                color=col,
                edgecolor="#1A202C",
                linewidth=0.7,
                zorder=3,
            )

        ax.axhline(1.0, color="#E53E3E", linestyle="--", linewidth=1.5, label="Optimal (J = 1.0)")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("scenario_", "S") for s in sc_keys], fontsize=9.5, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Jain's Fairness Index (J)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Benchmark Scenarios", fontsize=11, fontweight="bold")
        ax.set_title("Jain's Fairness Index Comparison Across Evaluated Workloads", fontsize=12.5, fontweight="bold", pad=12)
        ax.legend(loc="lower right", ncol=min(6, n_p + 1), frameon=True, fontsize=8.5)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved Jain fairness plot to {output_path}")

    return fig


# =========================================================================
# Plot 5: Average Waiting Time
# =========================================================================
def plot_waiting_time(
    suite_data: Dict[str, Any],
    output_path: Optional[str] = None,
    scenario_id: Optional[str] = None,
    figsize: Tuple[float, float] = (11.0, 5.5),
    dpi: int = 300,
) -> plt.Figure:
    """Plot 5: Compares Average Queue Waiting Time (ms) across algorithms."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    _apply_plot_style(ax)

    scenarios = suite_data.get("scenarios", {})
    if scenario_id and scenario_id in scenarios:
        target_scenarios = {scenario_id: scenarios[scenario_id]}
    else:
        target_scenarios = scenarios

    if not target_scenarios:
        ax.text(0.5, 0.5, "No scenario data available", ha="center", va="center")
        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        return fig

    if len(target_scenarios) == 1:
        s_data = list(target_scenarios.values())[0]
        policies = list(s_data.get("policies", {}).keys())
        wait_times = [
            s_data["policies"][p]["metrics"]["summary"].get("avg_waiting_time_ms", 0.0)
            for p in policies
        ]
        colors = [POLICY_COLORS.get(p, "#3182CE") for p in policies]

        bars = ax.bar(
            range(len(policies)),
            wait_times,
            width=0.5,
            color=colors,
            edgecolor="#1A202C",
            linewidth=1.0,
            zorder=3,
        )
        max_val = max(wait_times) if wait_times else 10.0
        for b, val in zip(bars, wait_times):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                val + max_val * 0.02,
                f"{val:.1f} ms",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )
        ax.set_xticks(range(len(policies)))
        ax.set_xticklabels([POLICY_DISPLAY.get(p, p.upper()) for p in policies], fontsize=10, fontweight="bold")
        ax.set_ylim(0, max_val * 1.18)
        ax.set_ylabel("Average Waiting Time (ms)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"Mean Queue Waiting Latency: {s_data.get('scenario_name', 'Scenario')}",
            fontsize=12.5,
            fontweight="bold",
            pad=12,
        )
    else:
        sc_keys = list(target_scenarios.keys())[:8]
        first_sc = list(target_scenarios.values())[0]
        policies = list(first_sc.get("policies", {}).keys())

        x = np.arange(len(sc_keys))
        n_p = len(policies)
        width = 0.8 / max(1, n_p)

        for i, p in enumerate(policies):
            vals = [
                target_scenarios[s]["policies"].get(p, {}).get("metrics", {}).get("summary", {}).get("avg_waiting_time_ms", 0.0)
                for s in sc_keys
            ]
            col = POLICY_COLORS.get(p, "#3182CE")
            ax.bar(
                x + i * width - (n_p - 1) * width / 2.0,
                vals,
                width=width,
                label=POLICY_DISPLAY.get(p, p.upper()),
                color=col,
                edgecolor="#1A202C",
                linewidth=0.7,
                zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("scenario_", "S") for s in sc_keys], fontsize=9.5, fontweight="bold")
        ax.set_ylabel("Average Waiting Time (ms)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Benchmark Scenarios", fontsize=11, fontweight="bold")
        ax.set_title("Average Queue Waiting Latency Across Evaluated Workloads", fontsize=12.5, fontweight="bold", pad=12)
        ax.legend(loc="upper right", ncol=min(5, n_p), frameon=True, fontsize=8.5)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved waiting time plot to {output_path}")

    return fig


# =========================================================================
# Plot 6: Response Time
# =========================================================================
def plot_response_time(
    suite_data: Dict[str, Any],
    output_path: Optional[str] = None,
    scenario_id: Optional[str] = None,
    figsize: Tuple[float, float] = (11.0, 5.5),
    dpi: int = 300,
) -> plt.Figure:
    """Plot 6: Compares Average Response Time (ms) from arrival to initial dispatch."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    _apply_plot_style(ax)

    scenarios = suite_data.get("scenarios", {})
    if scenario_id and scenario_id in scenarios:
        target_scenarios = {scenario_id: scenarios[scenario_id]}
    else:
        target_scenarios = scenarios

    if not target_scenarios:
        ax.text(0.5, 0.5, "No scenario data available", ha="center", va="center")
        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        return fig

    if len(target_scenarios) == 1:
        s_data = list(target_scenarios.values())[0]
        policies = list(s_data.get("policies", {}).keys())
        resp_times = [
            s_data["policies"][p]["metrics"]["summary"].get("avg_response_time_ms", 0.0)
            for p in policies
        ]
        colors = [POLICY_COLORS.get(p, "#3182CE") for p in policies]

        bars = ax.bar(
            range(len(policies)),
            resp_times,
            width=0.5,
            color=colors,
            edgecolor="#1A202C",
            linewidth=1.0,
            zorder=3,
        )
        max_val = max(resp_times) if resp_times else 10.0
        for b, val in zip(bars, resp_times):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                val + max_val * 0.02,
                f"{val:.1f} ms",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )
        ax.set_xticks(range(len(policies)))
        ax.set_xticklabels([POLICY_DISPLAY.get(p, p.upper()) for p in policies], fontsize=10, fontweight="bold")
        ax.set_ylim(0, max_val * 1.18)
        ax.set_ylabel("Average Response Time (ms)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"Mean Job Initial Response Latency: {s_data.get('scenario_name', 'Scenario')}",
            fontsize=12.5,
            fontweight="bold",
            pad=12,
        )
    else:
        sc_keys = list(target_scenarios.keys())[:8]
        first_sc = list(target_scenarios.values())[0]
        policies = list(first_sc.get("policies", {}).keys())

        x = np.arange(len(sc_keys))
        n_p = len(policies)
        width = 0.8 / max(1, n_p)

        for i, p in enumerate(policies):
            vals = [
                target_scenarios[s]["policies"].get(p, {}).get("metrics", {}).get("summary", {}).get("avg_response_time_ms", 0.0)
                for s in sc_keys
            ]
            col = POLICY_COLORS.get(p, "#3182CE")
            ax.bar(
                x + i * width - (n_p - 1) * width / 2.0,
                vals,
                width=width,
                label=POLICY_DISPLAY.get(p, p.upper()),
                color=col,
                edgecolor="#1A202C",
                linewidth=0.7,
                zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("scenario_", "S") for s in sc_keys], fontsize=9.5, fontweight="bold")
        ax.set_ylabel("Average Response Time (ms)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Benchmark Scenarios", fontsize=11, fontweight="bold")
        ax.set_title("Average Initial Response Latency Across Evaluated Workloads", fontsize=12.5, fontweight="bold", pad=12)
        ax.legend(loc="upper right", ncol=min(5, n_p), frameon=True, fontsize=8.5)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved response time plot to {output_path}")

    return fig


# =========================================================================
# Plot 7: Throughput
# =========================================================================
def plot_throughput(
    suite_data: Dict[str, Any],
    output_path: Optional[str] = None,
    scenario_id: Optional[str] = None,
    figsize: Tuple[float, float] = (11.0, 5.5),
    dpi: int = 300,
) -> plt.Figure:
    """Plot 7: Compares Job Throughput (jobs/sec) across algorithms."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    _apply_plot_style(ax)

    scenarios = suite_data.get("scenarios", {})
    if scenario_id and scenario_id in scenarios:
        target_scenarios = {scenario_id: scenarios[scenario_id]}
    else:
        target_scenarios = scenarios

    if not target_scenarios:
        ax.text(0.5, 0.5, "No scenario data available", ha="center", va="center")
        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        return fig

    if len(target_scenarios) == 1:
        s_data = list(target_scenarios.values())[0]
        policies = list(s_data.get("policies", {}).keys())
        tputs = [
            s_data["policies"][p]["metrics"]["summary"].get("throughput_jobs_per_sec", 0.0)
            for p in policies
        ]
        colors = [POLICY_COLORS.get(p, "#3182CE") for p in policies]

        bars = ax.bar(
            range(len(policies)),
            tputs,
            width=0.5,
            color=colors,
            edgecolor="#1A202C",
            linewidth=1.0,
            zorder=3,
        )
        max_val = max(tputs) if tputs else 10.0
        for b, val in zip(bars, tputs):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                val + max_val * 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )
        ax.set_xticks(range(len(policies)))
        ax.set_xticklabels([POLICY_DISPLAY.get(p, p.upper()) for p in policies], fontsize=10, fontweight="bold")
        ax.set_ylim(0, max_val * 1.18)
        ax.set_ylabel("Throughput (jobs/sec)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"System Inference Throughput: {s_data.get('scenario_name', 'Scenario')}",
            fontsize=12.5,
            fontweight="bold",
            pad=12,
        )
    else:
        sc_keys = list(target_scenarios.keys())[:8]
        first_sc = list(target_scenarios.values())[0]
        policies = list(first_sc.get("policies", {}).keys())

        x = np.arange(len(sc_keys))
        n_p = len(policies)
        width = 0.8 / max(1, n_p)

        for i, p in enumerate(policies):
            vals = [
                target_scenarios[s]["policies"].get(p, {}).get("metrics", {}).get("summary", {}).get("throughput_jobs_per_sec", 0.0)
                for s in sc_keys
            ]
            col = POLICY_COLORS.get(p, "#3182CE")
            ax.bar(
                x + i * width - (n_p - 1) * width / 2.0,
                vals,
                width=width,
                label=POLICY_DISPLAY.get(p, p.upper()),
                color=col,
                edgecolor="#1A202C",
                linewidth=0.7,
                zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("scenario_", "S") for s in sc_keys], fontsize=9.5, fontweight="bold")
        ax.set_ylabel("Throughput (jobs/sec)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Benchmark Scenarios", fontsize=11, fontweight="bold")
        ax.set_title("System Inference Throughput Across Evaluated Workloads", fontsize=12.5, fontweight="bold", pad=12)
        ax.legend(loc="upper right", ncol=min(5, n_p), frameon=True, fontsize=8.5)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved throughput plot to {output_path}")

    return fig


# =========================================================================
# Plot 8: Preemption Count
# =========================================================================
def plot_preemption_count(
    suite_data: Dict[str, Any],
    output_path: Optional[str] = None,
    scenario_id: Optional[str] = None,
    figsize: Tuple[float, float] = (11.0, 5.5),
    dpi: int = 300,
) -> plt.Figure:
    """Plot 8: Compares total Preemption Counts incurred across algorithms."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    _apply_plot_style(ax)

    scenarios = suite_data.get("scenarios", {})
    if scenario_id and scenario_id in scenarios:
        target_scenarios = {scenario_id: scenarios[scenario_id]}
    else:
        target_scenarios = scenarios

    if not target_scenarios:
        ax.text(0.5, 0.5, "No scenario data available", ha="center", va="center")
        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        return fig

    if len(target_scenarios) == 1:
        s_data = list(target_scenarios.values())[0]
        policies = list(s_data.get("policies", {}).keys())
        preempts = [
            s_data["policies"][p]["metrics"]["summary"].get("preemption_count", 0)
            for p in policies
        ]
        colors = [POLICY_COLORS.get(p, "#3182CE") for p in policies]

        bars = ax.bar(
            range(len(policies)),
            preempts,
            width=0.5,
            color=colors,
            edgecolor="#1A202C",
            linewidth=1.0,
            zorder=3,
        )
        max_val = max(preempts) if preempts else 1
        for b, val in zip(bars, preempts):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                val + max(0.2, max_val * 0.02),
                str(val),
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )
        ax.set_xticks(range(len(policies)))
        ax.set_xticklabels([POLICY_DISPLAY.get(p, p.upper()) for p in policies], fontsize=10, fontweight="bold")
        ax.set_ylim(0, max_val * 1.25 + 0.5)
        ax.set_ylabel("Number of Preemptions", fontsize=11, fontweight="bold")
        ax.set_title(
            f"Preemption Frequency Comparison: {s_data.get('scenario_name', 'Scenario')}",
            fontsize=12.5,
            fontweight="bold",
            pad=12,
        )
    else:
        sc_keys = list(target_scenarios.keys())[:8]
        first_sc = list(target_scenarios.values())[0]
        policies = list(first_sc.get("policies", {}).keys())

        x = np.arange(len(sc_keys))
        n_p = len(policies)
        width = 0.8 / max(1, n_p)

        for i, p in enumerate(policies):
            vals = [
                target_scenarios[s]["policies"].get(p, {}).get("metrics", {}).get("summary", {}).get("preemption_count", 0)
                for s in sc_keys
            ]
            col = POLICY_COLORS.get(p, "#3182CE")
            ax.bar(
                x + i * width - (n_p - 1) * width / 2.0,
                vals,
                width=width,
                label=POLICY_DISPLAY.get(p, p.upper()),
                color=col,
                edgecolor="#1A202C",
                linewidth=0.7,
                zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("scenario_", "S") for s in sc_keys], fontsize=9.5, fontweight="bold")
        ax.set_ylabel("Preemptions Count", fontsize=11, fontweight="bold")
        ax.set_xlabel("Benchmark Scenarios", fontsize=11, fontweight="bold")
        ax.set_title("Preemption Frequency Across Evaluated Workload Scenarios", fontsize=12.5, fontweight="bold", pad=12)
        ax.legend(loc="upper right", ncol=min(5, n_p), frameon=True, fontsize=8.5)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved preemption count plot to {output_path}")

    return fig


# =========================================================================
# Plot 9: Per-Tenant GPU Service
# =========================================================================
def plot_per_tenant_service(
    scenario_data: Dict[str, Any],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12.0, 6.0),
    dpi: int = 300,
) -> plt.Figure:
    """Plot 9: Compares absolute GPU service duration (ms) received by each tenant across algorithms."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    _apply_plot_style(ax)

    policies = list(scenario_data.get("policies", {}).keys())
    if not policies:
        ax.text(0.5, 0.5, "No policy data available", ha="center", va="center")
        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        return fig

    all_tenants = sorted(list(scenario_data.get("tenants", {}).keys()))
    if not all_tenants:
        for p in policies:
            srv = scenario_data["policies"][p]["metrics"].get("per_tenant_gpu_service_ms", {})
            for t in srv.keys():
                if t not in all_tenants:
                    all_tenants.append(t)
        all_tenants.sort()

    x = np.arange(len(all_tenants))
    n_p = len(policies)
    width = 0.8 / max(1, n_p)

    max_val = 10.0
    for i, p in enumerate(policies):
        srv = scenario_data["policies"][p]["metrics"].get("per_tenant_gpu_service_ms", {})
        vals = [srv.get(t, 0.0) for t in all_tenants]
        if vals:
            max_val = max(max_val, max(vals))
        col = POLICY_COLORS.get(p, "#3182CE")
        bars = ax.bar(
            x + i * width - (n_p - 1) * width / 2.0,
            vals,
            width=width,
            label=POLICY_DISPLAY.get(p, p.upper()),
            color=col,
            edgecolor="#1A202C",
            linewidth=0.8,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Tenant: {t}" for t in all_tenants], fontsize=10, fontweight="bold")
    ax.set_ylabel("GPU Service Delivered (ms)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max_val * 1.2)

    sc_name = scenario_data.get("scenario_name", scenario_data.get("scenario_id", "Scenario"))
    ax.set_title(
        f"Per-Tenant GPU Compute Service Allocation: {sc_name}",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    ax.legend(
        loc="upper right",
        ncol=min(5, n_p),
        frameon=True,
        facecolor="#F7FAFC",
        edgecolor="#CBD5E0",
        fontsize=9,
    )

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved per-tenant GPU service plot to {output_path}")

    return fig


# =========================================================================
# Plot 10: Comparison Between Scheduling Algorithms (Master Dashboard)
# =========================================================================
def plot_algorithm_comparison(
    scenario_data: Dict[str, Any],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (16.0, 10.0),
    dpi: int = 300,
) -> plt.Figure:
    """Plot 10: Comprehensive Multi-Metric Comparison Dashboard across Schedulers.
    
    Renders an 8-panel publication-quality grid comparing:
    1. GPU Utilization (%)
    2. Throughput (jobs/sec)
    3. Average Waiting Time (ms)
    4. Average Response Time (ms)
    5. Average Turnaround Time (ms)
    6. Jain's Fairness Index (J)
    7. Preemption Count
    8. Scheduling Overhead (ms)
    """
    policies = list(scenario_data.get("policies", {}).keys())
    if not policies:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
        ax.text(0.5, 0.5, "No policy data available", ha="center", va="center")
        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        return fig

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=figsize, dpi=dpi)
    sc_name = scenario_data.get("scenario_name", scenario_data.get("scenario_id", "Scenario"))
    fig.suptitle(
        f"Multi-Metric Comparative Performance Dashboard: {sc_name}\n"
        f"Evaluation of 5 Scheduling Algorithms Under Identical Multi-Tenant Workload",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )

    colors = [POLICY_COLORS.get(p, "#3182CE") for p in policies]
    labels = [POLICY_DISPLAY.get(p, p.upper()) for p in policies]
    x = range(len(policies))

    metric_specs = [
        # (title, key, unit, higher_is_better)
        ("GPU Utilization", "gpu_utilization_pct", "%", True),
        ("Throughput", "throughput_jobs_per_sec", "jobs/s", True),
        ("Avg Waiting Time", "avg_waiting_time_ms", "ms", False),
        ("Avg Response Time", "avg_response_time_ms", "ms", False),
        ("Avg Turnaround Time", "avg_turnaround_time_ms", "ms", False),
        ("Jain's Fairness Index", "jains_fairness_index", "J", True),
        ("Preemptions", "preemption_count", "count", False),
        ("Scheduling Overhead", "scheduling_overhead_ms", "ms", False),
    ]

    for idx, (m_title, m_key, m_unit, higher_better) in enumerate(metric_specs):
        r = idx // 4
        c = idx % 4
        ax = axes[r, c]
        _apply_plot_style(ax)

        vals = [
            scenario_data["policies"][p]["metrics"]["summary"].get(m_key, 0.0)
            for p in policies
        ]

        bars = ax.bar(
            x,
            vals,
            width=0.55,
            color=colors,
            edgecolor="#1A202C",
            linewidth=0.9,
            zorder=3,
        )

        max_v = max(vals) if vals else 1.0
        for b, v in zip(bars, vals):
            fmt_v = f"{v:.1f}" if isinstance(v, float) and v < 1000 else (f"{v:.4f}" if m_key == "jains_fairness_index" else str(int(v)))
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                v + max_v * 0.02,
                fmt_v,
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, fontweight="bold", rotation=20, ha="right")
        ax.set_title(f"{m_title} ({m_unit})", fontsize=10.5, fontweight="bold", pad=8)
        ax.set_ylim(0, max_v * 1.22 if max_v > 0 else 1.0)

        # Highlight optimal direction
        indicator = "▲ Higher is better" if higher_better else "▼ Lower is better"
        ax.text(
            0.03,
            0.92,
            indicator,
            transform=ax.transAxes,
            fontsize=7.5,
            color="#718096",
            fontstyle="italic",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved comprehensive algorithm comparison plot to {output_path}")

    return fig
