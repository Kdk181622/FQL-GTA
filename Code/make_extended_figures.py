"""Generates the new figures added in this revision: the ablation-study bars and
learning curves, the multi-environment generalization comparison, and the
4-panel training-diagnostics figure (moving-average reward, TD-error evolution,
epsilon decay, state-visitation heatmap). Reads only from results/ -- no
experiments are re-run here.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "Results"
FIG_DIR = HERE.parent / "Figures"
FIG_DIR.mkdir(exist_ok=True)

BASELINE_COLOR = "#4C72B0"
GTA_COLOR = "#DD8452"
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})


def moving_average(x, window=20):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def fig_ablation():
    summary = pd.read_csv(RESULTS_DIR / "ablation_summary.csv")
    curves = pd.read_csv(RESULTS_DIR / "ablation_learning_curves.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ax = axes[0]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    ax.bar(summary["label"], summary["avg_reward"], yerr=summary["std_dev"],
           capsize=6, color=colors, edgecolor="black", linewidth=0.6)
    for i, (m, s) in enumerate(zip(summary["avg_reward"], summary["std_dev"])):
        ax.text(i, m + s + 5, f"{m:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("Average episode reward (full run, 3 seeds)")
    ax.set_title("Ablation study: contribution of each design choice")
    ax.tick_params(axis="x", rotation=15)

    ax = axes[1]
    variant_cols = {
        "baseline": ("Baseline", "#4C72B0"),
        "no_eligibility": ("No Eligibility Trace", "#DD8452"),
        "no_gaussian": ("No Gaussian MF", "#55A868"),
        "full_gta": ("Full FQL-GTA", "#C44E52"),
    }
    for variant, (label, color) in variant_cols.items():
        seed_cols = [c for c in curves.columns if c.startswith(f"{variant}__seed")]
        mean_curve = curves[seed_cols].mean(axis=1)
        ax.plot(curves["step"], mean_curve, label=label, color=color, linewidth=1.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return (mean across 3 seeds)")
    ax.set_title("Ablation learning curves")
    ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_ablation.png", bbox_inches="tight")
    plt.close(fig)


def fig_incremental_ablation():
    summary = pd.read_csv(RESULTS_DIR / "incremental_ablation_summary.csv")
    curves = pd.read_csv(RESULTS_DIR / "incremental_ablation_learning_curves.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    colors = ["#4C72B0", "#55A868", "#8172B2", "#C44E52"]

    ax = axes[0]
    ax.plot(summary["label"], summary["final_reward_mean"], marker="o",
            color="#333333", linewidth=1.5, zorder=1)
    ax.errorbar(summary["label"], summary["final_reward_mean"], yerr=summary["final_reward_std"],
                fmt="none", ecolor="black", capsize=6, zorder=2)
    ax.scatter(summary["label"], summary["final_reward_mean"], color=colors, s=110,
               edgecolor="black", linewidth=0.8, zorder=3)
    for i, (m, s) in enumerate(zip(summary["final_reward_mean"], summary["final_reward_std"])):
        ax.text(i, m + s + 8, f"{m:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("Trailing-50 return (3 seeds)")
    ax.set_title("Incremental ablation: build-up path")
    ax.tick_params(axis="x", rotation=20)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")

    ax = axes[1]
    variant_cols = {
        "baseline": ("Baseline FQL", "#4C72B0"),
        "gaussian_only": ("+ Gaussian MF", "#55A868"),
        "gaussian_eligibility": ("+ Eligibility Traces", "#8172B2"),
        "full_gta": ("+ Adaptive epsilon (Full)", "#C44E52"),
    }
    for variant, (label, color) in variant_cols.items():
        seed_cols = [c for c in curves.columns if c.startswith(f"{variant}__seed")]
        mean_curve = curves[seed_cols].mean(axis=1)
        ax.plot(curves["step"], mean_curve, label=label, color=color, linewidth=1.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return (mean across 3 seeds)")
    ax.set_title("Incremental ablation learning curves")
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle("Additive path: each step adds exactly one FQL-GTA component to the last", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig11_incremental_ablation.png", bbox_inches="tight")
    plt.close(fig)


def fig_multi_env():
    summary = pd.read_csv(RESULTS_DIR / "multi_env_summary.csv")
    envs = summary["env"].unique()

    fig, axes = plt.subplots(1, len(envs), figsize=(4.2 * len(envs), 4.4))
    for ax, env in zip(axes, envs):
        sub = summary[summary["env"] == env]
        means = [sub[sub["variant"] == v]["trailing50_mean"].values[0] for v in ["baseline", "full_gta"]]
        stds = [sub[sub["variant"] == v]["trailing50_std"].values[0] for v in ["baseline", "full_gta"]]
        ax.bar(["Baseline", "Full FQL-GTA"], means, yerr=stds, capsize=6,
               color=[BASELINE_COLOR, GTA_COLOR], edgecolor="black", linewidth=0.6)
        ax.set_title(env)
        ax.set_ylabel("Trailing-50 mean return")
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + (abs(s) + abs(m) * 0.05), f"{m:.1f}", ha="center", fontsize=9)

    fig.suptitle("Multi-environment generalization (3 seeds per environment)", y=1.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8_multi_env.png", bbox_inches="tight")
    plt.close(fig)


def fig_diagnostics():
    with open(RESULTS_DIR / "diagnostics.json") as f:
        diag = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    for variant, color, label in [("baseline", BASELINE_COLOR, "Baseline"),
                                   ("full_gta", GTA_COLOR, "Full FQL-GTA")]:
        returns = np.array(diag[variant]["returns"])
        ma = moving_average(returns, window=20)
        ax.plot(range(len(ma)), ma, color=color, label=label, linewidth=1.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("20-episode moving average return")
    ax.set_title("Moving-average reward")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[0, 1]
    for variant, color, label in [("baseline", BASELINE_COLOR, "Baseline"),
                                   ("full_gta", GTA_COLOR, "Full FQL-GTA")]:
        td = np.array(diag[variant]["td_errors"])
        ma = moving_average(td, window=20)
        ax.plot(range(len(ma)), ma, color=color, label=label, linewidth=1.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean |TD error| per episode (20-ep moving avg)")
    ax.set_title("TD-error evolution")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1, 0]
    eps_gta = np.array(diag["full_gta"]["epsilons"])
    eps_base = np.array(diag["baseline"]["epsilons"])
    ax.plot(range(len(eps_gta)), eps_gta, color=GTA_COLOR, label="Full FQL-GTA (adaptive)", linewidth=1.8)
    ax.plot(range(len(eps_base)), eps_base, color=BASELINE_COLOR, label="Baseline (fixed)", linewidth=1.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Exploration rate (epsilon)")
    ax.set_title("Epsilon decay curve")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1, 1]
    visited = np.array(diag["full_gta"]["visited_states"])
    # Pole angle (dim 2) vs pole angular velocity (dim 3) -- the two dimensions
    # that dominate the balancing task and carry the most fuzzy-set resolution.
    angle = visited[:, 2]
    angular_vel = visited[:, 3]
    h = ax.hist2d(angle, angular_vel, bins=60, cmap="viridis")
    fig.colorbar(h[3], ax=ax, label="Visit count")
    ax.set_xlabel("Pole angle (rad)")
    ax.set_ylabel("Pole angular velocity (rad/s)")
    ax.set_title("State-visitation heatmap (Full FQL-GTA, 400 episodes)")

    fig.suptitle("Training diagnostics: Baseline vs. Full FQL-GTA on CartPole-v1", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig9_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # fig_ablation() (fig7, 4-condition leave-one-out) is superseded by the
    # journal revision's full 2^3 factorial ablation -- see
    # make_figures_journal.py -> fig_ablation_factorial() (fig14).
    # fig_multi_env() (fig8, baseline-vs-GTA per environment) is superseded by
    # the journal revision's formal zero-shot-vs-tuned transfer study -- see
    # make_figures_journal.py -> fig_cross_env_transfer() (fig16).
    fig_incremental_ablation()
    fig_diagnostics()
    print("Extended figures written to", FIG_DIR)
