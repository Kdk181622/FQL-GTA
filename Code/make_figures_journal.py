"""Generates the figures added for the journal revision: the full 8-condition
2^3-factorial ablation, the hyperparameter sensitivity sweeps, the cross-
environment zero-shot-vs-tuned transfer comparison, and the RL-baselines
(FQL/FQL-GTA vs. hand-rolled DQN vs. stable-baselines3 DQN/PPO) comparison.
Reads only from results/ -- no experiments are re-run here.
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

plt.rcParams.update({"figure.dpi": 130, "font.size": 10})
BASELINE_COLOR = "#4C72B0"
GTA_COLOR = "#DD8452"


def fig_ablation_factorial():
    summary = pd.read_csv(RESULTS_DIR / "ablation_summary.csv")
    curves = pd.read_csv(RESULTS_DIR / "ablation_learning_curves.csv")
    variant_order = ["baseline", "epsilon_only", "eligibility_only", "no_gaussian",
                      "gaussian_only", "no_eligibility", "gaussian_eligibility", "full_gta"]
    label_map = dict(zip(
        ["Off / Off / Off (baseline FQL)", "Off / Off / On", "Off / On / Off", "Off / On / On",
         "On / Off / Off", "On / Off / On", "On / On / Off", "On / On / On (full FQL-GTA)"],
        variant_order,
    ))
    summary = summary.set_index("label").loc[list(label_map.keys())].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, 8))
    ax.bar(range(8), summary["avg_reward"], yerr=summary["std_dev"], capsize=5,
           color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(8))
    ax.set_xticklabels(summary["label"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Average episode reward (full run, 3 seeds)")
    ax.set_title("2$^3$ factorial ablation: Gaussian MF / Eligibility Trace / Adaptive $\\epsilon$")

    ax = axes[1]
    for variant, label in zip(variant_order, summary["label"]):
        seed_cols = [c for c in curves.columns if c.startswith(f"{variant}__seed")]
        mean_curve = curves[seed_cols].mean(axis=1)
        lw = 2.4 if variant in ("baseline", "full_gta") else 1.2
        alpha = 1.0 if variant in ("baseline", "full_gta") else 0.75
        ax.plot(curves["step"], mean_curve, label=label, linewidth=lw, alpha=alpha)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return (mean across 3 seeds)")
    ax.set_title("Learning curves, all 8 factorial conditions")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig14_ablation_factorial.png", dpi=200)
    plt.close()


def fig_sensitivity():
    summary = pd.read_csv(RESULTS_DIR / "hyperparameter_sensitivity_summary.csv")
    with open(RESULTS_DIR / "hyperparameter_sensitivity_meta.json") as f:
        meta = json.load(f)
    defaults = meta["defaults"]
    titles = {"lam": r"Eligibility trace decay $\lambda$",
              "sigma_mult": r"Gaussian width multiplier $\sigma_{mult}$",
              "epsilon_decay": r"Epsilon decay rate"}

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    for ax, param in zip(axes, ["lam", "sigma_mult", "epsilon_decay"]):
        sub = summary[summary["param"] == param].sort_values("value")
        ax.errorbar(sub["value"], sub["trailing50_mean"], yerr=sub["trailing50_std"],
                    marker="o", capsize=4, color=GTA_COLOR, linewidth=1.8)
        ax.axvline(defaults[param], color="grey", linestyle="--", linewidth=1.2,
                   label=f"CartPole-tuned default ({defaults[param]})")
        ax.set_title(titles[param], fontsize=11)
        ax.set_xlabel("Value")
        ax.set_ylabel("Trailing-50 return (mean $\\pm$ SD, 5 seeds)")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig15_hyperparameter_sensitivity.png", dpi=200)
    plt.close()


def fig_cross_env_transfer():
    summary = pd.read_csv(RESULTS_DIR / "cross_env_transfer_summary.csv")
    envs = summary["env"].unique().tolist()

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(len(envs))
    width = 0.35
    zero_shot = [summary[(summary.env == e) & (summary.condition == "zero_shot")]["trailing50_mean"].iloc[0] for e in envs]
    zero_shot_std = [summary[(summary.env == e) & (summary.condition == "zero_shot")]["trailing50_std"].iloc[0] for e in envs]
    tuned = [summary[(summary.env == e) & (summary.condition == "tuned")]["trailing50_mean"].iloc[0] for e in envs]
    tuned_std = [summary[(summary.env == e) & (summary.condition == "tuned")]["trailing50_std"].iloc[0] for e in envs]

    ax.bar(x - width / 2, zero_shot, width, yerr=zero_shot_std, capsize=5,
           label="Zero-shot (CartPole-tuned)", color=BASELINE_COLOR, edgecolor="black", linewidth=0.6)
    ax.bar(x + width / 2, tuned, width, yerr=tuned_std, capsize=5,
           label="Tuned (per-environment)", color=GTA_COLOR, edgecolor="black", linewidth=0.6)
    solve_thresholds = {"MountainCar-v0": -110.0, "Acrobot-v1": -100.0}
    for i, e in enumerate(envs):
        ax.hlines(solve_thresholds[e], i - 0.4, i + 0.4, color="#2E7D32", linestyle="--", linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.set_ylabel("Trailing-50 episode return (mean $\\pm$ SD, 3 seeds)")
    ax.set_title("Cross-environment transfer: zero-shot vs. per-environment tuning")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig16_cross_env_transfer.png", dpi=200)
    plt.close()


def fig_rl_baselines():
    table = pd.read_csv(RESULTS_DIR / "rl_baselines_comparison.csv").set_index("method")
    order = ["Baseline FQL", "Improved FQL-GTA", "DQN (hand-rolled)",
              "DQN (stable-baselines3)", "PPO (stable-baselines3)"]
    table = table.loc[order]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#8172B2", "#C44E52"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    ax = axes[0]
    ax.bar(range(5), table["final_return_mean"], yerr=table["final_return_std"], capsize=5,
           color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(5)); ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Final trailing-window return (mean $\\pm$ SD, 10 seeds)")
    ax.set_title("Final performance")

    ax = axes[1]
    ax.bar(range(5), table["n_params"], color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(5)); ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Learnable parameters (log scale)")
    ax.set_title("Model size")

    ax = axes[2]
    ax.bar(range(5), table["train_time_sec_mean"], yerr=table["train_time_sec_std"], capsize=5,
           color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(5)); ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean training time per seed (s)")
    ax.set_title("Training cost")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig17_rl_baselines.png", dpi=200)
    plt.close()


def fig_factorial_interactions():
    """Requires results/factorial_effects.csv from run_factorial_effects.py."""
    variant_coding = {
        "baseline": (-1, -1, -1), "epsilon_only": (-1, -1, 1),
        "eligibility_only": (-1, 1, -1), "no_gaussian": (-1, 1, 1),
        "gaussian_only": (1, -1, -1), "no_eligibility": (1, -1, 1),
        "gaussian_eligibility": (1, 1, -1), "full_gta": (1, 1, 1),
    }
    df = pd.read_csv(RESULTS_DIR / "ablation_per_seed.csv")
    df["G"] = df["variant"].map(lambda v: variant_coding[v][0])
    df["T"] = df["variant"].map(lambda v: variant_coding[v][1])
    df["A"] = df["variant"].map(lambda v: variant_coding[v][2])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    pairs = [("G", "T", "Gaussian MF", "Eligibility Trace"),
             ("G", "A", "Gaussian MF", "Adaptive Epsilon"),
             ("T", "A", "Eligibility Trace", "Adaptive Epsilon")]
    colors = {-1: "#4C72B0", 1: "#C44E52"}
    for ax, (f1, f2, l1, l2) in zip(axes, pairs):
        for lvl in [-1, 1]:
            sub = df[df[f2] == lvl].groupby(f1)["trailing50_return"].mean()
            ax.plot(["Off", "On"], [sub[-1], sub[1]], marker="o", linewidth=2.2,
                    color=colors[lvl], label=f"{l2} {'On' if lvl == 1 else 'Off'}")
        ax.set_xlabel(l1)
        ax.set_title(f"{l1} x {l2}")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean trailing-50 return")
    fig.suptitle("Two-way interaction plots (non-parallel lines = interaction)", y=1.03)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig18_factorial_interactions.png", dpi=200, bbox_inches="tight")
    plt.close()


def fig_performance_vs_cost():
    rl = pd.read_csv(RESULTS_DIR / "rl_baselines_comparison.csv").set_index("method")
    base = rl.loc["Baseline FQL"]
    order = ["Baseline FQL", "Improved FQL-GTA", "DQN (hand-rolled)",
             "DQN (stable-baselines3)", "PPO (stable-baselines3)"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#8172B2", "#C44E52"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for m, c in zip(order, colors):
        row = rl.loc[m]
        time_x = row["train_time_sec_mean"] / base["train_time_sec_mean"]
        perf_y = (row["final_return_mean"] - base["final_return_mean"]) / base["final_return_mean"] * 100
        size = 300 + row["n_params"] / rl["n_params"].max() * 1200
        ax.scatter(time_x, perf_y, s=size, color=c, alpha=0.75, edgecolor="black", linewidth=1.2,
                   label=m, zorder=3)
        ax.annotate(m, (time_x, perf_y), textcoords="offset points",
                    xytext=(0, 18 if m != "Baseline FQL" else -22), ha="center", fontsize=9)
    ax.axhline(0, color="grey", linestyle=":", linewidth=1)
    ax.axvline(1, color="grey", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Training-time cost relative to Baseline FQL (log scale)")
    ax.set_ylabel("Final-return improvement over Baseline FQL (%)")
    ax.set_title("Performance gain vs. computational cost\n(marker area proportional to parameter count)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig19_performance_vs_cost.png", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    fig_ablation_factorial()
    fig_sensitivity()
    fig_cross_env_transfer()
    fig_rl_baselines()
    fig_factorial_interactions()
    fig_performance_vs_cost()
    print("Journal figures saved: fig14_ablation_factorial.png, fig15_hyperparameter_sensitivity.png, "
          "fig16_cross_env_transfer.png, fig17_rl_baselines.png, fig18_factorial_interactions.png, "
          "fig19_performance_vs_cost.png")
