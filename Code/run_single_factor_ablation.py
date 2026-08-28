"""Single-factor ablation: adds exactly ONE FQL-GTA enhancement to the plain
baseline at a time -- independent of the other two -- to isolate each design
choice's individual contribution without the confounding of the leave-one-out
study (which removes one enhancement from the full method) or the incremental
study (which stacks enhancements cumulatively). Uses the same unified fql_core
engine, environment, seed convention (3 seeds), and episode budget (400) as
run_ablation_study.py so all three ablation views are directly comparable.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fql_core import FuzzyQLearningAgent, ENV_SPECS, train

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "Results"
RESULTS_DIR.mkdir(exist_ok=True)

ENV_NAME = "CartPole-v1"
SEEDS = [0, 1, 2]
EPISODES = 400
SOLVE_THRESHOLD = 475.0

VARIANT_FLAGS = {
    "baseline": dict(use_gaussian=False, use_eligibility=False, use_adaptive_epsilon=False),
    "gaussian_only": dict(use_gaussian=True, use_eligibility=False, use_adaptive_epsilon=False),
    "eligibility_only": dict(use_gaussian=False, use_eligibility=True, use_adaptive_epsilon=False),
    "adaptive_eps_only": dict(use_gaussian=False, use_eligibility=False, use_adaptive_epsilon=True),
    "full_gta": dict(use_gaussian=True, use_eligibility=True, use_adaptive_epsilon=True),
}
LABELS = {
    "baseline": "Baseline",
    "gaussian_only": "+ Gaussian MF only",
    "eligibility_only": "+ Eligibility Trace only",
    "adaptive_eps_only": "+ Adaptive epsilon only",
    "full_gta": "Full FQL-GTA (all three)",
}


def run_all():
    spec = ENV_SPECS[ENV_NAME]
    rows = []
    curves = {}
    for variant, flags in VARIANT_FLAGS.items():
        for seed in SEEDS:
            t0 = time.time()
            agent = FuzzyQLearningAgent(
                state_ranges=spec["state_ranges"], sets_per_dim=spec["sets_per_dim"],
                n_actions=spec["n_actions"], seed=seed, **flags,
            )
            result = train(agent, ENV_NAME, episodes=EPISODES, seed=seed)
            returns = result["returns"]
            trailing = float(returns[-50:].mean())
            solved = trailing >= SOLVE_THRESHOLD
            rows.append({
                "variant": variant, "label": LABELS[variant], "seed": seed,
                "avg_reward": float(returns.mean()),
                "trailing50_return": trailing,
                "std_reward": float(returns.std()),
                "solved": bool(solved),
                "train_time_sec": time.time() - t0,
            })
            curves[f"{variant}__seed{seed}"] = returns.tolist()
            print(f"{LABELS[variant]:28s} | seed {seed} | avg={returns.mean():6.1f} | "
                  f"trailing50={trailing:6.1f} | solved={solved}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "single_factor_ablation_per_seed.csv", index=False)

    summary = df.groupby("label").agg(
        avg_reward=("avg_reward", "mean"),
        std_dev=("avg_reward", "std"),
        trailing50_mean=("trailing50_return", "mean"),
        solve_rate=("solved", "mean"),
        mean_train_time_sec=("train_time_sec", "mean"),
    ).reset_index()
    order = [LABELS[v] for v in VARIANT_FLAGS]
    summary["label"] = pd.Categorical(summary["label"], categories=order, ordered=True)
    summary = summary.sort_values("label").reset_index(drop=True)
    summary.to_csv(RESULTS_DIR / "single_factor_ablation_summary.csv", index=False)

    max_len = max(len(v) for v in curves.values())
    curves_df = pd.DataFrame({"step": range(max_len)})
    for key, hist in curves.items():
        curves_df[key] = pd.Series(hist).reindex(range(max_len))
    curves_df.to_csv(RESULTS_DIR / "single_factor_ablation_learning_curves.csv", index=False)

    print("\n" + summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    t0 = time.time()
    run_all()
    print(f"\nTotal single-factor ablation time: {time.time() - t0:.1f}s")
