"""Incremental (additive) ablation study on CartPole-v1: unlike
run_ablation_study.py's leave-one-out design (which removes one component from
the full method at a time), this adds one component at a time on top of the
unmodified baseline -- Baseline -> +Gaussian MF -> +Eligibility Traces ->
+Adaptive epsilon (= Full FQL-GTA) -- to show the build-up path some readers
find more intuitive. Both designs answer slightly different questions and are
reported side by side rather than one replacing the other.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fql_core import make_agent, train

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "Results"
RESULTS_DIR.mkdir(exist_ok=True)

ENV_NAME = "CartPole-v1"
SEEDS = [0, 1, 2]
EPISODES = 400
STEP_LABELS = {
    "baseline": "Baseline FQL",
    "gaussian_only": "+ Gaussian MF",
    "gaussian_eligibility": "+ Eligibility Traces",
    "full_gta": "+ Adaptive epsilon (Full FQL-GTA)",
}


def run_all():
    rows = []
    curves = {}
    for variant, label in STEP_LABELS.items():
        for seed in SEEDS:
            t0 = time.time()
            agent = make_agent(ENV_NAME, variant=variant, seed=seed)
            result = train(agent, ENV_NAME, episodes=EPISODES, seed=seed)
            returns = result["returns"]
            rows.append({
                "variant": variant, "label": label, "seed": seed,
                "final_reward": float(returns[-50:].mean()),
                "avg_reward": float(returns.mean()),
                "train_time_sec": time.time() - t0,
            })
            curves[f"{variant}__seed{seed}"] = returns.tolist()
            print(f"{label} | seed {seed} | final_reward(trailing50)={returns[-50:].mean():.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "incremental_ablation_per_seed.csv", index=False)

    summary = df.groupby("label").agg(
        final_reward_mean=("final_reward", "mean"), final_reward_std=("final_reward", "std"),
        avg_reward_mean=("avg_reward", "mean"),
    ).reset_index()
    order = list(STEP_LABELS.values())
    summary["label"] = pd.Categorical(summary["label"], categories=order, ordered=True)
    summary = summary.sort_values("label").reset_index(drop=True)
    summary.to_csv(RESULTS_DIR / "incremental_ablation_summary.csv", index=False)

    max_len = max(len(v) for v in curves.values())
    curves_df = pd.DataFrame({"step": range(max_len)})
    for key, hist in curves.items():
        curves_df[key] = pd.Series(hist).reindex(range(max_len))
    curves_df.to_csv(RESULTS_DIR / "incremental_ablation_learning_curves.csv", index=False)

    print("\n" + summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    t0 = time.time()
    run_all()
    print(f"\nTotal incremental ablation time: {time.time() - t0:.1f}s")
