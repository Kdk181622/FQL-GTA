"""Hyperparameter sensitivity analysis for FQL-GTA on CartPole-v1: one-factor-
-at-a-time sweeps over the eligibility-trace decay (lambda), the Gaussian
membership-function width multiplier (sigma_mult), and the epsilon-decay rate,
holding the other two at their CartPole-tuned defaults (lambda=0.5,
sigma_mult=0.35, epsilon_decay=0.99). Grids and seed count match the frozen
protocol in configs/default_config.json -> hyperparameter_sensitivity.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fql_core import FuzzyQLearningAgent, ENV_SPECS, train

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "Results"
RESULTS_DIR.mkdir(exist_ok=True)

ENV_NAME = "CartPole-v1"
SEEDS = [0, 1, 2, 3, 4]
EPISODES = 400

DEFAULTS = dict(lam=0.5, sigma_mult=0.35, epsilon_decay=0.99)
SWEEPS = {
    "lam": [0.0, 0.3, 0.5, 0.7, 0.9, 0.99],
    "sigma_mult": [0.2, 0.35, 0.5, 0.7],
    "epsilon_decay": [0.90, 0.95, 0.99, 0.995, 0.999],
}


def run_condition(param_name, value, seed):
    spec = ENV_SPECS[ENV_NAME]
    kwargs = dict(DEFAULTS)
    kwargs[param_name] = value
    agent = FuzzyQLearningAgent(
        state_ranges=spec["state_ranges"], sets_per_dim=spec["sets_per_dim"],
        n_actions=spec["n_actions"], seed=seed,
        use_gaussian=True, use_eligibility=True, use_adaptive_epsilon=True,
        **kwargs,
    )
    result = train(agent, ENV_NAME, episodes=EPISODES, seed=seed)
    return result["returns"]


def run_all():
    rows = []
    for param_name, values in SWEEPS.items():
        for value in values:
            for seed in SEEDS:
                t0 = time.time()
                returns = run_condition(param_name, value, seed)
                trailing = float(returns[-50:].mean())
                rows.append({
                    "param": param_name, "value": value, "seed": seed,
                    "avg_reward": float(returns.mean()),
                    "trailing50_return": trailing,
                    "train_time_sec": time.time() - t0,
                })
                print(f"{param_name}={value} | seed {seed} | avg={returns.mean():.1f} | "
                      f"trailing50={trailing:.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "hyperparameter_sensitivity_per_seed.csv", index=False)

    summary = df.groupby(["param", "value"]).agg(
        avg_reward_mean=("avg_reward", "mean"),
        avg_reward_std=("avg_reward", "std"),
        trailing50_mean=("trailing50_return", "mean"),
        trailing50_std=("trailing50_return", "std"),
    ).reset_index()
    summary.to_csv(RESULTS_DIR / "hyperparameter_sensitivity_summary.csv", index=False)

    with open(RESULTS_DIR / "hyperparameter_sensitivity_meta.json", "w") as f:
        json.dump({"defaults": DEFAULTS, "sweeps": SWEEPS, "seeds": SEEDS,
                    "episodes_per_seed": EPISODES}, f, indent=2)

    print("\n" + summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    t0 = time.time()
    run_all()
    print(f"\nTotal hyperparameter sensitivity time: {time.time() - t0:.1f}s")
