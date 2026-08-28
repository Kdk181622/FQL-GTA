"""Ablation study: isolates the individual and joint contribution of each
FQL-GTA design choice on CartPole-v1 via the unified fql_core engine.

Full 2^3 factorial over three binary design choices -- Gaussian membership
functions, Q(lambda) eligibility traces, and adaptive epsilon -- giving all
eight configurations at the corners of the design cube, per the frozen v1.0
config (configs/default_config.json -> ablation_study).
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

# All eight corners of the 2^3 design cube: (Gaussian MF, eligibility trace,
# adaptive epsilon). Labels encode the on/off pattern directly so the table
# reads as a factorial design, not an arbitrary name.
VARIANT_LABELS = {
    "baseline": "Off / Off / Off (baseline FQL)",
    "epsilon_only": "Off / Off / On",
    "eligibility_only": "Off / On / Off",
    "no_gaussian": "Off / On / On",
    "gaussian_only": "On / Off / Off",
    "no_eligibility": "On / Off / On",
    "gaussian_eligibility": "On / On / Off",
    "full_gta": "On / On / On (full FQL-GTA)",
}


def run_all():
    rows = []
    curves = {}
    for variant in VARIANT_LABELS:
        for seed in SEEDS:
            t0 = time.time()
            agent = make_agent(ENV_NAME, variant=variant, seed=seed)
            result = train(agent, ENV_NAME, episodes=EPISODES, seed=seed)
            returns = result["returns"]
            trailing = returns[-50:].mean()
            rows.append({
                "variant": variant, "label": VARIANT_LABELS[variant], "seed": seed,
                "avg_reward": float(returns.mean()),
                "trailing50_return": float(trailing),
                "std_reward": float(returns.std()),
                "train_time_sec": time.time() - t0,
            })
            curves[f"{variant}__seed{seed}"] = returns.tolist()
            print(f"{VARIANT_LABELS[variant]} | seed {seed} | "
                  f"avg_reward={returns.mean():.1f} | trailing50={trailing:.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "ablation_per_seed.csv", index=False)

    # Table format matches the brief's requested layout: Model | Avg Reward | Std Dev
    # "Avg Reward" is the mean episode return across the full run (not just the
    # trailing window), and "Std Dev" is the across-seed standard deviation of
    # that per-seed average -- i.e. how consistent the variant is across seeds.
    summary = df.groupby("label").agg(
        avg_reward=("avg_reward", "mean"),
        std_dev=("avg_reward", "std"),
        trailing50_mean=("trailing50_return", "mean"),
    ).reset_index()
    order = [VARIANT_LABELS[v] for v in VARIANT_LABELS]
    summary["label"] = pd.Categorical(summary["label"], categories=order, ordered=True)
    summary = summary.sort_values("label").reset_index(drop=True)
    summary.to_csv(RESULTS_DIR / "ablation_summary.csv", index=False)

    max_len = max(len(v) for v in curves.values())
    curves_df = pd.DataFrame({"step": range(max_len)})
    for key, hist in curves.items():
        curves_df[key] = pd.Series(hist).reindex(range(max_len))
    curves_df.to_csv(RESULTS_DIR / "ablation_learning_curves.csv", index=False)

    print("\n" + summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    t0 = time.time()
    run_all()
    print(f"\nTotal ablation study time: {time.time() - t0:.1f}s")
