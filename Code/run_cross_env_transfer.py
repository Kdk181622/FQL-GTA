"""Cross-environment transfer study, framed as a formal research question:
does the CartPole-tuned FQL-GTA (lambda=0.5, sigma_mult=0.35, epsilon_decay=0.99)
transfer zero-shot to MountainCar-v0 and Acrobot-v1, or does it need
per-environment retuning to be competitive?

Zero-shot: the CartPole-tuned hyperparameters are applied directly to each
target environment, no retuning.

Tuned: a one-factor-at-a-time coordinate search over the same three
hyperparameters and the same value grids used in the CartPole sensitivity
sweep (run_hyperparameter_sensitivity.py), selected on a held-out validation
seed (99, disjoint from the evaluation seeds) to avoid leakage, then evaluated
on the same seeds/episode budget as the zero-shot condition for a fair
comparison. Protocol matches configs/default_config.json -> cross_environment_transfer.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fql_core import FuzzyQLearningAgent, ENV_SPECS, make_agent, train

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "Results"
RESULTS_DIR.mkdir(exist_ok=True)

TARGET_ENVS = ["MountainCar-v0", "Acrobot-v1"]
EVAL_SEEDS = [0, 1, 2]
VALIDATION_SEED = 99
EPISODES = 400

CARTPOLE_TUNED = dict(lam=0.5, sigma_mult=0.35, epsilon_decay=0.99)
SWEEP_ORDER = ["lam", "sigma_mult", "epsilon_decay"]
SWEEPS = {
    "lam": [0.0, 0.3, 0.5, 0.7, 0.9, 0.99],
    "sigma_mult": [0.2, 0.35, 0.5, 0.7],
    "epsilon_decay": [0.90, 0.95, 0.99, 0.995, 0.999],
}


def make_full_gta(env_name, seed, **hp):
    spec = ENV_SPECS[env_name]
    return FuzzyQLearningAgent(
        state_ranges=spec["state_ranges"], sets_per_dim=spec["sets_per_dim"],
        n_actions=spec["n_actions"], seed=seed,
        use_gaussian=True, use_eligibility=True, use_adaptive_epsilon=True,
        **hp,
    )


def trailing50(env_name, seed, hp):
    agent = make_full_gta(env_name, seed, **hp)
    returns = train(agent, env_name, episodes=EPISODES, seed=seed)["returns"]
    return float(returns[-50:].mean()), returns


def coordinate_search(env_name):
    """One-factor-at-a-time search on the held-out validation seed, starting
    from the CartPole-tuned defaults."""
    current = dict(CARTPOLE_TUNED)
    trace = []
    for param in SWEEP_ORDER:
        best_val, best_score = current[param], -np.inf
        for val in SWEEPS[param]:
            trial = dict(current)
            trial[param] = val
            score, _ = trailing50(env_name, VALIDATION_SEED, trial)
            trace.append({"env": env_name, "param": param, "value": val, "val_trailing50": score})
            if score > best_score:
                best_score, best_val = score, val
        current[param] = best_val
        print(f"[{env_name}] tuned {param} -> {best_val} (val trailing50={best_score:.1f})")
    return current, trace


def run():
    all_rows = []
    all_curves = {}
    all_trace = []

    for env_name in TARGET_ENVS:
        # --- Zero-shot: CartPole-tuned hyperparameters applied directly. ---
        for seed in EVAL_SEEDS:
            score, returns = trailing50(env_name, seed, CARTPOLE_TUNED)
            all_rows.append({"env": env_name, "condition": "zero_shot", "seed": seed,
                              "trailing50_return": score, "auc": float(returns.mean()),
                              **{f"hp_{k}": v for k, v in CARTPOLE_TUNED.items()}})
            all_curves[f"{env_name}__zero_shot__seed{seed}"] = returns.tolist()
            print(f"{env_name} | zero_shot | seed {seed} | trailing50={score:.1f}")

        # --- Tuned: coordinate search on validation seed, then evaluate. ---
        best_hp, trace = coordinate_search(env_name)
        all_trace.extend(trace)
        for seed in EVAL_SEEDS:
            score, returns = trailing50(env_name, seed, best_hp)
            all_rows.append({"env": env_name, "condition": "tuned", "seed": seed,
                              "trailing50_return": score, "auc": float(returns.mean()),
                              **{f"hp_{k}": v for k, v in best_hp.items()}})
            all_curves[f"{env_name}__tuned__seed{seed}"] = returns.tolist()
            print(f"{env_name} | tuned ({best_hp}) | seed {seed} | trailing50={score:.1f}")

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / "cross_env_transfer_per_seed.csv", index=False)

    summary = df.groupby(["env", "condition"]).agg(
        trailing50_mean=("trailing50_return", "mean"),
        trailing50_std=("trailing50_return", "std"),
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
    ).reset_index()
    summary.to_csv(RESULTS_DIR / "cross_env_transfer_summary.csv", index=False)

    pd.DataFrame(all_trace).to_csv(RESULTS_DIR / "cross_env_transfer_search_trace.csv", index=False)

    max_len = max(len(v) for v in all_curves.values())
    curves_df = pd.DataFrame({"step": range(max_len)})
    for key, hist in all_curves.items():
        curves_df[key] = pd.Series(hist).reindex(range(max_len))
    curves_df.to_csv(RESULTS_DIR / "cross_env_transfer_learning_curves.csv", index=False)

    with open(RESULTS_DIR / "cross_env_transfer_meta.json", "w") as f:
        json.dump({"target_envs": TARGET_ENVS, "eval_seeds": EVAL_SEEDS,
                    "validation_seed": VALIDATION_SEED, "episodes_per_seed": EPISODES,
                    "cartpole_tuned_defaults": CARTPOLE_TUNED, "sweep_order": SWEEP_ORDER,
                    "sweeps": SWEEPS}, f, indent=2)

    print("\n" + summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    t0 = time.time()
    run()
    print(f"\nTotal cross-environment transfer study time: {time.time() - t0:.1f}s")
