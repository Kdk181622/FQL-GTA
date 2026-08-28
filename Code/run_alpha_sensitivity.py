"""Baseline-FQL learning-rate (alpha) sensitivity sweep.

The frozen config uses alpha=0.3 for the classical FQL baseline and alpha=0.2
for the proposed FQL-GTA agent -- a genuine fourth changed variable in the
primary 30-seed comparison, on top of the three proposed components (Gaussian
membership, eligibility traces, adaptive epsilon). This script determines
whether a better-tuned classical baseline (same 30 seeds, same 500 episodes,
alpha swept over a small grid) closes any of the gap, so the manuscript can
report a tuned-baseline comparison alongside the canonical one rather than
leaving the learning-rate difference undocumented.

Writes:
  Results/alpha_sensitivity_per_seed.csv   - one row per (alpha, seed)
  Results/alpha_sensitivity_summary.csv    - one row per alpha (aggregate)
  Results/alpha_sensitivity_meta.json      - best alpha + tuned-vs-GTA stats
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from fql_baseline import FuzzyQLearningBaseline, train as train_baseline

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "Results"

ALPHAS = [0.1, 0.2, 0.3, 0.5]
SEEDS = list(range(30))
EPISODES = 500
SOLVE_THRESHOLD = 475.0
SOLVE_WINDOW = 50


def is_finally_solved(history):
    return history[-SOLVE_WINDOW:].mean() >= SOLVE_THRESHOLD


def episodes_to_solve(history):
    for k in range(SOLVE_WINDOW, len(history) + 1):
        if history[k - SOLVE_WINDOW:k].mean() >= SOLVE_THRESHOLD:
            return k
    return np.nan


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    pooled_std = np.sqrt(((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2)
                          / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled_std


def run():
    t_start = time.time()
    per_seed_rows = []
    trailing_by_alpha = {a: [] for a in ALPHAS}

    for alpha in ALPHAS:
        for seed in SEEDS:
            agent = FuzzyQLearningBaseline(seed=seed, alpha=alpha)
            history = train_baseline(agent, episodes=EPISODES, seed=seed)
            trailing50 = float(history[-SOLVE_WINDOW:].mean())
            trailing_by_alpha[alpha].append(trailing50)
            per_seed_rows.append({
                "alpha": alpha, "seed": seed,
                "trailing50_return": trailing50,
                "best_return": float(history.max()),
                "auc": float(history.mean()),
                "episodes_to_solve": episodes_to_solve(history),
                "finally_solved": bool(is_finally_solved(history)),
            })
        elapsed = time.time() - t_start
        print(f"alpha={alpha} done ({len(SEEDS)} seeds) -- {elapsed:.0f}s elapsed")

    per_seed = pd.DataFrame(per_seed_rows)
    per_seed.to_csv(RESULTS_DIR / "alpha_sensitivity_per_seed.csv", index=False)

    summary_rows = []
    for alpha in ALPHAS:
        sub = per_seed[per_seed["alpha"] == alpha]
        summary_rows.append({
            "alpha": alpha,
            "final_return_mean": sub["trailing50_return"].mean(),
            "final_return_std": sub["trailing50_return"].std(ddof=1),
            "solved_rate": sub["finally_solved"].mean(),
            "n_seeds_solved": int(sub["finally_solved"].sum()),
            "n_seeds_total": len(sub),
            "ever_crossed_threshold_rate": sub["episodes_to_solve"].notna().mean(),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "alpha_sensitivity_summary.csv", index=False)
    print("\n=== Alpha sensitivity summary ===")
    print(summary.round(3).to_string(index=False))

    # Best-tuned alpha = highest mean trailing-50 return across the 30 seeds.
    best_alpha = float(summary.loc[summary["final_return_mean"].idxmax(), "alpha"])
    tuned = np.array(trailing_by_alpha[best_alpha])

    # Compare the tuned baseline against the existing 30-seed FQL-GTA result
    # (same seeds 0-29, same episode budget, already in Results/learning_curves.csv).
    curves = pd.read_csv(RESULTS_DIR / "learning_curves.csv")
    gta_sub = curves[curves["agent"] == "Improved FQL-GTA"]
    gta_trailing = np.array([
        gta_sub[gta_sub["seed"] == s].sort_values("episode")["return"].values[-SOLVE_WINDOW:].mean()
        for s in SEEDS
    ])

    d = cohens_d(gta_trailing, tuned)
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(tuned, gta_trailing)

    # Also record the canonical alpha=0.3 vs GTA comparison for reference
    # (should match the existing primary_comparison stats in
    # Results/statistical_tests.json as a consistency check).
    canonical = np.array(trailing_by_alpha[0.3])
    d_canonical = cohens_d(gta_trailing, canonical)
    wilcoxon_stat_canonical, wilcoxon_p_canonical = stats.wilcoxon(canonical, gta_trailing)

    meta = {
        "alphas_tested": ALPHAS,
        "n_seeds": len(SEEDS),
        "episodes_per_seed": EPISODES,
        "best_tuned_alpha": best_alpha,
        "canonical_alpha": 0.3,
        "gta_alpha": 0.2,
        "tuned_vs_gta": {
            "tuned_alpha": best_alpha,
            "tuned_final_return_mean": float(tuned.mean()),
            "tuned_final_return_std": float(tuned.std(ddof=1)),
            "gta_final_return_mean": float(gta_trailing.mean()),
            "gta_final_return_std": float(gta_trailing.std(ddof=1)),
            "cohens_d": float(d),
            "wilcoxon_statistic": float(wilcoxon_stat),
            "wilcoxon_p_value": float(wilcoxon_p),
        },
        "canonical_vs_gta": {
            "canonical_alpha": 0.3,
            "canonical_final_return_mean": float(canonical.mean()),
            "canonical_final_return_std": float(canonical.std(ddof=1)),
            "gta_final_return_mean": float(gta_trailing.mean()),
            "gta_final_return_std": float(gta_trailing.std(ddof=1)),
            "cohens_d": float(d_canonical),
            "wilcoxon_statistic": float(wilcoxon_stat_canonical),
            "wilcoxon_p_value": float(wilcoxon_p_canonical),
        },
    }
    with open(RESULTS_DIR / "alpha_sensitivity_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nBest tuned alpha: {best_alpha}")
    print(f"Tuned ({best_alpha}) vs GTA: d={d:.2f}, Wilcoxon p={wilcoxon_p:.2e}")
    print(f"Canonical (0.3) vs GTA: d={d_canonical:.2f}, Wilcoxon p={wilcoxon_p_canonical:.2e}")
    print(f"\nTotal time: {time.time() - t_start:.0f}s")
    return summary, meta


if __name__ == "__main__":
    run()
