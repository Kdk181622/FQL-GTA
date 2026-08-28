"""
Experiment runner for the Fuzzy Q-Learning study on CartPole-v1.

Trains the baseline FQL (triangular MF, fixed epsilon) and the improved
FQL-GTA (Gaussian MF, Q(lambda) eligibility traces, adaptive epsilon) across
multiple random seeds, records per-episode returns, computes summary metrics,
and writes machine-readable artefacts used by the manuscript and the notebook:

  results/learning_curves.csv   - per-episode return, every seed, both agents
  results/summary_metrics.csv   - aggregated comparison metrics
  results/metrics.json          - the same metrics as JSON for programmatic use

Reproducibility: a fixed list of seeds is used and every environment reset is
seeded deterministically, so the reported figures regenerate exactly.
"""

import json
import numpy as np
import pandas as pd

from fql_baseline import FuzzyQLearningBaseline, train as train_baseline
from fql_improved import FuzzyQLearningImproved, train as train_improved

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
SEEDS = list(range(30))  # v1.0 frozen config: 30 independent seeds for journal-grade power
EPISODES = 500
SOLVE_THRESHOLD = 475.0     # trailing-50-episode mean defining a "solved" run
SOLVE_WINDOW = 50


def episodes_to_solve(history):
    """Return the first episode index whose trailing-window mean crosses the bar."""
    for k in range(SOLVE_WINDOW, len(history) + 1):
        if history[k - SOLVE_WINDOW:k].mean() >= SOLVE_THRESHOLD:
            return k
    return np.nan


def is_finally_solved(history):
    """True only if the run is STILL at/above the bar at the end of training.

    episodes_to_solve() alone is not a valid "solved" indicator: a run can
    cross the bar mid-training and then regress (observed directly in this
    study's seed 4, which peaked at 495.9 around episode 429 then collapsed
    to 235.1 by episode 500). Reporting "solved" from first-crossing alone
    silently counts that regression as a success.
    """
    return history[-SOLVE_WINDOW:].mean() >= SOLVE_THRESHOLD


def moving_average(x, w=SOLVE_WINDOW):
    """Trailing moving average with a shrinking leading window."""
    return np.array([x[max(0, i - w + 1):i + 1].mean() for i in range(len(x))])


def run():
    curve_rows = []
    per_seed = {"baseline": [], "improved": []}

    for seed in SEEDS:
        # Baseline agent.
        base = FuzzyQLearningBaseline(seed=seed)
        h_base = train_baseline(base, episodes=EPISODES, seed=seed)
        per_seed["baseline"].append(h_base)

        # Improved agent.
        imp = FuzzyQLearningImproved(seed=seed)
        h_imp = train_improved(imp, episodes=EPISODES, seed=seed)
        per_seed["improved"].append(h_imp)

        for ep in range(EPISODES):
            curve_rows.append({"seed": seed, "episode": ep + 1,
                               "agent": "Baseline FQL", "return": h_base[ep]})
            curve_rows.append({"seed": seed, "episode": ep + 1,
                               "agent": "Improved FQL-GTA", "return": h_imp[ep]})

    curves = pd.DataFrame(curve_rows)
    curves.to_csv("../Results/learning_curves.csv", index=False)

    # -----------------------------------------------------------------------
    # Aggregate summary metrics
    # -----------------------------------------------------------------------
    def ci95(x):
        x = np.asarray(x)
        n = len(x)
        if n < 2:
            return [float("nan"), float("nan")]
        from scipy import stats as _stats
        sem = x.std(ddof=1) / np.sqrt(n)
        lo, hi = _stats.t.interval(0.95, df=n - 1, loc=x.mean(), scale=sem)
        return [float(lo), float(hi)]

    def agg(stack):
        stack = np.vstack(stack)                       # shape [n_seeds, episodes]
        final = stack[:, -50:].mean(axis=1)            # trailing-50 mean per seed
        best = stack.max(axis=1)                       # best episode per seed
        auc = stack.mean(axis=1)                       # mean return over training
        solves = np.array([episodes_to_solve(h) for h in stack], dtype=float)
        finally_solved = np.array([is_finally_solved(h) for h in stack])
        solved_rate = float(np.mean(finally_solved))
        # Median convergence speed is reported only over seeds that stayed
        # solved through the end of training -- a seed that regressed after
        # first crossing the bar should not count toward "how fast it solved".
        solved_speeds = solves[finally_solved]
        return {
            "final_return_mean": float(final.mean()),
            "final_return_std": float(final.std(ddof=1)),
            "final_return_median": float(np.median(final)),
            "final_return_iqr": [float(np.percentile(final, 25)), float(np.percentile(final, 75))],
            "final_return_min": float(final.min()),
            "final_return_max": float(final.max()),
            "final_return_95ci": ci95(final),
            "best_return_mean": float(best.mean()),
            "auc_mean": float(auc.mean()),
            "auc_std": float(auc.std(ddof=1)),
            "auc_95ci": ci95(auc),
            "solved_rate": solved_rate,
            "n_seeds_solved": int(finally_solved.sum()),
            "n_seeds_total": int(len(stack)),
            "median_episodes_to_solve": (float(np.nanmedian(solved_speeds))
                                         if solved_rate > 0 else None),
            "ever_crossed_threshold_rate": float(np.mean(~np.isnan(solves))),
        }

    metrics = {"Baseline FQL": agg(per_seed["baseline"]),
               "Improved FQL-GTA": agg(per_seed["improved"])}

    with open("../Results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    summary = pd.DataFrame(metrics).T
    summary.to_csv("../Results/summary_metrics.csv")

    # Also persist the mean learning curves for plotting.
    mean_curves = pd.DataFrame({
        "episode": np.arange(1, EPISODES + 1),
        "baseline_mean": np.vstack(per_seed["baseline"]).mean(axis=0),
        "baseline_ma": moving_average(np.vstack(per_seed["baseline"]).mean(axis=0)),
        "improved_mean": np.vstack(per_seed["improved"]).mean(axis=0),
        "improved_ma": moving_average(np.vstack(per_seed["improved"]).mean(axis=0)),
    })
    mean_curves.to_csv("../Results/mean_learning_curves.csv", index=False)

    print("=== Summary metrics ===")
    print(summary.round(2).to_string())
    return metrics, mean_curves


if __name__ == "__main__":
    run()
