"""Computational-cost comparison between the primary Baseline FQL and Improved
FQL-GTA agents: real wall-clock training time (matched hyperparameters/episode
budget, same machine, same process, run back-to-back to avoid drift), exact
trainable-parameter counts, and per-step asymptotic update complexity.

This is a genuine benchmark, not an estimate: both agents are trained for the
same number of episodes on the same seeds, timed with time.perf_counter().
"""
import json
import time
from pathlib import Path

import numpy as np

from fql_baseline import FuzzyQLearningBaseline, train as train_baseline
from fql_improved import FuzzyQLearningImproved, train as train_improved

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "Results"

SEEDS = [0, 1, 2]
EPISODES = 500


def time_agent(agent_cls, train_fn, seeds):
    per_seed_times = []
    for seed in seeds:
        agent = agent_cls(seed=seed)
        t0 = time.perf_counter()
        train_fn(agent, episodes=EPISODES, seed=seed)
        per_seed_times.append(time.perf_counter() - t0)
    return agent, per_seed_times


def main():
    print(f"Timing {len(SEEDS)} seeds x {EPISODES} episodes per agent "
          f"(back-to-back, same process)...")

    base_agent, base_times = time_agent(FuzzyQLearningBaseline, train_baseline, SEEDS)
    imp_agent, imp_times = time_agent(FuzzyQLearningImproved, train_improved, SEEDS)

    n_rules_base = base_agent.n_rules
    n_actions_base = base_agent.n_actions
    theta_params_base = base_agent.theta.size

    n_rules_imp = imp_agent.n_rules
    n_actions_imp = imp_agent.n_actions
    theta_params_imp = imp_agent.theta.size
    etrace_params_imp = imp_agent.e_trace.size

    result = {
        "episodes_per_run": EPISODES,
        "seeds": SEEDS,
        "baseline": {
            "n_rules": n_rules_base,
            "n_actions": n_actions_base,
            "trainable_params_theta": theta_params_base,
            "runtime_memory_params_total": theta_params_base,
            "per_seed_train_time_sec": base_times,
            "mean_train_time_sec": float(np.mean(base_times)),
            "std_train_time_sec": float(np.std(base_times)),
            "mean_time_per_episode_ms": float(np.mean(base_times) / EPISODES * 1000),
            "per_step_update_complexity": "O(K) -- updates one action-column of theta",
        },
        "improved_fql_gta": {
            "n_rules": n_rules_imp,
            "n_actions": n_actions_imp,
            "trainable_params_theta": theta_params_imp,
            "eligibility_trace_params": etrace_params_imp,
            "runtime_memory_params_total": theta_params_imp + etrace_params_imp,
            "per_seed_train_time_sec": imp_times,
            "mean_train_time_sec": float(np.mean(imp_times)),
            "std_train_time_sec": float(np.std(imp_times)),
            "mean_time_per_episode_ms": float(np.mean(imp_times) / EPISODES * 1000),
            "per_step_update_complexity": "O(K*A) -- Q(lambda) update touches the full trace matrix",
        },
    }
    result["time_overhead_pct"] = float(
        (result["improved_fql_gta"]["mean_train_time_sec"]
         - result["baseline"]["mean_train_time_sec"])
        / result["baseline"]["mean_train_time_sec"] * 100
    )
    result["memory_overhead_pct"] = float(
        (result["improved_fql_gta"]["runtime_memory_params_total"]
         - result["baseline"]["runtime_memory_params_total"])
        / result["baseline"]["runtime_memory_params_total"] * 100
    )

    with open(RESULTS_DIR / "computational_cost.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nBaseline:  {result['baseline']['mean_train_time_sec']:.2f}s "
          f"+/- {result['baseline']['std_train_time_sec']:.2f}s "
          f"({result['baseline']['runtime_memory_params_total']} params)")
    print(f"FQL-GTA:   {result['improved_fql_gta']['mean_train_time_sec']:.2f}s "
          f"+/- {result['improved_fql_gta']['std_train_time_sec']:.2f}s "
          f"({result['improved_fql_gta']['runtime_memory_params_total']} params)")
    print(f"Time overhead:   {result['time_overhead_pct']:+.1f}%")
    print(f"Memory overhead: {result['memory_overhead_pct']:+.1f}%")


if __name__ == "__main__":
    main()
