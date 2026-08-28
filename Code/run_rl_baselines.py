"""Deep-RL baseline comparison for the journal submission: FQL / FQL-GTA vs.
a real hand-rolled DQN, and vs. stable-baselines3's DQN and PPO, all on
CartPole-v1, all measured in this one script for a like-for-like timing and
parameter-count comparison (frozen protocol: configs/default_config.json ->
rl_baselines).

FQL and the hand-rolled DQN are trained episode-by-episode (500 episodes,
matching the primary comparison's protocol), so their learning curves and
solve-rate use the exact same trailing-50-episode "solved" definition as
run_experiments.py. The two stable-baselines3 agents are trained on a fixed
timestep budget (their natural training unit) and their per-episode returns
are recovered from the Monitor wrapper's episode-reward log, then scored with
the identical trailing-50 / AUC / solved-rate definitions for comparability.
10 seeds (0-9) are used for every method -- smaller than the primary FQL
comparison's 30 seeds, given the added compute of running two additional deep
RL algorithms end to end.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import gymnasium as gym

from fql_baseline import FuzzyQLearningBaseline, train as train_fql_baseline
from fql_improved import FuzzyQLearningImproved, train as train_fql_improved
from dqn_baseline import train_dqn, n_params as dqn_n_params

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "Results"
RESULTS_DIR.mkdir(exist_ok=True)

SEEDS = list(range(10))
EPISODES = 500              # FQL / hand-rolled DQN training budget
SB3_TOTAL_TIMESTEPS = 60_000  # stable-baselines3 training budget (its native unit);
# chosen to be the same order of magnitude as the FQL-GTA agent's own total
# environment-step budget under 500 episodes (mean AUC return x 500 episodes
# approx 147,000 steps for the full training run, but the bulk of solved
# performance is reached well before that; 60k timesteps is standard practice
# for solving CartPole-v1 with SB3's default hyperparameters).
SOLVE_THRESHOLD = 475.0
SOLVE_WINDOW = 50


def episodes_to_solve(history):
    for k in range(SOLVE_WINDOW, len(history) + 1):
        if np.mean(history[k - SOLVE_WINDOW:k]) >= SOLVE_THRESHOLD:
            return k
    return np.nan


def is_finally_solved(history):
    return np.mean(history[-SOLVE_WINDOW:]) >= SOLVE_THRESHOLD


def agg(per_seed_histories, train_times, n_params):
    stack_final, stack_auc, stack_solved = [], [], []
    for h in per_seed_histories:
        h = np.asarray(h, dtype=float)
        w = min(SOLVE_WINDOW, len(h))
        stack_final.append(h[-w:].mean())
        stack_auc.append(h.mean())
        stack_solved.append(is_finally_solved(h) if len(h) >= SOLVE_WINDOW else False)
    stack_final = np.array(stack_final)
    stack_auc = np.array(stack_auc)
    stack_solved = np.array(stack_solved)
    return {
        "final_return_mean": float(stack_final.mean()),
        "final_return_std": float(stack_final.std(ddof=1)),
        "auc_mean": float(stack_auc.mean()),
        "auc_std": float(stack_auc.std(ddof=1)),
        "solved_rate": float(stack_solved.mean()),
        "n_seeds_solved": int(stack_solved.sum()),
        "n_seeds_total": len(per_seed_histories),
        "train_time_sec_mean": float(np.mean(train_times)),
        "train_time_sec_std": float(np.std(train_times, ddof=1)) if len(train_times) > 1 else 0.0,
        "n_params": int(n_params),
        "n_episodes_per_seed_mean": float(np.mean([len(h) for h in per_seed_histories])),
    }


def run_sb3(algo_cls, algo_name, seed):
    env = Monitor(gym.make("CartPole-v1"))
    model = algo_cls("MlpPolicy", env, verbose=0, seed=seed)
    t0 = time.time()
    model.learn(total_timesteps=SB3_TOTAL_TIMESTEPS)
    train_time = time.time() - t0
    returns = env.get_episode_rewards()
    n_params = sum(p.numel() for p in model.policy.parameters())
    env.close()
    return np.array(returns, dtype=float), train_time, n_params


def run():
    results = {}

    # --- Baseline FQL and Improved FQL-GTA (re-timed here for a like-for-like
    #     comparison; returns are deterministic given the seed, so these match
    #     the primary 30-seed run's first 10 seeds exactly). ---
    for name, cls, trainer in [
        ("Baseline FQL", FuzzyQLearningBaseline, train_fql_baseline),
        ("Improved FQL-GTA", FuzzyQLearningImproved, train_fql_improved),
    ]:
        histories, times = [], []
        n_params = None
        for seed in SEEDS:
            agent = cls(seed=seed)
            t0 = time.time()
            h = trainer(agent, episodes=EPISODES, seed=seed)
            times.append(time.time() - t0)
            histories.append(h)
            n_params = agent.theta.size
            print(f"{name} | seed {seed} | final50={h[-50:].mean():.1f} | time={times[-1]:.1f}s")
        results[name] = agg(histories, times, n_params)

    # --- Hand-rolled DQN. ---
    histories, times = [], []
    n_params = None
    for seed in SEEDS:
        out = train_dqn(episodes=EPISODES, seed=seed)
        histories.append(out["returns"])
        times.append(out["train_time_sec"])
        n_params = out["n_params"]
        print(f"DQN (hand-rolled) | seed {seed} | final50={out['returns'][-50:].mean():.1f} | "
              f"time={times[-1]:.1f}s")
    results["DQN (hand-rolled)"] = agg(histories, times, n_params)

    # --- stable-baselines3 DQN and PPO. ---
    for algo_cls, algo_name in [(DQN, "DQN (stable-baselines3)"), (PPO, "PPO (stable-baselines3)")]:
        histories, times = [], []
        n_params = None
        for seed in SEEDS:
            h, t, np_ = run_sb3(algo_cls, algo_name, seed)
            histories.append(h)
            times.append(t)
            n_params = np_
            tail = h[-50:].mean() if len(h) >= 1 else float("nan")
            print(f"{algo_name} | seed {seed} | n_episodes={len(h)} | final50={tail:.1f} | "
                  f"time={t:.1f}s")
        results[algo_name] = agg(histories, times, n_params)

    with open(RESULTS_DIR / "rl_baselines_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    table = pd.DataFrame(results).T
    table.index.name = "method"
    table.to_csv(RESULTS_DIR / "rl_baselines_comparison.csv")
    print("\n=== RL baselines comparison ===")
    print(table[["final_return_mean", "auc_mean", "solved_rate",
                  "train_time_sec_mean", "n_params"]].round(2).to_string())
    return results


if __name__ == "__main__":
    t0 = time.time()
    run()
    print(f"\nTotal time: {time.time() - t0:.1f}s")
