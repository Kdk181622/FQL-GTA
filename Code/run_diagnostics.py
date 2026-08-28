"""Captures per-episode diagnostic traces (TD-error magnitude, epsilon, and every
visited raw state) for one baseline run and one full FQL-GTA run on CartPole-v1,
used only for the richer diagnostic visualizations (moving-average reward,
TD-error evolution, epsilon-decay curve, state-visitation heatmap). Kept separate
from run_ablation_study.py so that script doesn't pay the memory/time cost of
tracking every visited state across all variants and seeds.
"""
import json
from pathlib import Path

import numpy as np

from fql_core import make_agent, train

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "Results"
RESULTS_DIR.mkdir(exist_ok=True)

ENV_NAME = "CartPole-v1"
EPISODES = 400
SEED = 0


def run():
    out = {}
    for variant in ["baseline", "full_gta"]:
        agent = make_agent(ENV_NAME, variant=variant, seed=SEED)
        result = train(agent, ENV_NAME, episodes=EPISODES, seed=SEED, track_state_visits=True)
        out[variant] = {
            "returns": result["returns"].tolist(),
            "td_errors": result["td_errors"].tolist(),
            "epsilons": result["epsilons"].tolist(),
            "visited_states": result["visited_states"].tolist(),
        }
        print(f"{variant}: {len(result['returns'])} episodes, "
              f"{len(result['visited_states'])} state visits captured")

    with open(RESULTS_DIR / "diagnostics.json", "w") as f:
        json.dump(out, f)
    print("Saved results/diagnostics.json")


if __name__ == "__main__":
    run()
