"""Traces the full state -> membership -> firing strength -> Q-values -> action
chain for real states visited by a trained FQL-GTA agent on CartPole-v1, for
direct use in the manuscript's interpretability section. This is the concrete
evidence behind the reframed novelty claim: not "the combination of Gaussian MF
+ Q(lambda) + adaptive epsilon is itself novel" but "a systematically evaluated,
inspectable enhancement to FQL whose decisions can be traced rule-by-rule."

Fuzzy sets are identified by their numeric centre (these are evenly spaced,
data-driven partitions, not hand-labelled linguistic categories such as
"Low"/"High" -- assigning invented semantic labels would overstate what the
model actually represents), so every reported quantity is read directly off
the trained agent with nothing added for presentation.
"""
import json
from pathlib import Path

import numpy as np
import gymnasium as gym

from fql_core import make_agent, train, gaussian_membership

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "Results"
RESULTS_DIR.mkdir(exist_ok=True)

ENV_NAME = "CartPole-v1"
TRAIN_EPISODES = 500
TRAIN_SEED = 0
N_WORKED_EXAMPLES = 5
TOP_K_RULES = 4
DIM_NAMES = ["cart position", "cart velocity", "pole angle", "pole angular velocity"]
ACTION_NAMES = ["push cart left", "push cart right"]


def state_dims_summary(agent, state):
    clipped = [np.clip(state[d], agent.state_ranges[d][0], agent.state_ranges[d][1])
               for d in range(len(agent.state_ranges))]
    per_dim = []
    for d in range(len(agent.state_ranges)):
        degrees = gaussian_membership(clipped[d], agent.centers[d], agent.sigmas[d])
        top2 = np.argsort(degrees)[::-1][:2]
        per_dim.append({
            "dimension": DIM_NAMES[d],
            "raw_value": float(state[d]),
            "clipped_value": float(clipped[d]),
            "dominant_sets": [
                {"set_center": float(agent.centers[d][i]), "membership_degree": float(degrees[i])}
                for i in top2
            ],
        })
    return per_dim


def worked_example(agent, state):
    phi = agent._firing_strengths(state)
    q = agent.q_values(phi)
    top_rules_idx = np.argsort(phi)[::-1][:TOP_K_RULES]
    return {
        "raw_state": [float(x) for x in state],
        "per_dimension_membership": state_dims_summary(agent, state),
        "top_firing_rules": [
            {"rule_index": int(i), "firing_strength": float(phi[i]),
             "q_push_left": float(agent.theta[i, 0]), "q_push_right": float(agent.theta[i, 1])}
            for i in top_rules_idx
        ],
        "blended_q_values": {"push_left": float(q[0]), "push_right": float(q[1])},
        "selected_action": ACTION_NAMES[int(np.argmax(q))],
        "n_rules_total": int(agent.n_rules),
        "n_rules_firing_above_0.01": int(np.sum(phi > 0.01)),
    }


def run():
    agent = make_agent(ENV_NAME, variant="full_gta", seed=TRAIN_SEED)
    train_result = train(agent, ENV_NAME, episodes=TRAIN_EPISODES, seed=TRAIN_SEED)
    print(f"Trained FQL-GTA: final-50 mean return = {train_result['returns'][-50:].mean():.1f}")

    env = gym.make(ENV_NAME)
    state, _ = env.reset(seed=999)
    examples = []
    step = 0
    while len(examples) < N_WORKED_EXAMPLES and step < 500:
        action, phi, was_greedy = agent.act(state, greedy=True)
        if step % 20 == 0:
            examples.append(worked_example(agent, state))
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
        step += 1
        if terminated or truncated:
            state, _ = env.reset(seed=999 + step)
    env.close()

    out = {
        "env": ENV_NAME, "train_episodes": TRAIN_EPISODES, "train_seed": TRAIN_SEED,
        "final50_train_return": float(train_result["returns"][-50:].mean()),
        "n_rules_total": int(agent.n_rules),
        "examples": examples,
    }
    with open(RESULTS_DIR / "learned_rules_worked_examples.json", "w") as f:
        json.dump(out, f, indent=2)

    for i, ex in enumerate(examples):
        print(f"\n--- Worked example {i+1} ---")
        print(f"state={np.round(ex['raw_state'],3).tolist()}")
        print(f"n_rules_firing>0.01={ex['n_rules_firing_above_0.01']}/{ex['n_rules_total']}")
        print(f"top rule: idx={ex['top_firing_rules'][0]['rule_index']} "
              f"firing={ex['top_firing_rules'][0]['firing_strength']:.3f}")
        print(f"Q(left)={ex['blended_q_values']['push_left']:.2f}  "
              f"Q(right)={ex['blended_q_values']['push_right']:.2f}  "
              f"-> {ex['selected_action']}")

    print(f"\nSaved {len(examples)} worked examples to results/learned_rules_worked_examples.json")
    return out


if __name__ == "__main__":
    run()
