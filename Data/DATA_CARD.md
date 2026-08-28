# Data Card — Fuzzy Q-Learning on CartPole-v1

**v1.0 (journal revision) note:** the primary comparison was re-run at 30
independent seeds (0-29), up from the assignment-track version's 5 seeds; the
ablation study was extended from 4 leave-one-out conditions to the full 2^3
factorial (8 conditions); and hyperparameter-sensitivity, cross-environment
transfer, and deep-RL baseline studies were added. All of it is driven by the
single frozen configuration in `../configs/default_config.json`. Row counts
below reflect the current (v1.0) artefacts.

## Nature of the data
Reinforcement learning does not consume a static labelled dataset. The learning
signal is generated online through agent–environment interaction with the
**CartPole-v1** control benchmark from the Gymnasium (formerly OpenAI Gym)
library, plus **MountainCar-v0** and **Acrobot-v1** for the cross-environment
transfer study. For transparency and reproducibility, two data artefacts are
exported:

| File | Description | Rows |
|------|-------------|------|
| `cartpole_transitions.csv` | Greedy-policy transitions logged from a trained Improved FQL-GTA agent | 8,610 |
| `../Results/learning_curves.csv` | Per-episode return, every seed, both agents (30 seeds x 500 episodes x 2 agents) | 30,000 |

## Environment specification (CartPole-v1)
- **Objective:** keep a pole balanced upright on a moving cart.
- **State space:** 4 continuous variables
  - cart position, operating range approx. [-2.4, 2.4]
  - cart velocity (unbounded; clipped to [-3.0, 3.0])
  - pole angle in radians, approx. [-0.21, 0.21] (~12 degrees)
  - pole angular velocity (unbounded; clipped to [-3.5, 3.5])
- **Action space:** 2 discrete actions — push cart left (0) or right (1).
- **Reward:** +1 for every timestep the pole remains balanced.
- **Episode termination:** pole angle or cart position leaves the valid range,
  or 500 steps are reached (the maximum return for v1).
- **Solve criterion (this study):** trailing 50-episode mean return >= 475.

## Transition-log schema (`cartpole_transitions.csv`)
`episode, t, cart_pos, cart_vel, pole_angle, pole_ang_vel, action, reward,
next_cart_pos, next_cart_vel, next_pole_angle, next_pole_ang_vel, terminated`

Each row is one Markov transition (s, a, r, s', done) collected under the trained
greedy policy. This log supports offline inspection, batch/off-policy
experiments, and independent verification of the reported behaviour.

## Licence and provenance
Gymnasium is distributed under the MIT Licence. All transition data was
generated locally by the accompanying source code; no third-party or personal
data is involved.
