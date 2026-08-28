"""Unified, environment-agnostic Fuzzy Q-Learning agent.

Generalises fql_baseline.py / fql_improved.py into a single configurable class so
the same engine can (a) run on environments beyond CartPole, and (b) support an
ablation study that toggles each of the three FQL-GTA design choices -- Gaussian
membership functions, Q(lambda) eligibility traces, and adaptive epsilon --
independently. fql_baseline.py and fql_improved.py are left untouched so the
original, already-verified CartPole comparison keeps working exactly as before;
this module is the engine behind the new multi-environment and ablation studies.
"""
import numpy as np
import gymnasium as gym


def triangular_centers(low, high, n_sets):
    return np.linspace(low, high, n_sets)


def triangular_membership(x, centers):
    width = centers[1] - centers[0]
    return np.maximum(0.0, 1.0 - np.abs(x - centers) / width)


def gaussian_membership(x, centers, sigma):
    return np.exp(-0.5 * ((x - centers) / sigma) ** 2)


# Operating ranges, fuzzy-set resolution, action count, and episode length for
# each benchmark. Acrobot's angle dimensions are given as cos/sin in [-1, 1] by
# Gymnasium; the two angular-velocity dimensions use their documented bounds.
ENV_SPECS = {
    "CartPole-v1": dict(
        state_ranges=[(-2.4, 2.4), (-3.0, 3.0), (-0.21, 0.21), (-3.5, 3.5)],
        sets_per_dim=[3, 3, 6, 6], n_actions=2, max_steps=500,
    ),
    "MountainCar-v0": dict(
        state_ranges=[(-1.2, 0.6), (-0.07, 0.07)],
        sets_per_dim=[8, 8], n_actions=3, max_steps=200,
    ),
    "Acrobot-v1": dict(
        state_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),
                       (-12.57, 12.57), (-28.27, 28.27)],
        sets_per_dim=[3, 3, 3, 3, 4, 4], n_actions=3, max_steps=500,
    ),
}


class FuzzyQLearningAgent:
    """FQL agent with independently toggleable Gaussian MF / eligibility trace /
    adaptive-epsilon design choices, for use as both a multi-environment agent
    and an ablation-study building block."""

    def __init__(self, state_ranges, sets_per_dim, n_actions,
                 alpha=0.2, gamma=0.99,
                 use_gaussian=True, use_eligibility=True, use_adaptive_epsilon=True,
                 epsilon=0.1, epsilon_start=0.5, epsilon_min=0.005, epsilon_decay=0.99,
                 lam=0.5, sigma_mult=0.35, seed=0):
        self.state_ranges = state_ranges
        self.sets_per_dim = sets_per_dim
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.use_gaussian = use_gaussian
        self.use_eligibility = use_eligibility
        self.use_adaptive_epsilon = use_adaptive_epsilon
        self.lam = lam
        self.sigma_mult = sigma_mult
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start if use_adaptive_epsilon else epsilon
        self.rng = np.random.default_rng(seed)

        self.centers = [triangular_centers(lo, hi, n)
                         for (lo, hi), n in zip(state_ranges, sets_per_dim)]
        if use_gaussian:
            # Width = sigma_mult x inter-centre spacing keeps membership local
            # (each state strongly activates only neighbouring rules), matching
            # the original FQL-GTA design (see fql_improved.py). Default 0.35
            # is the CartPole-tuned value; sigma_mult is exposed as a parameter
            # so the cross-environment transfer study can retune it per environment.
            self.sigmas = [sigma_mult * (c[1] - c[0]) for c in self.centers]

        self.n_rules = int(np.prod(sets_per_dim))
        self.theta = np.zeros((self.n_rules, n_actions))
        self.e_trace = np.zeros((self.n_rules, n_actions)) if use_eligibility else None

    def _firing_strengths(self, state):
        clipped = [np.clip(state[d], self.state_ranges[d][0], self.state_ranges[d][1])
                   for d in range(len(self.state_ranges))]
        if self.use_gaussian:
            deg = [gaussian_membership(clipped[d], self.centers[d], self.sigmas[d])
                   for d in range(len(self.state_ranges))]
        else:
            deg = [triangular_membership(clipped[d], self.centers[d])
                   for d in range(len(self.state_ranges))]
        phi = deg[0]
        for d in range(1, len(deg)):
            phi = np.outer(phi, deg[d]).ravel()
        total = phi.sum()
        phi = np.ones(self.n_rules) / self.n_rules if total <= 1e-12 else phi / total
        return phi

    def q_values(self, phi):
        return phi @ self.theta

    def act(self, state, greedy=False):
        phi = self._firing_strengths(state)
        if (not greedy) and self.rng.random() < self.epsilon:
            action = int(self.rng.integers(self.n_actions))
            was_greedy = False
        else:
            action = int(np.argmax(self.q_values(phi)))
            was_greedy = True
        return action, phi, was_greedy

    def reset_traces(self):
        if self.use_eligibility:
            self.e_trace.fill(0.0)

    def update(self, phi, action, reward, next_state, done, was_greedy):
        q_sa = self.q_values(phi)[action]
        if done:
            target = reward
        else:
            phi_next = self._firing_strengths(next_state)
            target = reward + self.gamma * np.max(self.q_values(phi_next))
        td_error = target - q_sa

        if self.use_eligibility:
            self.e_trace[:, action] += phi
            self.theta += self.alpha * td_error * self.e_trace
            if was_greedy and not done:
                self.e_trace *= self.gamma * self.lam
            else:
                self.e_trace.fill(0.0)
        else:
            self.theta[:, action] += self.alpha * td_error * phi

        return td_error

    def decay_epsilon(self):
        if self.use_adaptive_epsilon:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def make_agent(env_name, variant="full_gta", seed=0, **overrides):
    """variant: 'baseline' | 'no_eligibility' | 'no_gaussian' | 'full_gta'
    (leave-one-out ablation) or 'gaussian_only' | 'gaussian_eligibility'
    (incremental/additive ablation -- baseline and full_gta double as that
    path's first and last steps), plus 'epsilon_only' | 'eligibility_only'
    to complete the full 2^3 factorial (all 8 combinations of the three
    design choices) used by the ablation study."""
    spec = ENV_SPECS[env_name]
    flags = {
        "baseline": dict(use_gaussian=False, use_eligibility=False, use_adaptive_epsilon=False),
        "no_eligibility": dict(use_gaussian=True, use_eligibility=False, use_adaptive_epsilon=True),
        "no_gaussian": dict(use_gaussian=False, use_eligibility=True, use_adaptive_epsilon=True),
        "full_gta": dict(use_gaussian=True, use_eligibility=True, use_adaptive_epsilon=True),
        # Incremental path: each step adds exactly one component on top of the last.
        "gaussian_only": dict(use_gaussian=True, use_eligibility=False, use_adaptive_epsilon=False),
        "gaussian_eligibility": dict(use_gaussian=True, use_eligibility=True, use_adaptive_epsilon=False),
        # Remaining two corners of the 2^3 factorial cube.
        "epsilon_only": dict(use_gaussian=False, use_eligibility=False, use_adaptive_epsilon=True),
        "eligibility_only": dict(use_gaussian=False, use_eligibility=True, use_adaptive_epsilon=False),
    }[variant]
    kwargs = dict(state_ranges=spec["state_ranges"], sets_per_dim=spec["sets_per_dim"],
                  n_actions=spec["n_actions"], seed=seed)
    kwargs.update(flags)
    kwargs.update(overrides)
    return FuzzyQLearningAgent(**kwargs)


def train(agent, env_name, episodes, seed=0, track_state_visits=False):
    """Trains agent on env_name; returns returns/TD-error/epsilon histories and,
    optionally, every visited raw state (for a state-visitation heatmap)."""
    max_steps = ENV_SPECS[env_name]["max_steps"]
    env = gym.make(env_name)
    returns, td_errors, epsilons = [], [], []
    visited = [] if track_state_visits else None
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        agent.reset_traces()
        total = 0.0
        ep_td = []
        for _ in range(max_steps):
            action, phi, was_greedy = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            td_error = agent.update(phi, action, reward, next_state, terminated, was_greedy)
            ep_td.append(abs(td_error))
            if track_state_visits:
                visited.append(np.asarray(state, dtype=float))
            state = next_state
            total += reward
            if done:
                break
        agent.decay_epsilon()
        returns.append(total)
        td_errors.append(float(np.mean(ep_td)) if ep_td else 0.0)
        epsilons.append(agent.epsilon)
    env.close()
    out = {"returns": np.array(returns), "td_errors": np.array(td_errors),
           "epsilons": np.array(epsilons)}
    if track_state_visits:
        out["visited_states"] = np.array(visited)
    return out


if __name__ == "__main__":
    for env_name in ENV_SPECS:
        agent = make_agent(env_name, variant="full_gta", seed=0)
        result = train(agent, env_name, episodes=20, seed=0)
        print(f"{env_name}: rules={agent.n_rules}, "
              f"last-5 mean return={result['returns'][-5:].mean():.1f}")
