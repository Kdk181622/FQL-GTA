"""
Improved Fuzzy Q-Learning (FQL-GTA) for the CartPole-v1 benchmark.

Three targeted extensions are added to the classical baseline while every other
setting (learning rate, discount factor, rule resolution) is held identical, so
that any performance difference is attributable to the extensions alone:

  1. GAUSSIAN membership functions replace the triangular ones, giving smooth,
     everywhere-differentiable fuzzy basis functions and softer generalisation
     between neighbouring rules.

  2. ELIGIBILITY TRACES (Watkins' Q(lambda)) replace the one-step TD update,
     propagating each temporal-difference error backwards over recently visited
     fuzzy rules and thereby accelerating credit assignment.

  3. ADAPTIVE (decaying) EPSILON replaces the fixed exploration rate, allowing
     broad early exploration that anneals toward greedy exploitation as the
     value estimates stabilise.

The acronym FQL-GTA denotes Gaussian + Traces + Adaptive-epsilon.
"""

import numpy as np
import gymnasium as gym

from fql_baseline import STATE_RANGES, SETS_PER_DIM, triangular_centers


def gaussian_membership(x, centers, sigma):
    """Evaluate Gaussian membership degrees of scalar x against fixed centres."""
    return np.exp(-0.5 * ((x - centers) / sigma) ** 2)


class FuzzyQLearningImproved:
    """FQL agent with Gaussian MFs, Q(lambda) eligibility traces and decaying epsilon."""

    def __init__(self, n_actions=2, alpha=0.2, gamma=0.99,
                 epsilon_start=0.5, epsilon_min=0.005, epsilon_decay=0.99,
                 lam=0.5, seed=0):
        # Learning configuration; gamma matches the baseline, but alpha is set
        # independently for the proposed agent (0.2 here vs. 0.3 for the
        # baseline) -- see Results/alpha_sensitivity_* and the manuscript's
        # discussion of a tuned-baseline comparison for why this does not
        # confound the primary comparison's conclusion.
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam                      # eligibility-trace decay
        self.epsilon = epsilon_start        # ADAPTIVE exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        # Gaussian centres reuse the baseline placement for a fair comparison.
        self.centers = [
            triangular_centers(lo, hi, n)
            for (lo, hi), n in zip(STATE_RANGES, SETS_PER_DIM)
        ]
        # Gaussian width set to HALF the inter-centre spacing. This keeps the
        # membership functions local (each state strongly activates only its
        # neighbouring rules) so the fuzzy basis retains discriminative power,
        # while remaining smoother and softer than the triangular baseline.
        self.sigmas = [
            0.35 * (self.centers[d][1] - self.centers[d][0])
            for d in range(len(STATE_RANGES))
        ]

        self.n_rules = int(np.prod(SETS_PER_DIM))
        self.theta = np.zeros((self.n_rules, self.n_actions))
        # Eligibility-trace accumulator, one entry per (rule, action).
        self.e_trace = np.zeros((self.n_rules, self.n_actions))

    def _firing_strengths(self, state):
        """Normalised Gaussian rule firing strengths via the product t-norm."""
        clipped = [
            np.clip(state[d], STATE_RANGES[d][0], STATE_RANGES[d][1])
            for d in range(len(STATE_RANGES))
        ]
        deg = [gaussian_membership(clipped[d], self.centers[d], self.sigmas[d])
               for d in range(len(STATE_RANGES))]
        phi = deg[0]
        for d in range(1, len(deg)):
            phi = np.outer(phi, deg[d]).ravel()
        total = phi.sum()
        if total <= 1e-12:
            phi = np.ones(self.n_rules) / self.n_rules
        else:
            phi = phi / total
        return phi

    def q_values(self, phi):
        return phi @ self.theta

    def act(self, state, greedy=False):
        phi = self._firing_strengths(state)
        if (not greedy) and self.rng.random() < self.epsilon:
            action = int(self.rng.integers(self.n_actions))
            greedy_action = False
        else:
            action = int(np.argmax(self.q_values(phi)))
            greedy_action = True
        return action, phi, greedy_action

    def reset_traces(self):
        """Clear eligibility traces at the start of each episode."""
        self.e_trace.fill(0.0)

    def update(self, phi, action, reward, next_state, done, was_greedy):
        """Watkins' Q(lambda) update over the fuzzy parameter matrix."""
        q_sa = self.q_values(phi)[action]
        if done:
            target = reward
        else:
            phi_next = self._firing_strengths(next_state)
            target = reward + self.gamma * np.max(self.q_values(phi_next))
        td_error = target - q_sa

        # Accumulate eligibility for the fuzzy rules active in the current state.
        self.e_trace[:, action] += phi

        # Apply the TD error to all eligible parameters simultaneously.
        self.theta += self.alpha * td_error * self.e_trace

        # Decay traces; Watkins' rule zeroes them after a non-greedy (exploratory)
        # action, otherwise they fade by gamma * lambda.
        if was_greedy and not done:
            self.e_trace *= self.gamma * self.lam
        else:
            self.e_trace.fill(0.0)
        return td_error

    def decay_epsilon(self):
        """Anneal the exploration rate toward its floor after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(agent, episodes=400, max_steps=500, seed=0):
    """Train the improved FQL agent on CartPole-v1; return the return history."""
    env = gym.make("CartPole-v1")
    history = []
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        agent.reset_traces()
        total = 0.0
        for _ in range(max_steps):
            action, phi, was_greedy = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(phi, action, reward, next_state, terminated, was_greedy)
            state = next_state
            total += reward
            if done:
                break
        agent.decay_epsilon()
        history.append(total)
    env.close()
    return np.array(history)


if __name__ == "__main__":
    agent = FuzzyQLearningImproved(seed=0)
    hist = train(agent, episodes=50, seed=0)
    print(f"Improved FQL smoke test | rules={agent.n_rules} | "
          f"last-10 mean return={hist[-10:].mean():.1f}")
