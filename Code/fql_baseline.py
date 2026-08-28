"""
Baseline Fuzzy Q-Learning (FQL) controller for the CartPole-v1 benchmark.

This module reproduces the classical Fuzzy Q-Learning formulation (Jouffe, 1998)
using a Fuzzy Inference System (FIS) as a linear function approximator over the
continuous state space. Each input dimension is partitioned into TRIANGULAR
membership functions; the Cartesian product of the fuzzy sets defines the rule
base. The action-value function is approximated as a firing-strength-weighted
sum of per-rule action parameters, and is updated with the standard one-step
temporal-difference (TD) rule under a FIXED exploration rate (epsilon).

The class is deliberately kept dependency-light (NumPy + Gymnasium only) so that
the notebook and the experiment runner execute first-to-last without error.
"""

import numpy as np
import gymnasium as gym


# Bounded operating ranges used to place membership functions along each state
# dimension. CartPole cart-velocity and pole angular-velocity are theoretically
# unbounded, so practical clipping limits are applied.
STATE_RANGES = [
    (-2.4, 2.4),    # cart position
    (-3.0, 3.0),    # cart velocity (clipped)
    (-0.21, 0.21),  # pole angle (radians, ~12 degrees)
    (-3.5, 3.5),    # pole angular velocity (clipped)
]

# Number of triangular fuzzy sets per state dimension. More resolution is given
# to the pole angle and pole angular velocity, which dominate balancing.
SETS_PER_DIM = [3, 3, 6, 6]


def triangular_centers(low, high, n_sets):
    """Return evenly spaced centres for n_sets triangular membership functions."""
    return np.linspace(low, high, n_sets)


def triangular_membership(x, centers):
    """
    Evaluate triangular membership degrees of a scalar x against a set of
    evenly spaced triangular functions. Adjacent triangles overlap so that the
    degrees form a partition of unity across the operating range.
    """
    # Uniform spacing between adjacent centres defines the triangle half-width.
    width = centers[1] - centers[0]
    # Distance from x to each centre, scaled by the width, clipped to [0, 1].
    degrees = np.maximum(0.0, 1.0 - np.abs(x - centers) / width)
    return degrees


class FuzzyQLearningBaseline:
    """Classical Fuzzy Q-Learning agent with triangular MFs and fixed epsilon."""

    def __init__(self, n_actions=2, alpha=0.3, gamma=0.99, epsilon=0.1, seed=0):
        # Learning configuration.
        self.n_actions = n_actions
        self.alpha = alpha        # learning rate
        self.gamma = gamma        # discount factor
        self.epsilon = epsilon    # FIXED exploration rate (baseline)
        self.rng = np.random.default_rng(seed)

        # Pre-compute the triangular membership-function centres per dimension.
        self.centers = [
            triangular_centers(lo, hi, n)
            for (lo, hi), n in zip(STATE_RANGES, SETS_PER_DIM)
        ]

        # Total number of fuzzy rules is the product of sets across dimensions.
        self.n_rules = int(np.prod(SETS_PER_DIM))

        # Parameter matrix theta[rule, action] approximates the action value as
        # Q(s, a) = sum_i phi_i(s) * theta[i, a], with phi the normalised firing
        # strengths. Initialised to zero for an unbiased start.
        self.theta = np.zeros((self.n_rules, self.n_actions))

    def _firing_strengths(self, state):
        """
        Compute normalised rule firing strengths (fuzzy basis functions) for a
        given continuous state using the product t-norm across dimensions.
        """
        # Clip the raw state into the bounded operating ranges before fuzzifying.
        clipped = [
            np.clip(state[d], STATE_RANGES[d][0], STATE_RANGES[d][1])
            for d in range(len(STATE_RANGES))
        ]
        # Per-dimension membership degree vectors.
        deg = [triangular_membership(clipped[d], self.centers[d])
               for d in range(len(STATE_RANGES))]

        # Outer product across dimensions yields one firing strength per rule.
        phi = deg[0]
        for d in range(1, len(deg)):
            phi = np.outer(phi, deg[d]).ravel()

        total = phi.sum()
        if total <= 1e-12:
            # Degenerate fallback: uniform activation if nothing fires.
            phi = np.ones(self.n_rules) / self.n_rules
        else:
            phi = phi / total
        return phi

    def q_values(self, phi):
        """Return the action-value vector Q(s, .) for the given basis vector."""
        return phi @ self.theta

    def act(self, state, greedy=False):
        """Epsilon-greedy action selection over the fuzzy action values."""
        phi = self._firing_strengths(state)
        if (not greedy) and self.rng.random() < self.epsilon:
            action = int(self.rng.integers(self.n_actions))
        else:
            action = int(np.argmax(self.q_values(phi)))
        return action, phi

    def update(self, phi, action, reward, next_state, done):
        """One-step tabular-style TD update on the fuzzy parameters."""
        q_sa = self.q_values(phi)[action]
        if done:
            target = reward
        else:
            phi_next = self._firing_strengths(next_state)
            target = reward + self.gamma * np.max(self.q_values(phi_next))
        td_error = target - q_sa
        # Gradient of Q wrt theta[:, action] is exactly phi, so the update is
        # distributed across active rules in proportion to their firing strength.
        self.theta[:, action] += self.alpha * td_error * phi
        return td_error


def train(agent, episodes=400, max_steps=500, seed=0):
    """
    Train an FQL agent on CartPole-v1 and return the per-episode return history.
    A reward-shaping-free setup is used: the native +1-per-step reward applies.
    """
    env = gym.make("CartPole-v1")
    history = []
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        total = 0.0
        for _ in range(max_steps):
            action, phi = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(phi, action, reward, next_state, terminated)
            state = next_state
            total += reward
            if done:
                break
        history.append(total)
    env.close()
    return np.array(history)


if __name__ == "__main__":
    # Smoke test: a single short run to confirm the module executes cleanly.
    agent = FuzzyQLearningBaseline(seed=0)
    hist = train(agent, episodes=50, seed=0)
    print(f"Baseline FQL smoke test | rules={agent.n_rules} | "
          f"last-10 mean return={hist[-10:].mean():.1f}")
