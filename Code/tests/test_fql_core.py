"""Unit tests for the fuzzy-inference and eligibility-trace mechanics in
fql_core.py. Run with: python -m pytest tests/ -v  (from src/), or
python -m unittest discover tests  if pytest is not installed.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fql_core import (
    FuzzyQLearningAgent, ENV_SPECS, make_agent,
    triangular_membership, gaussian_membership, triangular_centers,
)


class TestMembershipFunctions(unittest.TestCase):
    def test_triangular_membership_peaks_at_one(self):
        centers = triangular_centers(-1.0, 1.0, 3)
        degrees = triangular_membership(centers[1], centers)  # exactly at a centre
        self.assertAlmostEqual(degrees[1], 1.0, places=6)

    def test_triangular_membership_bounded(self):
        centers = triangular_centers(-2.4, 2.4, 4)
        degrees = triangular_membership(10.0, centers)  # far outside range
        self.assertTrue(np.all(degrees >= 0.0))
        self.assertTrue(np.all(degrees <= 1.0))

    def test_gaussian_membership_peaks_at_one(self):
        centers = np.array([-1.0, 0.0, 1.0])
        degrees = gaussian_membership(0.0, centers, sigma=0.5)
        self.assertAlmostEqual(degrees[1], 1.0, places=6)

    def test_gaussian_membership_symmetric(self):
        centers = np.array([0.0])
        left = gaussian_membership(-0.5, centers, sigma=0.3)
        right = gaussian_membership(0.5, centers, sigma=0.3)
        self.assertAlmostEqual(left[0], right[0], places=6)


class TestFiringStrengths(unittest.TestCase):
    def setUp(self):
        self.agent = FuzzyQLearningAgent(
            state_ranges=[(-1.0, 1.0), (-1.0, 1.0)], sets_per_dim=[3, 3],
            n_actions=2, use_gaussian=True, use_eligibility=True,
            use_adaptive_epsilon=True, seed=0,
        )

    def test_firing_strengths_sum_to_one(self):
        phi = self.agent._firing_strengths(np.array([0.0, 0.0]))
        self.assertAlmostEqual(phi.sum(), 1.0, places=6)

    def test_firing_strengths_nonnegative(self):
        phi = self.agent._firing_strengths(np.array([0.7, -0.3]))
        self.assertTrue(np.all(phi >= 0.0))

    def test_n_rules_matches_product_of_sets(self):
        self.assertEqual(self.agent.n_rules, 9)  # 3 x 3

    def test_state_clipping_does_not_crash(self):
        # State well outside the declared range should still fuzzify cleanly.
        phi = self.agent._firing_strengths(np.array([100.0, -100.0]))
        self.assertAlmostEqual(phi.sum(), 1.0, places=6)


class TestEligibilityTraces(unittest.TestCase):
    def test_traces_reset_to_zero(self):
        agent = FuzzyQLearningAgent(
            state_ranges=[(-1.0, 1.0)], sets_per_dim=[3], n_actions=2,
            use_eligibility=True, seed=0,
        )
        agent.e_trace[:] = 5.0
        agent.reset_traces()
        self.assertTrue(np.all(agent.e_trace == 0.0))

    def test_no_eligibility_variant_has_no_trace(self):
        agent = FuzzyQLearningAgent(
            state_ranges=[(-1.0, 1.0)], sets_per_dim=[3], n_actions=2,
            use_eligibility=False, seed=0,
        )
        self.assertIsNone(agent.e_trace)

    def test_update_reduces_td_error_over_repeated_visits(self):
        # A stationary (state, action, reward) pair's TD error should shrink
        # in magnitude as theta converges toward the target, for a fixed next
        # state (done=True removes bootstrapping, isolating the update rule).
        agent = make_agent("CartPole-v1", variant="full_gta", seed=0)
        state = np.array([0.0, 0.0, 0.0, 0.0])
        first_error = None
        last_error = None
        for _ in range(50):
            action, phi, was_greedy = agent.act(state, greedy=True)
            err = agent.update(phi, action, reward=1.0, next_state=state, done=True, was_greedy=was_greedy)
            if first_error is None:
                first_error = abs(err)
            last_error = abs(err)
        self.assertLess(last_error, first_error)


class TestMakeAgentVariants(unittest.TestCase):
    def test_all_four_variants_construct(self):
        for variant in ["baseline", "no_eligibility", "no_gaussian", "full_gta"]:
            agent = make_agent("CartPole-v1", variant=variant, seed=0)
            self.assertIsInstance(agent, FuzzyQLearningAgent)

    def test_baseline_variant_flags(self):
        agent = make_agent("CartPole-v1", variant="baseline", seed=0)
        self.assertFalse(agent.use_gaussian)
        self.assertFalse(agent.use_eligibility)
        self.assertFalse(agent.use_adaptive_epsilon)

    def test_full_gta_variant_flags(self):
        agent = make_agent("CartPole-v1", variant="full_gta", seed=0)
        self.assertTrue(agent.use_gaussian)
        self.assertTrue(agent.use_eligibility)
        self.assertTrue(agent.use_adaptive_epsilon)

    def test_env_specs_rule_counts(self):
        for env_name, spec in ENV_SPECS.items():
            expected = 1
            for n in spec["sets_per_dim"]:
                expected *= n
            agent = make_agent(env_name, variant="full_gta", seed=0)
            self.assertEqual(agent.n_rules, expected)


class TestEpsilonDecay(unittest.TestCase):
    def test_adaptive_epsilon_decreases(self):
        agent = make_agent("CartPole-v1", variant="full_gta", seed=0)
        start = agent.epsilon
        for _ in range(10):
            agent.decay_epsilon()
        self.assertLess(agent.epsilon, start)

    def test_adaptive_epsilon_floors_at_minimum(self):
        agent = make_agent("CartPole-v1", variant="full_gta", seed=0)
        for _ in range(10000):
            agent.decay_epsilon()
        self.assertGreaterEqual(agent.epsilon, agent.epsilon_min)
        self.assertAlmostEqual(agent.epsilon, agent.epsilon_min, places=6)

    def test_fixed_epsilon_never_changes(self):
        agent = make_agent("CartPole-v1", variant="baseline", seed=0)
        start = agent.epsilon
        for _ in range(10):
            agent.decay_epsilon()
        self.assertEqual(agent.epsilon, start)


if __name__ == "__main__":
    unittest.main()
