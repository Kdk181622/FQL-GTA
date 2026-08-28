"""A minimal, real DQN (Mnih et al., 2015 style) for CartPole-v1: a 2-hidden-
-layer MLP Q-network, experience replay, a periodically-synced target network,
and epsilon-greedy exploration. Kept deliberately small and dependency-light
(PyTorch + Gymnasium only) so its parameter count, memory footprint, and
training time can be measured directly and compared against FQL/FQL-GTA on
equal footing, rather than asserting a generic "typical DQN" figure.
"""
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

HIDDEN_SIZE = 64  # a genuinely "small" DQN, not an inflated strawman


class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity, seed=0):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = self.rng.sample(self.buffer, batch_size)
        s, a, r, s2, done = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32), np.array(s2, dtype=np.float32),
                np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def train_dqn(episodes=400, max_steps=500, seed=0,
              gamma=0.99, lr=1e-3, batch_size=64, buffer_capacity=10000,
              epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995,
              target_sync_every=10, min_buffer_before_learning=500):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_net = QNetwork(state_dim, n_actions)
    target_net = QNetwork(state_dim, n_actions)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity, seed=seed)

    epsilon = epsilon_start
    rng = np.random.default_rng(seed)
    returns = []

    t0 = time.time()
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        total = 0.0
        for _ in range(max_steps):
            if rng.random() < epsilon:
                action = int(rng.integers(n_actions))
            else:
                with torch.no_grad():
                    q = q_net(torch.as_tensor(state, dtype=torch.float32))
                    action = int(torch.argmax(q).item())
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, float(terminated))
            state = next_state
            total += reward

            if len(buffer) >= max(batch_size, min_buffer_before_learning):
                s, a, r, s2, d = buffer.sample(batch_size)
                s_t = torch.as_tensor(s)
                a_t = torch.as_tensor(a)
                r_t = torch.as_tensor(r)
                s2_t = torch.as_tensor(s2)
                d_t = torch.as_tensor(d)

                q_sa = q_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = target_net(s2_t).max(1).values
                    target = r_t + gamma * (1.0 - d_t) * q_next
                loss = nn.functional.mse_loss(q_sa, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if ep % target_sync_every == 0:
            target_net.load_state_dict(q_net.state_dict())
        returns.append(total)
    env.close()
    train_time = time.time() - t0

    return {
        "returns": np.array(returns),
        "train_time_sec": train_time,
        "n_params": n_params(q_net),
        "n_params_with_target": n_params(q_net) + n_params(target_net),
        "buffer_capacity": buffer_capacity,
        "hidden_size": HIDDEN_SIZE,
    }


if __name__ == "__main__":
    out = train_dqn(episodes=50, seed=0)
    print(f"DQN smoke test | params={out['n_params']} | "
          f"last-10 mean return={out['returns'][-10:].mean():.1f} | "
          f"time={out['train_time_sec']:.1f}s")
