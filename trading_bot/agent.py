"""Linear softmax policy trained with policy gradient (REINFORCE with a baseline)."""
import numpy as np

from env import ACTIONS


class Agent:
    def __init__(self, n_features: int, rng: np.random.Generator, lr: float = 0.01,
                 weights: np.ndarray | None = None):
        self.rng = rng
        self.lr = lr
        d = n_features + 3  # + one-hot of current position
        self.W = weights.copy() if weights is not None else rng.normal(0, 0.01, (d, 3))
        self.baseline = 0.0

    def _x(self, feat, pos):
        onehot = np.zeros(3)
        onehot[pos + 1] = 1.0
        return np.concatenate([feat, onehot])

    def _probs(self, x):
        z = x @ self.W
        z -= z.max()
        p = np.exp(z)
        return p / p.sum()

    def act(self, feat, pos, greedy=False):
        p = self._probs(self._x(feat, pos))
        i = int(p.argmax()) if greedy else int(self.rng.choice(3, p=p))
        return int(ACTIONS[i])

    def train_episode(self, X, rets, cost, start, length):
        """One stochastic episode; update weights from per-step rewards."""
        pos, grads, rewards = 0, [], []
        for t in range(start, start + length):
            x = self._x(X[t], pos)
            p = self._probs(x)
            i = int(self.rng.choice(3, p=p))
            new_pos = int(ACTIONS[i])
            r = new_pos * rets[t] - cost * abs(new_pos - pos)
            g = -np.outer(x, p)
            g[:, i] += x
            grads.append(g)
            rewards.append(r)
            pos = new_pos
        rewards = np.array(rewards)
        scale = rewards.std() + 1e-8
        adv = (rewards - self.baseline) / scale
        self.baseline = 0.9 * self.baseline + 0.1 * rewards.mean()
        self.W += self.lr * sum(a * g for a, g in zip(adv, grads)) / length
