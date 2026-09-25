"""Parameter update rules applied after backpropagation computes the gradients."""

from __future__ import annotations

import numpy as np


class SGD:
    """Mini-batch gradient descent with optional (classical) momentum.

    v <- mu * v - lr * dW
    W <- W + v
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9) -> None:
        self.lr = lr
        self.momentum = momentum
        self._velocity: dict[int, np.ndarray] = {}

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        for i, (p, g) in enumerate(zip(params, grads)):
            if self.momentum:
                v = self._velocity.get(i)
                if v is None:
                    v = np.zeros_like(p)
                v = self.momentum * v - self.lr * g
                self._velocity[i] = v
                p += v
            else:
                p -= self.lr * g


class Adam:
    """Adam: per-parameter adaptive learning rates from 1st/2nd moment estimates."""

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self._m: dict[int, np.ndarray] = {}
        self._v: dict[int, np.ndarray] = {}

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            m = self._m.get(i, np.zeros_like(p))
            v = self._v.get(i, np.zeros_like(p))
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * g * g
            self._m[i], self._v[i] = m, v
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def make_optimizer(name: str, lr: float):
    if name == "sgd":
        return SGD(lr=lr, momentum=0.0)
    if name == "momentum":
        return SGD(lr=lr, momentum=0.9)
    if name == "adam":
        return Adam(lr=lr)
    raise ValueError(f"Unknown optimizer '{name}'")
