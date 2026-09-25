"""Activation functions implemented from scratch with NumPy.

Every activation exposes two methods:

* ``forward(z)``      -> a = g(z)
* ``backward(z, a)``  -> g'(z), the element-wise derivative used by backprop.

Both the pre-activation ``z`` and the activation ``a`` are passed to
``backward`` because some derivatives are cheapest to express in terms of the
output (sigmoid: a * (1 - a), tanh: 1 - a**2).
"""
from __future__ import annotations

import numpy as np


class Activation:
    name = "base"

    def forward(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.name


class ReLU(Activation):
    """g(z) = max(0, z). Cheap, non-saturating for z > 0, can 'die' for z < 0."""

    name = "relu"

    def forward(self, z):
        return np.maximum(0.0, z)

    def backward(self, z, a):
        return (z > 0).astype(z.dtype)


class LeakyReLU(Activation):
    """g(z) = z if z > 0 else alpha * z. Keeps a small gradient for z < 0."""

    name = "leaky_relu"

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha

    def forward(self, z):
        return np.where(z > 0, z, self.alpha * z)

    def backward(self, z, a):
        return np.where(z > 0, 1.0, self.alpha).astype(z.dtype)


class Sigmoid(Activation):
    """g(z) = 1 / (1 + e^-z). Squashes to (0, 1); saturates -> vanishing gradients."""

    name = "sigmoid"

    def forward(self, z):
        # Numerically stable piecewise formulation (avoids overflow in exp).
        out = np.empty_like(z)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def backward(self, z, a):
        return a * (1.0 - a)


class Tanh(Activation):
    """g(z) = tanh(z). Zero-centred version of the sigmoid, range (-1, 1)."""

    name = "tanh"

    def forward(self, z):
        return np.tanh(z)

    def backward(self, z, a):
        return 1.0 - a**2


class Linear(Activation):
    name = "linear"

    def forward(self, z):
        return z

    def backward(self, z, a):
        return np.ones_like(z)


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax used by the output layer (turns logits into probabilities)."""
    shifted = z - z.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


ACTIVATIONS: dict[str, type[Activation]] = {
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "linear": Linear,
}


def get_activation(name: str) -> Activation:
    try:
        return ACTIVATIONS[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown activation '{name}'. Choose from {sorted(ACTIVATIONS)}") from exc
