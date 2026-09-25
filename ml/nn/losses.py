"""Cost / error functions.

The network's output layer produces logits ``z``; softmax turns them into class
probabilities ``p``. Each loss returns the scalar cost *and* the gradient of
that cost with respect to the logits (dJ/dz), which is where backpropagation
starts.
"""

from __future__ import annotations

import numpy as np

from .activations import softmax

EPS = 1e-12


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


class CrossEntropyLoss:
    """Categorical cross-entropy with a softmax output.

        J = -(1/N) * sum_i sum_k  w_k * y_ik * log(p_ik)

    ``w_k`` are optional class weights (used to counter class imbalance).
    Combined with softmax the gradient w.r.t. the logits has the famously
    simple form  dJ/dz = w_(y_i) * (p - y) / N.
    """

    name = "cross_entropy"

    def __init__(self, class_weights: np.ndarray | None = None) -> None:
        self.class_weights = class_weights

    def __call__(self, logits: np.ndarray, y_onehot: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        n = logits.shape[0]
        probs = softmax(logits)
        sample_w = (
            np.ones(n, dtype=logits.dtype)
            if self.class_weights is None
            else (y_onehot @ self.class_weights).astype(logits.dtype)
        )
        per_sample = -np.sum(y_onehot * np.log(probs + EPS), axis=1)
        loss = float(np.sum(sample_w * per_sample) / n)
        grad = (probs - y_onehot) * sample_w[:, None] / n
        return loss, grad, probs


class MSELoss:
    """Mean squared error between softmax probabilities and one-hot targets.

        J = (1/N) * sum_i sum_k (p_ik - y_ik)^2

    Included for comparison: it is a valid cost, but its gradient through the
    softmax is small when the network is confidently wrong, so it learns slower
    than cross-entropy on classification problems.
    """

    name = "mse"

    def __init__(self, class_weights: np.ndarray | None = None) -> None:
        self.class_weights = class_weights

    def __call__(self, logits, y_onehot):
        n = logits.shape[0]
        probs = softmax(logits)
        sample_w = (
            np.ones(n, dtype=logits.dtype)
            if self.class_weights is None
            else (y_onehot @ self.class_weights).astype(logits.dtype)
        )
        diff = probs - y_onehot
        loss = float(np.sum(sample_w * np.sum(diff**2, axis=1)) / n)
        dp = 2.0 * diff * sample_w[:, None] / n
        # Backprop through softmax: dJ/dz_j = p_j * (dJ/dp_j - sum_k dJ/dp_k p_k)
        grad = probs * (dp - np.sum(dp * probs, axis=1, keepdims=True))
        return loss, grad, probs


LOSSES = {"cross_entropy": CrossEntropyLoss, "mse": MSELoss}
