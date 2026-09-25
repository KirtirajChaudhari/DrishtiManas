"""A multilayer perceptron (fully connected feed-forward network) written from scratch.

Notation for layer l (1..L):

    Z[l] = A[l-1] @ W[l] + b[l]          (affine transform)
    A[l] = g[l](Z[l])                    (activation; softmax for the output layer)

Forward propagation evaluates these equations from the input A[0] = X to the
output probabilities. Backpropagation applies the chain rule in reverse:

    dZ[L] = dJ/dZ[L]                     (given by the loss, e.g. (P - Y) / N)
    dW[l] = A[l-1].T @ dZ[l] + lambda * W[l]
    db[l] = sum over batch of dZ[l]
    dA[l-1] = dZ[l] @ W[l].T
    dZ[l-1] = dA[l-1] * g'[l-1](Z[l-1])

and the optimizer uses dW, db to update the parameters.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .activations import get_activation, softmax
from .losses import LOSSES, one_hot
from .optimizers import make_optimizer


@dataclass
class MLPConfig:
    input_dim: int
    num_classes: int
    hidden_layers: list[int] = field(default_factory=lambda: [128])
    activation: str = "relu"
    loss: str = "cross_entropy"
    optimizer: str = "momentum"
    learning_rate: float = 0.01
    batch_size: int = 64
    epochs: int = 30
    l2: float = 1e-4
    lr_decay: float = 1.0  # multiply learning rate by this after every epoch
    early_stopping_patience: int | None = 8
    class_weights: list[float] | None = None
    dropout: float = 0.0  # inverted dropout applied to hidden-layer activations during training only
    dtype: str = "float32"  # float32 trains ~2x faster; gradient checks use float64
    seed: int = 42


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float | None
    val_acc: float | None
    grad_norm: float
    learning_rate: float
    seconds: float


class MLP:
    def __init__(self, config: MLPConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.dtype = np.dtype(config.dtype)
        self.hidden_act = get_activation(config.activation)
        sizes = [config.input_dim, *config.hidden_layers, config.num_classes]
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            self.weights.append(self._init_weight(fan_in, fan_out).astype(self.dtype))
            self.biases.append(np.zeros(fan_out, dtype=self.dtype))
        class_w = None if config.class_weights is None else np.asarray(config.class_weights, dtype=np.float64)
        self.loss_fn = LOSSES[config.loss](class_weights=class_w)
        self.history: list[EpochStats] = []

    # ------------------------------------------------------------------ init
    def _init_weight(self, fan_in: int, fan_out: int) -> np.ndarray:
        # He initialisation suits ReLU-family activations; Xavier/Glorot suits
        # sigmoid/tanh. Both keep the activation variance stable across layers.
        if self.config.activation in {"relu", "leaky_relu"}:
            std = np.sqrt(2.0 / fan_in)
        else:
            std = np.sqrt(1.0 / fan_in)
        return self.rng.normal(0.0, std, size=(fan_in, fan_out))

    @property
    def params(self) -> list[np.ndarray]:
        return [*self.weights, *self.biases]

    @property
    def num_parameters(self) -> int:
        return int(sum(p.size for p in self.params))

    # --------------------------------------------------------------- forward
    def forward(
        self, X: np.ndarray, training: bool = False
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray | None]]:
        """Return output logits plus caches needed by backprop.

        ``activations[l]`` is what actually fed the next layer (post-dropout when
        ``training`` and ``dropout > 0``); ``raw_activations[l]`` is the undropped
        ``g(Z[l])``, which activation derivatives (e.g. sigmoid, tanh) must be
        evaluated against; ``masks[l]`` is the inverted-dropout mask used to turn
        one into the other (``None`` when dropout was inactive for that layer).
        """
        A = X = np.asarray(X, dtype=self.dtype)
        activations = [X]
        raw_activations = [X]
        pre_acts: list[np.ndarray] = []
        masks: list[np.ndarray | None] = [None]
        last = len(self.weights) - 1
        for i, (W, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            Z = A @ W + b
            pre_acts.append(Z)
            if i == last:
                return Z, pre_acts, activations, raw_activations, masks  # logits; softmax applied in the loss
            g = self.hidden_act.forward(Z)
            raw_activations.append(g)
            mask = None
            if training and self.config.dropout > 0:
                keep = 1.0 - self.config.dropout
                mask = (self.rng.random(g.shape) < keep).astype(self.dtype) / keep
            masks.append(mask)
            A = g if mask is None else g * mask
            activations.append(A)
        raise RuntimeError("network has no layers")

    def predict_proba(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        out = []
        for start in range(0, X.shape[0], batch_size):
            logits, *_ = self.forward(X[start : start + batch_size])
            out.append(softmax(logits))
        return np.vstack(out)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    # -------------------------------------------------------------- backward
    def loss_and_gradients(
        self, X: np.ndarray, Y: np.ndarray, training: bool = True
    ) -> tuple[float, list[np.ndarray], np.ndarray]:
        """One forward + backward pass. Returns (loss, grads in `params` order, probs).

        ``training=True`` (the default, used by ``fit``) activates dropout when
        configured; ``gradient_check`` calls with ``training=False`` so the
        numerical and analytic gradients compare a deterministic function.
        """
        logits, pre_acts, activations, raw_activations, masks = self.forward(X, training=training)
        data_loss, dZ, probs = self.loss_fn(logits, Y.astype(self.dtype, copy=False))
        dZ = dZ.astype(self.dtype, copy=False)
        l2 = self.config.l2
        reg_loss = 0.5 * l2 * sum(float(np.sum(W * W)) for W in self.weights)

        n_layers = len(self.weights)
        dWs: list[np.ndarray] = [np.empty(0)] * n_layers
        dbs: list[np.ndarray] = [np.empty(0)] * n_layers
        for layer in reversed(range(n_layers)):
            A_prev = activations[layer]
            dWs[layer] = A_prev.T @ dZ + l2 * self.weights[layer]
            dbs[layer] = dZ.sum(axis=0)
            if layer > 0:
                # Gradient wrt activations[layer] (the post-dropout output of hidden layer `layer`).
                dA = dZ @ self.weights[layer].T
                mask = masks[layer]
                if mask is not None:
                    dA = dA * mask  # backprop through the dropout multiply
                dZ = dA * self.hidden_act.backward(pre_acts[layer - 1], raw_activations[layer])
        return data_loss + reg_loss, [*dWs, *dbs], probs

    # ------------------------------------------------------------------- fit
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        probs = self.predict_proba(X)
        Y = one_hot(y, self.config.num_classes)
        logp = np.log(probs + 1e-12)
        per_sample = -np.sum(Y * logp, axis=1)
        if self.loss_fn.class_weights is not None:
            per_sample = per_sample * (Y @ self.loss_fn.class_weights)
        return float(per_sample.mean()), float((probs.argmax(axis=1) == y).mean())

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        on_epoch: Callable[[EpochStats], None] | None = None,
    ) -> list[EpochStats]:
        cfg = self.config
        optimizer = make_optimizer(cfg.optimizer, cfg.learning_rate)
        X = np.asarray(X, dtype=self.dtype)
        Y = one_hot(y, cfg.num_classes).astype(self.dtype)
        n = X.shape[0]
        best_val = np.inf
        best_params: list[np.ndarray] | None = None
        bad_epochs = 0

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.perf_counter()
            order = self.rng.permutation(n)
            losses, correct, grad_norms = [], 0, []
            for start in range(0, n, cfg.batch_size):
                idx = order[start : start + cfg.batch_size]
                loss, grads, probs = self.loss_and_gradients(X[idx], Y[idx])
                optimizer.step(self.params, grads)
                losses.append(loss * len(idx))
                correct += int((probs.argmax(axis=1) == y[idx]).sum())
                grad_norms.append(float(np.sqrt(sum(float(np.sum(g * g)) for g in grads))))

            val_loss = val_acc = None
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)

            stats = EpochStats(
                epoch=epoch,
                train_loss=float(np.sum(losses) / n),
                train_acc=correct / n,
                val_loss=val_loss,
                val_acc=val_acc,
                grad_norm=float(np.mean(grad_norms)),
                learning_rate=optimizer.lr,
                seconds=time.perf_counter() - t0,
            )
            self.history.append(stats)
            if on_epoch:
                on_epoch(stats)

            optimizer.lr *= cfg.lr_decay

            if val_loss is not None and cfg.early_stopping_patience:
                if val_loss < best_val - 1e-4:
                    best_val = val_loss
                    best_params = [p.copy() for p in self.params]
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= cfg.early_stopping_patience:
                        break

        if best_params is not None:
            for p, best in zip(self.params, best_params):
                p[...] = best
        return self.history

    # ------------------------------------------------------------ persistence
    def state_dict(self) -> dict[str, np.ndarray]:
        state = {f"W{i}": W for i, W in enumerate(self.weights)}
        state.update({f"b{i}": b for i, b in enumerate(self.biases)})
        return state

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        for i in range(len(self.weights)):
            self.weights[i] = np.asarray(state[f"W{i}"], dtype=self.dtype)
            self.biases[i] = np.asarray(state[f"b{i}"], dtype=self.dtype)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, config=json.dumps(asdict(self.config)), **self.state_dict())

    @classmethod
    def load(cls, path: str | Path) -> "MLP":
        data = np.load(path, allow_pickle=False)
        model = cls(MLPConfig(**json.loads(str(data["config"]))))
        model.load_state_dict({k: data[k] for k in data.files if k != "config"})
        return model


def gradient_check(
    model: MLP,
    X: np.ndarray,
    y: np.ndarray,
    num_checks: int = 30,
    eps: float = 1e-5,
    dropout_seed: int | None = None,
) -> float:
    """Compare backprop gradients to centred finite differences.

    Returns the maximum relative error over randomly sampled parameters; values
    around 1e-7 or smaller mean the analytic gradients are correct.

    ``dropout_seed``, if given, resets ``model.rng`` to that seed before every
    forward pass so a config with ``dropout > 0`` still sees an identical mask
    on every evaluation (the mask depends only on shapes, never on the
    parameter being perturbed) -- this lets the check validate the dropout
    backward pass too. When ``None`` (the default), the check runs with
    ``training=False`` so dropout is inactive and the comparison is exact
    regardless of configuration.
    """
    Y = one_hot(y, model.config.num_classes)
    training = dropout_seed is not None

    def loss_at() -> float:
        if dropout_seed is not None:
            model.rng = np.random.default_rng(dropout_seed)
        loss, _, _ = model.loss_and_gradients(X, Y, training=training)
        return loss

    if dropout_seed is not None:
        model.rng = np.random.default_rng(dropout_seed)
    _, grads, _ = model.loss_and_gradients(X, Y, training=training)

    rng = np.random.default_rng(0)
    worst = 0.0
    for p, g in zip(model.params, grads, strict=True):
        for _ in range(max(1, num_checks // len(model.params))):
            idx = tuple(rng.integers(0, s) for s in p.shape)
            old = p[idx]
            p[idx] = old + eps
            plus = loss_at()
            p[idx] = old - eps
            minus = loss_at()
            p[idx] = old
            numeric = (plus - minus) / (2 * eps)
            analytic = g[idx]
            denom = max(abs(numeric) + abs(analytic), 1e-7)  # floor: dead ReLUs give 0 vs ~1e-12 noise
            worst = max(worst, abs(numeric - analytic) / denom)
    return worst
