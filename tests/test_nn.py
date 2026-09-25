"""Unit tests for the from-scratch neural network."""

import numpy as np
import pytest

from ml.metrics import classification_report
from ml.nn import MLP, MLPConfig, get_activation, gradient_check, softmax


@pytest.mark.parametrize("activation", ["relu", "leaky_relu", "sigmoid", "tanh"])
@pytest.mark.parametrize("loss", ["cross_entropy", "mse"])
def test_backprop_matches_numerical_gradient(activation, loss):
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(16, 6)), rng.integers(0, 3, 16)
    model = MLP(
        MLPConfig(
            input_dim=6,
            num_classes=3,
            hidden_layers=[5, 4],
            activation=activation,
            loss=loss,
            l2=1e-2,
            class_weights=[1.0, 2.0, 0.5],
            dtype="float64",
        )
    )
    # Non-zero biases keep pre-activations off the ReLU kink at exactly z = 0,
    # where the finite-difference derivative is undefined.
    for b in model.biases:
        b[:] = rng.normal(0, 0.1, size=b.shape)
    assert gradient_check(model, X, y, num_checks=60) < 1e-5


def test_dropout_backward_matches_numerical_gradient():
    """Fix the RNG seed before every forward pass so the dropout mask is identical across the
    plus/minus/analytic evaluations (it depends only on shapes, never on the perturbed weight),
    which lets finite differences validate the dropout backward pass too."""
    rng = np.random.default_rng(5)
    X, y = rng.normal(size=(24, 6)), rng.integers(0, 3, 24)
    model = MLP(MLPConfig(input_dim=6, num_classes=3, hidden_layers=[5, 4], dropout=0.4, dtype="float64", seed=1))
    for b in model.biases:
        b[:] = rng.normal(0, 0.1, size=b.shape)
    assert gradient_check(model, X, y, num_checks=60, dropout_seed=123) < 1e-5


def test_dropout_zeros_activations_and_preserves_expectation():
    rng = np.random.default_rng(0)
    model = MLP(MLPConfig(input_dim=20, num_classes=2, hidden_layers=[3000], dropout=0.5, dtype="float64"))
    _, _, activations, raw, masks = model.forward(rng.normal(size=(4, 20)), training=True)
    # ReLU already zeros ~50% on its own; with a further 50% dropout keep-rate the total zero
    # fraction should be close to 1 - 0.5*0.5 = 0.75.
    assert 0.65 < float((activations[1] == 0).mean()) < 0.85
    assert masks[1] is not None and masks[0] is None  # no mask on the input layer
    # Inverted dropout scales surviving units by 1/keep, so the mean is roughly preserved.
    assert activations[1].mean() == pytest.approx(raw[1].mean(), rel=0.15)
    # Inference (training=False) must never apply dropout, regardless of the configured rate:
    # only ReLU's own ~50% zeroing remains, not dropout's additional ~50% on top of it.
    _, _, eval_acts, _, eval_masks = model.forward(rng.normal(size=(4, 20)), training=False)
    assert all(m is None for m in eval_masks)
    assert 0.4 < float((eval_acts[1] == 0).mean()) < 0.6


@pytest.mark.parametrize("name", ["relu", "leaky_relu", "sigmoid", "tanh"])
def test_activation_derivatives(name):
    act = get_activation(name)
    z = np.linspace(-3, 3, 41) + 1e-3  # avoid the ReLU kink at exactly 0
    eps = 1e-6
    numeric = (act.forward(z + eps) - act.forward(z - eps)) / (2 * eps)
    assert np.allclose(act.backward(z, act.forward(z)), numeric, atol=1e-5)


def test_softmax_rows_sum_to_one_and_is_stable():
    p = softmax(np.array([[1000.0, 1000.0, 0.0], [-5.0, 0.0, 5.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.isfinite(p).all()


def test_hidden_layer_solves_xor_but_perceptron_cannot():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, size=(1200, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
    mlp = MLP(MLPConfig(input_dim=2, num_classes=2, hidden_layers=[16], epochs=40, learning_rate=0.05))
    mlp.fit(X[:900], y[:900], X[900:], y[900:])
    linear = MLP(MLPConfig(input_dim=2, num_classes=2, hidden_layers=[], epochs=40, learning_rate=0.05))
    linear.fit(X[:900], y[:900], X[900:], y[900:])
    assert (mlp.predict(X[900:]) == y[900:]).mean() > 0.9
    assert (linear.predict(X[900:]) == y[900:]).mean() < 0.7


def test_training_loss_decreases():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(600, 10))
    y = (X[:, :4].argmax(axis=1)).astype(int)
    model = MLP(MLPConfig(input_dim=10, num_classes=4, hidden_layers=[32], epochs=15, early_stopping_patience=None))
    model.fit(X, y)
    losses = [h.train_loss for h in model.history]
    assert losses[-1] < 0.5 * losses[0]


def test_save_and_load_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(5, 8))
    model = MLP(MLPConfig(input_dim=8, num_classes=3, hidden_layers=[4]))
    model.save(tmp_path / "m.npz")
    loaded = MLP.load(tmp_path / "m.npz")
    assert np.allclose(model.predict_proba(X), loaded.predict_proba(X))


def test_metrics_match_scikit_learn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(4)
    y_true, y_pred = rng.integers(0, 4, 300), rng.integers(0, 4, 300)
    ours = classification_report(y_true, y_pred, ["a", "b", "c", "d"])
    assert ours["accuracy"] == pytest.approx(sk.accuracy_score(y_true, y_pred))
    assert ours["precision_macro"] == pytest.approx(sk.precision_score(y_true, y_pred, average="macro"))
    assert ours["recall_macro"] == pytest.approx(sk.recall_score(y_true, y_pred, average="macro"))
    assert ours["f1_macro"] == pytest.approx(sk.f1_score(y_true, y_pred, average="macro"))
    assert ours["confusion_matrix"] == sk.confusion_matrix(y_true, y_pred).tolist()
