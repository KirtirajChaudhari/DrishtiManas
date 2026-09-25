"""End-to-end experiment pipeline for the DrishtiManas OCT classifier.

    python -m ml.train            # full run (~35-45 min: tuning + ensemble training)
    python -m ml.train --quick    # smoke test on a tiny subset (~1 min)

Stages (mirrors the assignment's suggested architecture):

 1. Dataset preparation      OCTMNIST train / val / test, class distribution
 2. Pre-processing           grayscale, centre-crop, resize (64x64 by default), scale to [0, 1]
 3. Feature extraction       raw pixels vs HOG vs pixels+HOG, z-score standardised
 4. Baselines                single-layer perceptron, softmax regression, scikit-learn MLP
 5. ANN from scratch         forward propagation, activations, cross-entropy, backprop
 6. Hyperparameter tuning    greedy search over learning rate, dropout, hidden neurons, hidden layers,
                             batch size, activation, optimizer, cost function, L2, epochs
 7. Best model selected      on the validation set (macro-F1), retrained on the full training set
 8. Ensemble + calibration   N independently-seeded models averaged; a per-class log-bias tuned on a
                             class-balanced validation subsample, then evaluated ONCE on the test set
 9. Artifacts                model bundle (all ensemble members), JSON report, PNG figures
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import CLASS_CODES, CLASSES, Split, balanced_subset, class_counts, load_octmnist
from .features import FeatureExtractor
from .inference import ARTIFACTS_DIR, save_bundle
from .metrics import classification_report
from .nn import MLP, MLPConfig, gradient_check

NUM_CLASSES = len(CLASSES)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def finite(obj):
    """Replace NaN / inf (e.g. from a diverging learning rate) with None so the report is valid JSON."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite(v) for v in obj]
    return obj


def history_dicts(model: MLP) -> list[dict]:
    return [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in asdict(h).items()} for h in model.history]


def run(cfg: MLPConfig, Xtr, ytr, Xva, yva, label: str = "") -> dict:
    model = MLP(cfg)
    t0 = time.perf_counter()
    model.fit(Xtr, ytr, Xva, yva)
    seconds = time.perf_counter() - t0
    val = classification_report(yva, model.predict(Xva), CLASS_CODES)
    val_losses = np.array([h.val_loss for h in model.history], dtype=float)
    best_epoch = int(np.nanargmin(val_losses)) + 1 if np.isfinite(val_losses).any() else len(val_losses)
    log(
        f"  {label:<28} val_acc={val['accuracy']:.4f} val_f1={val['f1_macro']:.4f} "
        f"epochs={len(model.history)} ({seconds:.1f}s)"
    )
    return {
        "model": model,
        "val_acc": val["accuracy"],
        "val_f1": val["f1_macro"],
        "train_acc": model.history[best_epoch - 1].train_acc,
        "epochs_run": len(model.history),
        "best_epoch": best_epoch,
        "seconds": round(seconds, 2),
        "params": model.num_parameters,
        "history": history_dicts(model),
    }


def summarise(result: dict, value, display: str | None = None) -> dict:
    return {
        "value": value,
        "label": display if display is not None else str(value),
        **{k: result[k] for k in ("val_acc", "val_f1", "train_acc", "epochs_run", "best_epoch", "seconds", "params")},
        "history": result["history"],
    }


# --------------------------------------------------------------------------- stages
def feature_candidates(size: int) -> list[dict]:
    """Feature sets compared in stage 3. Raw pixels are capped at 32x32 (1,024 inputs) so the
    first weight matrix stays a manageable size; HOG runs on the full-resolution image.

    A multi-scale HOG variant (coarse + fine cell size) is included alongside the single-scale
    one: a coarse cell captures large-scale contrast (the overall retinal layer structure), a
    fine cell captures small-scale texture (e.g. individual drusen deposits), and a single cell
    size cannot represent both at once.
    """
    pixel_size = min(size, 32)
    coarse, fine = (4, 3) if size <= 32 else (8, 6)
    base = {"image_size": size, "pixel_size": pixel_size}
    return [
        {**base, "kind": "pixels", "hog_cells": [coarse]},
        {**base, "kind": "hog", "hog_cells": [coarse]},
        {**base, "kind": "pixels+hog", "hog_cells": [coarse]},
        {**base, "kind": "pixels+hog", "hog_cells": [coarse, fine]},
    ]


def feature_label(spec: dict) -> str:
    parts = []
    if "pixels" in spec["kind"]:
        parts.append(f"pixels {spec['pixel_size']}x{spec['pixel_size']}")
    if "hog" in spec["kind"]:
        cells = "+".join(str(c) for c in spec["hog_cells"])
        parts.append(f"HOG (cell {cells}px)")
    return " + ".join(parts)


def feature_comparison(train: Split, val: Split, base: MLPConfig, size: int) -> tuple[list[dict], dict]:
    log("Stage 3: feature extraction comparison")
    rows = []
    for spec in feature_candidates(size):
        fx = FeatureExtractor(**spec)
        Xtr, Xva = fx.fit_transform(train.images), fx.transform(val.images)
        res = run(replace(base, input_dim=fx.dim), Xtr, train.labels, Xva, val.labels, label=feature_label(spec))
        rows.append({**summarise(res, spec["kind"], feature_label(spec)), "dim": fx.dim, "spec": spec})
    best = max(rows, key=lambda r: r["val_f1"])["spec"]
    return rows, best


def baselines(Xtr, ytr, Xva, yva, Xte, yte, quick: bool) -> list[dict]:
    """Reference points before the from-scratch network is tuned."""
    from sklearn.linear_model import Perceptron
    from sklearn.neural_network import MLPClassifier

    log("Stage 4: baselines")
    out = []

    def record(name: str, description: str, pred_va, pred_te, seconds: float) -> None:
        va = classification_report(yva, pred_va, CLASS_CODES)
        te = classification_report(yte, pred_te, CLASS_CODES)
        log(f"  {name:<40} val_acc={va['accuracy']:.4f} test_acc={te['accuracy']:.4f}")
        out.append(
            {
                "name": name,
                "description": description,
                "val_acc": va["accuracy"],
                "val_f1": va["f1_macro"],
                "test_acc": te["accuracy"],
                "test_f1": te["f1_macro"],
                "seconds": round(seconds, 2),
            }
        )

    t0 = time.perf_counter()
    perceptron = Perceptron(max_iter=30, random_state=0).fit(Xtr, ytr)
    record(
        "Single-layer perceptron (Rosenblatt)",
        "Linear decision boundaries, step activation, perceptron learning rule (scikit-learn).",
        perceptron.predict(Xva),
        perceptron.predict(Xte),
        time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    softmax_reg = MLP(
        MLPConfig(
            input_dim=Xtr.shape[1],
            num_classes=NUM_CLASSES,
            hidden_layers=[],
            epochs=20,
            learning_rate=0.01,
            early_stopping_patience=5,
        )
    )
    softmax_reg.fit(Xtr, ytr, Xva, yva)
    record(
        "Softmax regression (0 hidden layers, ours)",
        "Our NumPy network with no hidden layer: a linear classifier trained by gradient descent.",
        softmax_reg.predict(Xva),
        softmax_reg.predict(Xte),
        time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    sk_mlp = MLPClassifier(
        hidden_layer_sizes=(100,), max_iter=10 if quick else 40, early_stopping=True, random_state=0
    ).fit(Xtr, ytr)
    record(
        "Baseline MLP (scikit-learn, 1x100 ReLU, Adam)",
        "Off-the-shelf multilayer perceptron with default settings, used as the baseline accuracy.",
        sk_mlp.predict(Xva),
        sk_mlp.predict(Xte),
        time.perf_counter() - t0,
    )
    return out


def stack(cfg: MLPConfig, depth: int) -> list[int]:
    """`depth` hidden layers starting at the current first-layer width, halving each time."""
    width = cfg.hidden_layers[0] if cfg.hidden_layers else 256
    return [max(width // 2**i, 8) for i in range(depth)]


MIN_GAIN = 0.005  # adopt a new value only if validation macro-F1 improves by >= 0.5 points (filters seed noise)


def ensemble_proba(models: list[MLP], X: np.ndarray) -> np.ndarray:
    """Average the softmax probabilities of several independently-seeded models."""
    return np.mean([m.predict_proba(X) for m in models], axis=0)


def balanced_calibration_indices(labels: np.ndarray, seed: int = 9) -> np.ndarray:
    """Indices for a class-balanced subsample of `labels`.

    Calibration must be tuned against the distribution it will be *used* on. OCTMNIST's
    validation split mirrors the training set's imbalance (dominated by CNV and Normal), but
    the test set -- and any general-purpose deployment, where the true class mix is unknown --
    is effectively balanced. Tuning the bias on the full imbalanced validation set optimises for
    the wrong target: it can raise validation macro-F1 by pushing the decision rule further
    towards the already-dominant classes, which then actively hurts the minority classes on a
    balanced target. Subsampling validation to be class-balanced fixes the mismatch.
    """
    rng = np.random.default_rng(seed)
    per_class = int(np.bincount(labels).min())
    idx = np.concatenate([rng.choice(np.flatnonzero(labels == c), per_class, replace=False) for c in np.unique(labels)])
    return rng.permutation(idx)


def tune_class_bias(probs: np.ndarray, y: np.ndarray, num_classes: int, iters: int = 4) -> tuple[np.ndarray, float]:
    """Coordinate-ascent search for an additive per-class log-bias b that maximises macro-F1 of
    argmax(log(p) + b) on `probs`/`y` (a form of post-hoc logit adjustment for class imbalance,
    Menon et al. 2021). It never touches the trained weights -- it only shifts each class's
    effective decision threshold, which is exactly what's needed when one class (here CNV)
    systematically absorbs predictions that belong to a harder class (here Drusen)."""
    candidates = [-1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    bias = np.zeros(num_classes)
    logp = np.log(probs + 1e-12)

    def macro_f1(b: np.ndarray) -> float:
        pred = (logp + b).argmax(axis=1)
        return classification_report(y, pred, [str(i) for i in range(num_classes)])["f1_macro"]

    best_f1 = macro_f1(bias)
    for _ in range(iters):
        improved = False
        for c in range(num_classes):
            best_c, best_c_f1 = bias[c], best_f1
            for cand in candidates:
                trial = bias.copy()
                trial[c] = cand
                f1 = macro_f1(trial)
                if f1 > best_c_f1 + 1e-6:
                    best_c_f1, best_c = f1, cand
            if best_c != bias[c]:
                bias[c], best_f1, improved = best_c, best_c_f1, True
        if not improved:
            break
    return bias, best_f1


def tune(Xtr, ytr, Xva, yva, base: MLPConfig, quick: bool) -> tuple[list[dict], MLPConfig]:
    """Greedy coordinate search: hyperparameters are tuned one at a time, each experiment
    starting from the best configuration found so far (the incumbent)."""
    log("Stage 6: hyperparameter tuning (greedy, one hyperparameter at a time)")
    opt_lr = {"sgd": None, "momentum": None, "adam": 0.001}
    grid: list[tuple[str, str, list, callable, callable]] = [
        (
            "learning_rate",
            "Learning rate",
            [0.001, 0.003, 0.005, 0.01, 0.05, 0.1],
            lambda c, v: replace(c, learning_rate=v),
            lambda c: c.learning_rate,
        ),
        (
            "dropout",
            "Dropout (hidden layers)",
            [0.0, 0.1, 0.2, 0.3, 0.4],
            lambda c, v: replace(c, dropout=v),
            lambda c: c.dropout,
        ),
        (
            "hidden_neurons",
            "Hidden neurons (1st hidden layer)",
            [16, 64, 128, 256, 512],
            lambda c, v: replace(c, hidden_layers=[v, *c.hidden_layers[1:]]),
            lambda c: c.hidden_layers[0],
        ),
        (
            "hidden_layers",
            "Number of hidden layers",
            [0, 1, 2, 3],
            lambda c, v: replace(c, hidden_layers=stack(c, v)),
            lambda c: len(c.hidden_layers),
        ),
        (
            "batch_size",
            "Batch size",
            [16, 32, 64, 128, 256],
            lambda c, v: replace(c, batch_size=v),
            lambda c: c.batch_size,
        ),
        (
            "activation",
            "Activation function",
            ["relu", "leaky_relu", "tanh", "sigmoid"],
            lambda c, v: replace(c, activation=v),
            lambda c: c.activation,
        ),
        (
            "optimizer",
            "Optimizer",
            ["sgd", "momentum", "adam"],
            # plain SGD without momentum needs a ~5x larger step for a comparable effective learning rate
            lambda c, v: replace(
                c, optimizer=v, learning_rate=opt_lr[v] or (c.learning_rate * (5 if v == "sgd" else 1))
            ),
            lambda c: c.optimizer,
        ),
        ("loss", "Cost function", ["cross_entropy", "mse"], lambda c, v: replace(c, loss=v), lambda c: c.loss),
        ("l2", "L2 regularisation (lambda)", [0.0, 1e-4, 1e-3, 1e-2], lambda c, v: replace(c, l2=v), lambda c: c.l2),
    ]
    if quick:
        grid = [(k, t, vals[:2], fn, get) for k, t, vals, fn, get in grid]

    experiments = []
    incumbent = base
    incumbent_f1 = None
    for key, title, values, apply, get in grid:
        start_value = get(incumbent)
        if start_value not in values:
            values = [start_value, *values]
        log(f" {title} (incumbent: {start_value})")
        runs = []
        for v in values:
            cfg = incumbent if v == start_value else apply(incumbent, v)
            res = run(cfg, Xtr, ytr, Xva, yva, label=f"{key}={v}")
            label = f"{v} ({'-'.join(map(str, stack(incumbent, v))) or 'none'})" if key == "hidden_layers" else str(v)
            runs.append({**summarise(res, v, label), "config": asdict(cfg)})
        current = next((r for r in runs if r["value"] == start_value), None)
        if current is not None:
            incumbent_f1 = current["val_f1"]
        best = max(runs, key=lambda r: r["val_f1"])
        adopted = best["value"] if current is None or best["val_f1"] >= current["val_f1"] + MIN_GAIN else start_value
        if adopted != start_value:
            incumbent = MLPConfig(**next(r["config"] for r in runs if r["value"] == adopted))
            incumbent_f1 = best["val_f1"]
        log(f"  -> keep {key}={adopted}")
        for r in runs:
            r.pop("config")
        experiments.append({"key": key, "title": title, "incumbent": start_value, "best": adopted, "runs": runs})

    # Epochs: one long run without early stopping shows under- and over-fitting.
    log(" Number of epochs (single long run, no early stopping)")
    long_epochs = 8 if quick else 60
    res = run(replace(incumbent, epochs=long_epochs, early_stopping_patience=None), Xtr, ytr, Xva, yva, label="epochs")
    checkpoints = [e for e in (5, 10, 20, 30, 40, 60) if e <= long_epochs] or [long_epochs]
    epoch_runs = []
    for e in checkpoints:
        h = res["history"][e - 1]
        epoch_runs.append(
            {
                "value": e,
                "label": str(e),
                "val_acc": h["val_acc"],
                "val_loss": h["val_loss"],
                "train_acc": h["train_acc"],
                "train_loss": h["train_loss"],
            }
        )
    experiments.append(
        {
            "key": "epochs",
            "title": "Number of epochs",
            "best": res["best_epoch"],
            "runs": epoch_runs,
            "history": res["history"],
        }
    )

    base_res = run(base, Xtr, ytr, Xva, yva, label="base (re-check)")
    selection = [
        {"name": "tuned (greedy search)", "val_f1": incumbent_f1 or 0.0, "val_acc": None, "config": asdict(incumbent)},
        {
            "name": "base configuration",
            "val_f1": base_res["val_f1"],
            "val_acc": base_res["val_acc"],
            "config": asdict(base),
        },
    ]
    for r in experiments:
        for run_ in r["runs"]:
            if r["key"] != "epochs" and run_["val_f1"] == incumbent_f1:
                selection[0]["val_acc"] = run_["val_acc"]
    selection.sort(key=lambda r: r["val_f1"], reverse=True)
    experiments.append({"key": "selection", "title": "Model selection", "runs": selection})
    chosen = incumbent if selection[0]["name"].startswith("tuned") else base
    return experiments, chosen


def export_samples(test: Split, out_dir: Path, per_class: int = 3) -> list[dict]:
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    samples = []
    for c in CLASSES:
        idx = rng.choice(np.flatnonzero(test.labels == c["id"]), size=per_class, replace=False)
        for j, i in enumerate(idx):
            name = f"{c['code'].lower()}_{j + 1}.png"
            Image.fromarray(test.images[i]).save(out_dir / name)
            samples.append({"file": name, "label": c["code"], "name": c["name"], "test_index": int(i)})
    return samples


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="tiny smoke-test run")
    parser.add_argument("--data", default=None, help="path to octmnist.npz / octmnist_64.npz")
    parser.add_argument("--size", type=int, default=64, choices=[28, 64], help="OCTMNIST image resolution")
    parser.add_argument("--tune-per-class", type=int, default=4000, help="balanced images per class for tuning")
    parser.add_argument(
        "--ensemble-size", type=int, default=3, help="number of differently-seeded models averaged at inference"
    )
    parser.add_argument(
        "--out", default=None, help="output folder (default: artifacts/, or artifacts-quick/ with --quick)"
    )
    args = parser.parse_args()

    # Smoke tests never overwrite the committed model unless --out artifacts is given explicitly.
    out = Path(args.out) if args.out else (ARTIFACTS_DIR.parent / "artifacts-quick" if args.quick else ARTIFACTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    log("Stage 1: dataset preparation (OCTMNIST)")
    ds = load_octmnist(args.data, size=args.size)
    counts = {s: class_counts(getattr(ds, s).labels) for s in ("train", "val", "test")}
    log(f"  class counts {counts}")
    per_class = 150 if args.quick else args.tune_per_class
    tune_train = balanced_subset(ds.train, per_class, seed=0)
    val = ds.val if not args.quick else balanced_subset(ds.val, 100, seed=1)

    base = MLPConfig(
        input_dim=0,
        num_classes=NUM_CLASSES,
        hidden_layers=[256],
        activation="relu",
        optimizer="momentum",
        learning_rate=0.01,
        batch_size=64,
        epochs=6 if args.quick else 30,
        l2=1e-4,
        early_stopping_patience=5,
    )

    feat_rows, feature_spec = feature_comparison(tune_train, val, base, args.size)
    fx = FeatureExtractor(**feature_spec).fit(tune_train.images)
    Xtr, Xva, Xte = fx.transform(tune_train.images), fx.transform(val.images), fx.transform(ds.test.images)
    base = replace(base, input_dim=fx.dim)

    baseline_rows = baselines(Xtr, tune_train.labels, Xva, val.labels, Xte, ds.test.labels, args.quick)

    log("Stage 5: gradient check of our backpropagation")
    small = MLP(replace(base, hidden_layers=[32, 16], dtype="float64"))
    for b in small.biases:  # keep ReLU inputs off the kink at exactly 0
        b[:] = np.random.default_rng(0).normal(0, 0.1, size=b.shape)
    grad_err = gradient_check(small, Xtr[:64].astype(np.float64), tune_train.labels[:64], num_checks=80)
    log(f"  max relative error (analytic vs numerical) = {grad_err:.2e}")

    experiments, best_cfg = tune(Xtr, tune_train.labels, Xva, val.labels, base, args.quick)
    log(f"Selected configuration: {best_cfg}")

    log("Stage 7: final training on the training set with the selected hyperparameters")
    final_candidates = {
        "balanced": balanced_subset(ds.train, 300 if args.quick else 7754, seed=3),
        "full+class_weights": ds.train if not args.quick else balanced_subset(ds.train, 600, seed=4),
    }
    # The tuning subset is much smaller than the final training set; with ~4-12x more updates per
    # epoch the tuned learning rate can be too aggressive, so a 3x smaller rate is also tried.
    final_lrs = [best_cfg.learning_rate, round(best_cfg.learning_rate / 3, 6)]
    finals = []
    for regime, split in final_candidates.items():
        fx_final = FeatureExtractor(**feature_spec).fit(split.images)
        X_final, X_val_final = fx_final.transform(split.images), fx_final.transform(val.images)
        weights = None
        if regime.startswith("full"):
            freq = np.bincount(split.labels, minlength=NUM_CLASSES) / split.labels.size
            weights = (1.0 / (NUM_CLASSES * freq)).round(4).tolist()
        for lr in final_lrs:
            cfg = replace(
                best_cfg,
                learning_rate=lr,
                epochs=8 if args.quick else 60,
                early_stopping_patience=4 if args.quick else 8,
                lr_decay=0.97,
                class_weights=weights,
                input_dim=fx_final.dim,
            )
            res = run(cfg, X_final, split.labels, X_val_final, val.labels, label=f"final[{regime}, lr={lr}]")
            finals.append((res["val_f1"], f"{regime}, lr={lr}", res, fx_final, split))
        del X_final
    finals.sort(key=lambda f: f[0], reverse=True)
    _, regime, final, fx_final, final_split = finals[0]
    log(f"  chose training regime '{regime}'")

    # ---- Stage 8: ensemble of independently-seeded models + class-prior calibration ----
    # Averaging several models trained with different random seeds (same architecture and
    # data, different weight initialisation and mini-batch shuffling) cancels out some of
    # each individual model's variance -- a cheap, principled accuracy/F1 improvement that
    # costs almost nothing at inference (each model takes ~1ms).
    ensemble_n = max(1, 2 if args.quick else args.ensemble_size)
    log(f"Stage 8: ensemble of {ensemble_n} model(s) + class-prior calibration")
    X_final = fx_final.transform(final_split.images)
    X_val_final = fx_final.transform(val.images)

    members: list[MLP] = [final["model"]]  # the winning (regime, lr) run above is member 0
    member_stats = [{"seed": final["model"].config.seed, "val_acc": final["val_acc"], "val_f1": final["val_f1"]}]
    for seed in range(43, 43 + ensemble_n - 1):
        cfg = replace(final["model"].config, seed=seed)
        res = run(cfg, X_final, final_split.labels, X_val_final, val.labels, label=f"ensemble seed={seed}")
        members.append(res["model"])
        member_stats.append({"seed": seed, "val_acc": res["val_acc"], "val_f1": res["val_f1"]})

    val_probs = ensemble_proba(members, X_val_final)
    ensemble_val = classification_report(val.labels, val_probs.argmax(axis=1), CLASS_CODES)
    log(f"  ensemble (uncalibrated) val_acc={ensemble_val['accuracy']:.4f} val_f1={ensemble_val['f1_macro']:.4f}")

    # Tune the bias on a class-balanced subsample of validation (matches the balanced test
    # protocol), not the full imbalanced split -- see `balanced_calibration_indices`.
    cal_idx = balanced_calibration_indices(val.labels)
    class_bias, calibrated_cal_f1 = tune_class_bias(val_probs[cal_idx], val.labels[cal_idx], NUM_CLASSES)
    uncalibrated_cal_f1 = classification_report(val.labels[cal_idx], val_probs[cal_idx].argmax(axis=1), CLASS_CODES)[
        "f1_macro"
    ]
    log(
        f"  class bias {class_bias.round(3).tolist()}  balanced-calibration-set macro-F1 "
        f"{uncalibrated_cal_f1:.4f} -> {calibrated_cal_f1:.4f}"
    )

    def calibrated_predict(probs: np.ndarray) -> np.ndarray:
        return (np.log(probs + 1e-12) + class_bias).argmax(axis=1)

    val_report = classification_report(val.labels, calibrated_predict(val_probs), CLASS_CODES)
    X_test_final = fx_final.transform(ds.test.images)
    test_probs = ensemble_proba(members, X_test_final)
    test_report = classification_report(ds.test.labels, calibrated_predict(test_probs), CLASS_CODES)
    log(f"  TEST accuracy={test_report['accuracy']:.4f} macro-F1={test_report['f1_macro']:.4f}")

    # Ablation: what each post-tuning step contributes, on validation AND test. This is
    # reporting only -- every decision above was made on validation before the test set was
    # touched -- but it shows honestly whether a step that helped validation also helped test.
    single_val = members[0].predict_proba(X_val_final).argmax(axis=1)
    single_test = members[0].predict_proba(X_test_final).argmax(axis=1)
    ablation_steps = [
        ("Single model", single_val, single_test),
        (f"{len(members)}-model ensemble", val_probs.argmax(axis=1), test_probs.argmax(axis=1)),
        ("Ensemble + calibration (served)", calibrated_predict(val_probs), calibrated_predict(test_probs)),
    ]
    ablation_rows = []
    for name, pv, pt in ablation_steps:
        rv = classification_report(val.labels, pv, CLASS_CODES)
        rt = classification_report(ds.test.labels, pt, CLASS_CODES)
        ablation_rows.append(
            {
                "step": name,
                "val_acc": rv["accuracy"],
                "val_f1": rv["f1_macro"],
                "test_acc": rt["accuracy"],
                "test_f1": rt["f1_macro"],
                "test_per_class_f1": {c["class"]: c["f1"] for c in rt["per_class"]},
            }
        )
        log(
            f"  ablation {name:<32} val_f1={rv['f1_macro']:.4f} test_acc={rt['accuracy']:.4f} test_f1={rt['f1_macro']:.4f}"
        )

    samples = export_samples(ds.test, out / "samples")
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "OCTMNIST",
        "test_accuracy": test_report["accuracy"],
        "test_f1_macro": test_report["f1_macro"],
    }
    save_bundle(out / "model.npz", members, fx_final, meta, class_bias=class_bias)

    report = {
        "generated_at": meta["trained_at"],
        "quick": args.quick,
        "dataset": {
            "name": "OCTMNIST (MedMNIST v2)",
            "source": "Kermany et al., Cell 2018; Yang et al., Scientific Data 2023",
            "url": "https://medmnist.com/",
            "license": "CC BY 4.0",
            "image_shape": [args.size, args.size],
            "classes": CLASSES,
            "counts": counts,
            "tuning_subset_per_class": per_class,
            "final_training_regime": regime.split(",")[0],
            "final_training_size": int(final_split.labels.size),
        },
        "preprocessing": [
            "Decode image, apply EXIF orientation, drop alpha channel",
            "Convert to 8-bit grayscale",
            "Centre-crop to a square on the short edge",
            f"Resize to {args.size}x{args.size} pixels (bicubic)",
            "Scale intensities to [0, 1]",
            "Extract features and z-score standardise with training-set mean and std",
        ],
        "features": {
            "selected": feature_label(feature_spec),
            "spec": feature_spec,
            "dim": fx_final.dim,
            "comparison": feat_rows,
        },
        "baselines": baseline_rows,
        "gradient_check": {"max_relative_error": grad_err, "passed": bool(grad_err < 1e-4)},
        "tuning": {"base_config": asdict(base), "experiments": experiments},
        "final": {
            "config": asdict(members[0].config),
            "parameters": members[0].num_parameters,
            "layers": [fx_final.dim, *members[0].config.hidden_layers, NUM_CLASSES],
            "epochs_trained": final["epochs_run"],
            "best_epoch": final["best_epoch"],
            "training_seconds": final["seconds"],
            "history": final["history"],
            "regimes": [
                {"regime": r, "val_f1": f, "val_acc": res_["val_acc"], "epochs": res_["epochs_run"]}
                for f, r, res_, *_ in finals
            ],
            "ensemble": {
                "size": len(members),
                "members": member_stats,
                "total_parameters": members[0].num_parameters * len(members),
                "uncalibrated_validation": ensemble_val,
            },
            "ablation": {"rows": ablation_rows},
            "calibration": {
                "method": (
                    "per-class additive log-bias (logit adjustment), coordinate search on a "
                    "class-balanced subsample of validation (matches the balanced test protocol)"
                ),
                "bias": class_bias.round(4).tolist(),
                "calibration_set_size": int(len(cal_idx)),
                "val_f1_before": uncalibrated_cal_f1,
                "val_f1_after": calibrated_cal_f1,
                "full_validation_f1_before": ensemble_val["f1_macro"],
                "full_validation_f1_after": val_report["f1_macro"],
            },
            "validation": val_report,
            "test": test_report,
        },
        "samples": samples,
        "pipeline_seconds": round(time.perf_counter() - started, 1),
    }
    (out / "report.json").write_text(json.dumps(finite(report), indent=1, allow_nan=False))
    log(f"Wrote {out / 'report.json'} and {out / 'model.npz'}")

    try:
        from .figures import make_figures

        make_figures(report, out / "figures", ds)
        log(f"Figures written to {out / 'figures'}")
    except ImportError as exc:  # matplotlib is optional
        log(f"Skipping figures ({exc})")


if __name__ == "__main__":
    main()
