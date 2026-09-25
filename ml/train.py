"""End-to-end experiment pipeline for the DrishtiManas OCT classifier.

    python -m ml.train            # full run (~20-30 min on a laptop CPU)
    python -m ml.train --quick    # smoke test on a tiny subset (~1 min)

Stages (mirrors the assignment's suggested architecture):

 1. Dataset preparation      OCTMNIST train / val / test, class distribution
 2. Pre-processing           grayscale, centre-crop, 28x28, scale to [0, 1]
 3. Feature extraction       raw pixels vs HOG vs pixels+HOG, z-score standardised
 4. Baselines                single-layer perceptron, softmax regression, scikit-learn MLP
 5. ANN from scratch         forward propagation, activations, cross-entropy, backprop
 6. Hyperparameter tuning    learning rate, hidden neurons, hidden layers, batch size,
                             activation, optimizer, cost function, L2, epochs
 7. Best model selected      on the validation set (macro-F1), retrained on the full
                             training set and evaluated ONCE on the held-out test set
 8. Artifacts                model bundle, JSON report for the web app, PNG figures
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


def history_dicts(model: MLP) -> list[dict]:
    return [
        {k: (round(v, 5) if isinstance(v, float) else v) for k, v in asdict(h).items()}
        for h in model.history
    ]


def run(cfg: MLPConfig, Xtr, ytr, Xva, yva, label: str = "") -> dict:
    model = MLP(cfg)
    t0 = time.perf_counter()
    model.fit(Xtr, ytr, Xva, yva)
    seconds = time.perf_counter() - t0
    val = classification_report(yva, model.predict(Xva), CLASS_CODES)
    best_epoch = int(np.argmin([h.val_loss for h in model.history])) + 1
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
def feature_comparison(train: Split, val: Split, base: MLPConfig) -> tuple[list[dict], str]:
    log("Stage 3: feature extraction comparison")
    rows = []
    for kind in ("pixels", "hog", "pixels+hog"):
        fx = FeatureExtractor(kind=kind)
        Xtr, Xva = fx.fit_transform(train.images), fx.transform(val.images)
        res = run(replace(base, input_dim=fx.dim), Xtr, train.labels, Xva, val.labels, label=kind)
        rows.append({**summarise(res, kind), "dim": fx.dim})
    best = max(rows, key=lambda r: r["val_f1"])["value"]
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
    softmax_reg = MLP(MLPConfig(input_dim=Xtr.shape[1], num_classes=NUM_CLASSES, hidden_layers=[], epochs=20,
                                learning_rate=0.01, early_stopping_patience=5))
    softmax_reg.fit(Xtr, ytr, Xva, yva)
    record(
        "Softmax regression (0 hidden layers, ours)",
        "Our NumPy network with no hidden layer: a linear classifier trained by gradient descent.",
        softmax_reg.predict(Xva),
        softmax_reg.predict(Xte),
        time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    sk_mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10 if quick else 40, early_stopping=True,
                           random_state=0).fit(Xtr, ytr)
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


def tune(Xtr, ytr, Xva, yva, base: MLPConfig, quick: bool) -> tuple[list[dict], MLPConfig]:
    log("Stage 6: hyperparameter tuning (one factor at a time around the base configuration)")
    grid: list[tuple[str, str, list, callable]] = [
        ("learning_rate", "Learning rate", [0.001, 0.005, 0.01, 0.05, 0.1],
         lambda c, v: replace(c, learning_rate=v)),
        ("hidden_neurons", "Hidden neurons (1 hidden layer)", [16, 64, 128, 256, 512],
         lambda c, v: replace(c, hidden_layers=[v])),
        ("hidden_layers", "Number of hidden layers", [0, 1, 2, 3],
         lambda c, v: replace(c, hidden_layers=stack(c, v))),
        ("batch_size", "Batch size", [16, 32, 64, 128, 256],
         lambda c, v: replace(c, batch_size=v)),
        ("activation", "Activation function", ["relu", "leaky_relu", "tanh", "sigmoid"],
         lambda c, v: replace(c, activation=v)),
        ("optimizer", "Optimizer", ["sgd", "momentum", "adam"],
         lambda c, v: replace(c, optimizer=v, learning_rate={"sgd": 0.05, "momentum": 0.01, "adam": 0.001}[v])),
        ("loss", "Cost function", ["cross_entropy", "mse"],
         lambda c, v: replace(c, loss=v)),
        ("l2", "L2 regularisation (lambda)", [0.0, 1e-4, 1e-3, 1e-2],
         lambda c, v: replace(c, l2=v)),
    ]
    if quick:
        grid = [(k, t, vals[:2], fn) for k, t, vals, fn in grid]

    experiments = []
    best_values = {}
    for key, title, values, apply in grid:
        log(f" {title}")
        runs = []
        for v in values:
            res = run(apply(base, v), Xtr, ytr, Xva, yva, label=f"{key}={v}")
            label = f"{v} ({'-'.join(map(str, stack(base, v))) or 'none'})" if key == "hidden_layers" else str(v)
            runs.append(summarise(res, v, label))
        best = max(runs, key=lambda r: r["val_f1"])
        best_values[key] = best["value"]
        experiments.append({"key": key, "title": title, "best": best["value"], "runs": runs})

    # Epochs: one long run without early stopping shows under- and over-fitting.
    log(" Number of epochs (single long run, no early stopping)")
    long_epochs = 8 if quick else 60
    res = run(replace(base, epochs=long_epochs, early_stopping_patience=None), Xtr, ytr, Xva, yva, label="epochs")
    checkpoints = [e for e in (5, 10, 20, 30, 40, 60) if e <= long_epochs] or [long_epochs]
    epoch_runs = []
    for e in checkpoints:
        h = res["history"][e - 1]
        epoch_runs.append({"value": e, "label": str(e), "val_acc": h["val_acc"], "val_loss": h["val_loss"],
                           "train_acc": h["train_acc"], "train_loss": h["train_loss"]})
    best_epoch = res["best_epoch"]
    experiments.append({"key": "epochs", "title": "Number of epochs", "best": best_epoch, "runs": epoch_runs,
                        "history": res["history"]})

    # Combine the per-factor winners and confirm on the validation set.
    log(" Combining the best value of every hyperparameter")
    combined = base
    for key, _, _, apply in grid:
        if key == "optimizer":
            continue  # the optimizer changes the learning-rate scale; keep the tuned learning rate
        combined = apply(combined, best_values[key])
    combined = replace(combined, optimizer=base.optimizer)
    if best_values.get("optimizer") and best_values["optimizer"] != base.optimizer:
        alt = replace(combined, optimizer=best_values["optimizer"],
                      learning_rate={"sgd": 0.05, "momentum": 0.01, "adam": 0.001}[best_values["optimizer"]])
    else:
        alt = None
    candidates = [("base", base), ("combined", combined)] + ([("combined+optimizer", alt)] if alt else [])
    scored = []
    for name, cfg in candidates:
        r = run(cfg, Xtr, ytr, Xva, yva, label=name)
        scored.append((r["val_f1"], name, cfg, r))
    scored.sort(key=lambda s: s[0], reverse=True)
    selection = [
        {"name": name, "val_f1": r["val_f1"], "val_acc": r["val_acc"], "config": asdict(cfg)}
        for _, name, cfg, r in scored
    ]
    experiments.append({"key": "selection", "title": "Model selection", "runs": selection})
    return experiments, scored[0][2]


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
    parser.add_argument("--data", default=None, help="path to octmnist.npz")
    parser.add_argument("--tune-per-class", type=int, default=2000, help="balanced images per class for tuning")
    parser.add_argument("--out", default=str(ARTIFACTS_DIR))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    log("Stage 1: dataset preparation (OCTMNIST)")
    ds = load_octmnist(args.data)
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

    feat_rows, feature_kind = feature_comparison(tune_train, val, base)
    fx = FeatureExtractor(kind=feature_kind).fit(tune_train.images)
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
    finals = []
    for regime, split in final_candidates.items():
        fx_final = FeatureExtractor(kind=feature_kind).fit(split.images)
        X_final, X_val_final = fx_final.transform(split.images), fx_final.transform(val.images)
        weights = None
        if regime.startswith("full"):
            freq = np.bincount(split.labels, minlength=NUM_CLASSES) / split.labels.size
            weights = (1.0 / (NUM_CLASSES * freq)).round(4).tolist()
        cfg = replace(best_cfg, epochs=8 if args.quick else 60, early_stopping_patience=4 if args.quick else 8,
                      lr_decay=0.97, class_weights=weights, input_dim=fx_final.dim)
        res = run(cfg, X_final, split.labels, X_val_final, val.labels, label=f"final[{regime}]")
        finals.append((res["val_f1"], regime, res, fx_final, split))
    finals.sort(key=lambda f: f[0], reverse=True)
    _, regime, final, fx_final, final_split = finals[0]
    model: MLP = final["model"]
    log(f"  chose training regime '{regime}'")

    val_report = classification_report(val.labels, model.predict(fx_final.transform(val.images)), CLASS_CODES)
    test_pred = model.predict(fx_final.transform(ds.test.images))
    test_report = classification_report(ds.test.labels, test_pred, CLASS_CODES)
    log(f"  TEST accuracy={test_report['accuracy']:.4f} macro-F1={test_report['f1_macro']:.4f}")

    samples = export_samples(ds.test, out / "samples")
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "OCTMNIST",
        "test_accuracy": test_report["accuracy"],
        "test_f1_macro": test_report["f1_macro"],
    }
    save_bundle(out / "model.npz", model, fx_final, meta)

    report = {
        "generated_at": meta["trained_at"],
        "quick": args.quick,
        "dataset": {
            "name": "OCTMNIST (MedMNIST v2)",
            "source": "Kermany et al., Cell 2018; Yang et al., Scientific Data 2023",
            "url": "https://medmnist.com/",
            "license": "CC BY 4.0",
            "image_shape": [28, 28],
            "classes": CLASSES,
            "counts": counts,
            "tuning_subset_per_class": per_class,
            "final_training_regime": regime,
            "final_training_size": int(final_split.labels.size),
        },
        "preprocessing": [
            "Decode image, apply EXIF orientation, drop alpha channel",
            "Convert to 8-bit grayscale",
            "Centre-crop to a square on the short edge",
            "Resize to 28x28 pixels (bicubic)",
            "Scale intensities to [0, 1]",
            "Extract features and z-score standardise with training-set mean and std",
        ],
        "features": {"selected": feature_kind, "dim": fx_final.dim, "comparison": feat_rows},
        "baselines": baseline_rows,
        "gradient_check": {"max_relative_error": grad_err, "passed": bool(grad_err < 1e-4)},
        "tuning": {"base_config": asdict(base), "experiments": experiments},
        "final": {
            "config": asdict(model.config),
            "parameters": model.num_parameters,
            "layers": [fx_final.dim, *model.config.hidden_layers, NUM_CLASSES],
            "epochs_trained": final["epochs_run"],
            "best_epoch": final["best_epoch"],
            "training_seconds": final["seconds"],
            "history": final["history"],
            "regimes": [{"regime": r, "val_f1": f} for f, r, *_ in finals],
            "validation": val_report,
            "test": test_report,
        },
        "samples": samples,
        "pipeline_seconds": round(time.perf_counter() - started, 1),
    }
    (out / "report.json").write_text(json.dumps(report, indent=1))
    log(f"Wrote {out / 'report.json'} and {out / 'model.npz'}")

    try:
        from .figures import make_figures

        make_figures(report, out / "figures", ds)
        log(f"Figures written to {out / 'figures'}")
    except ImportError as exc:  # matplotlib is optional
        log(f"Skipping figures ({exc})")


if __name__ == "__main__":
    main()
