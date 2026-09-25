"""Static PNG figures for the written report (the web app draws its own charts)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TRAIN = "#2a78d6"
VAL = "#eb6834"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def curves(history: list[dict], out: Path, title_suffix: str = "") -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, key, label in ((axes[0], "loss", "Cross-entropy loss"), (axes[1], "acc", "Accuracy")):
        ax.plot(epochs, [h[f"train_{key}"] for h in history], color=TRAIN, label="Training")
        ax.plot(epochs, [h[f"val_{key}"] for h in history], color=VAL, label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs epochs{title_suffix}")
        ax.legend()
    _save(fig, out)


def confusion(cm: list[list[int]], labels: list[str], out: Path, title: str) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.grid(False)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, out)


def tuning(experiments: list[dict], out_dir: Path) -> None:
    for exp in experiments:
        if exp["key"] in ("selection",):
            continue
        runs = exp["runs"]
        if exp["key"] == "epochs":
            curves(exp["history"], out_dir / "tuning_epochs.png", " (no early stopping)")
            continue
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        labels = [r["label"] for r in runs]
        x = np.arange(len(runs))
        axes[0].bar(x - 0.2, [r["train_acc"] for r in runs], 0.38, color=TRAIN, label="Training")
        axes[0].bar(x + 0.2, [r["val_acc"] for r in runs], 0.38, color=VAL, label="Validation")
        axes[0].set_xticks(x, labels, rotation=20)
        lo = min(min(r["val_acc"] for r in runs), min(r["train_acc"] for r in runs))
        axes[0].set_ylim(max(0, lo - 0.1), 1)
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title(f"{exp['title']}: accuracy")
        axes[0].legend()
        for i, r in enumerate(runs):
            axes[1].plot(
                [h["epoch"] for h in r["history"]],
                [h["val_loss"] for h in r["history"]],
                color=SERIES[i % len(SERIES)],
                label=r["label"],
            )
        # Cap the axis so a diverging run (e.g. learning rate 0.1) does not flatten the others.
        first = sorted(r["history"][0]["val_loss"] for r in runs if r["history"] and r["history"][0]["val_loss"])
        if first:
            cap = first[len(first) // 2] * 2
            if any((h["val_loss"] or np.inf) > cap for r in runs for h in r["history"]):
                axes[1].set_ylim(0, cap)
                axes[1].text(
                    0.98,
                    0.02,
                    "axis capped; diverging runs clipped",
                    transform=axes[1].transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=7,
                    color=MUTED,
                )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation loss")
        axes[1].set_title(f"{exp['title']}: validation loss")
        axes[1].legend(fontsize=7)
        _save(fig, out_dir / f"tuning_{exp['key']}.png")


def activations(out: Path) -> None:
    z = np.linspace(-5, 5, 400)
    fns = {
        "ReLU": (np.maximum(0, z), (z > 0).astype(float)),
        "Leaky ReLU (0.01)": (np.where(z > 0, z, 0.01 * z), np.where(z > 0, 1, 0.01)),
        "Sigmoid": (1 / (1 + np.exp(-z)), (1 / (1 + np.exp(-z))) * (1 - 1 / (1 + np.exp(-z)))),
        "Tanh": (np.tanh(z), 1 - np.tanh(z) ** 2),
    }
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.6))
    for ax, (name, (f, d)) in zip(axes, fns.items()):
        ax.plot(z, f, color=TRAIN, label="g(z)")
        ax.plot(z, d, color=VAL, linestyle="--", label="g'(z)")
        ax.set_title(name)
        ax.set_xlabel("z")
        ax.legend(fontsize=7)
    _save(fig, out)


def samples_grid(ds, out: Path, classes: list[dict], per_class: int = 6) -> None:
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(len(classes), per_class, figsize=(per_class * 1.1, len(classes) * 1.2))
    for r, c in enumerate(classes):
        idx = rng.choice(np.flatnonzero(ds.train.labels == c["id"]), per_class, replace=False)
        for k, i in enumerate(idx):
            ax = axes[r, k]
            ax.imshow(ds.train.images[i], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if k == 0:
                ax.set_ylabel(c["code"], color=INK, fontsize=9)
    _save(fig, out)


def class_distribution(counts: dict, labels: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3))
    x = np.arange(len(labels))
    for i, split in enumerate(("train", "val", "test")):
        ax.bar(x + (i - 1) * 0.27, counts[split], 0.25, color=SERIES[i], label=split)
    ax.set_xticks(x, labels)
    ax.set_yscale("log")
    ax.set_ylabel("Images (log scale)")
    ax.set_title("OCTMNIST class distribution")
    ax.legend()
    _save(fig, out)


def make_figures(report: dict, out_dir: Path, ds) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [c["code"] for c in report["dataset"]["classes"]]
    curves(report["final"]["history"], out_dir / "final_training_curves.png")
    confusion(
        report["final"]["test"]["confusion_matrix"],
        labels,
        out_dir / "confusion_matrix_test.png",
        "Confusion matrix (test set)",
    )
    confusion(
        report["final"]["validation"]["confusion_matrix"],
        labels,
        out_dir / "confusion_matrix_val.png",
        "Confusion matrix (validation set)",
    )
    tuning(report["tuning"]["experiments"], out_dir)
    activations(out_dir / "activation_functions.png")
    samples_grid(ds, out_dir / "dataset_samples.png", report["dataset"]["classes"])
    class_distribution(report["dataset"]["counts"], labels, out_dir / "class_distribution.png")


if __name__ == "__main__":
    # Re-render the figures from an existing report without retraining:  python -m ml.figures
    import json

    from .data import load_octmnist
    from .inference import ARTIFACTS_DIR

    rep = json.loads((ARTIFACTS_DIR / "report.json").read_text())
    make_figures(rep, ARTIFACTS_DIR / "figures", load_octmnist(size=rep["dataset"]["image_shape"][0]))
    print("figures written to", ARTIFACTS_DIR / "figures")
