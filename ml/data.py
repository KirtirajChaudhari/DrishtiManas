"""OCTMNIST dataset loading.

OCTMNIST (MedMNIST v2, Yang et al., Scientific Data 2023) contains 109,309
retinal optical coherence tomography (OCT) B-scans from Kermany et al., Cell
2018, pre-processed to 28x28 grayscale and split into train / val / test.

Classes
    0  CNV     choroidal neovascularisation
    1  DME     diabetic macular edema
    2  Drusen  drusen (early age-related macular degeneration)
    3  Normal  healthy retina

License: CC BY 4.0.
"""
from __future__ import annotations

import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.environ.get("DRISHTI_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))

SOURCES = [
    "https://zenodo.org/records/10519652/files/octmnist.npz?download=1",
    "https://huggingface.co/datasets/albertvillanova/medmnist-v2/resolve/main/octmnist.npz",
]

CLASSES = [
    {
        "id": 0,
        "code": "CNV",
        "name": "Choroidal Neovascularization",
        "description": "Abnormal blood vessels growing beneath the retina (wet AMD). Needs urgent referral.",
        "urgency": "urgent",
    },
    {
        "id": 1,
        "code": "DME",
        "name": "Diabetic Macular Edema",
        "description": "Fluid accumulation in the macula caused by diabetic retinopathy.",
        "urgency": "urgent",
    },
    {
        "id": 2,
        "code": "DRUSEN",
        "name": "Drusen",
        "description": "Yellow deposits under the retina; an early sign of age-related macular degeneration.",
        "urgency": "routine",
    },
    {
        "id": 3,
        "code": "NORMAL",
        "name": "Normal",
        "description": "Healthy retina with preserved foveal contour and no fluid or deposits.",
        "urgency": "none",
    },
]
CLASS_CODES = [c["code"] for c in CLASSES]


@dataclass
class Split:
    images: np.ndarray  # (N, 28, 28) uint8
    labels: np.ndarray  # (N,) int64


@dataclass
class OCTMNIST:
    train: Split
    val: Split
    test: Split


def download(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in SOURCES:
        try:
            print(f"Downloading {url}")
            tmp = dest.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
                shutil.copyfileobj(resp, fh)
            tmp.rename(dest)
            return dest
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            errors.append(f"{url}: {exc}")
    raise RuntimeError(
        "Could not download OCTMNIST. Download octmnist.npz manually from "
        "https://zenodo.org/records/10519652 and place it at " + str(dest) + "\n" + "\n".join(errors)
    )


def load_octmnist(path: str | Path | None = None) -> OCTMNIST:
    path = Path(path) if path else DATA_DIR / "octmnist.npz"
    if not path.exists():
        download(path)
    data = np.load(path)

    def split(name: str) -> Split:
        return Split(
            images=data[f"{name}_images"].astype(np.uint8),
            labels=data[f"{name}_labels"].reshape(-1).astype(np.int64),
        )

    return OCTMNIST(train=split("train"), val=split("val"), test=split("test"))


def balanced_subset(split: Split, per_class: int, seed: int = 0) -> Split:
    """Sample up to `per_class` images of every class (undersamples the majority classes)."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(split.labels):
        members = np.flatnonzero(split.labels == c)
        idx.append(rng.choice(members, size=min(per_class, members.size), replace=False))
    order = rng.permutation(np.concatenate(idx))
    return Split(images=split.images[order], labels=split.labels[order])


def class_counts(labels: np.ndarray, num_classes: int = len(CLASSES)) -> list[int]:
    return np.bincount(labels, minlength=num_classes).tolist()
