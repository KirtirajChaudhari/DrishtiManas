"""Image preprocessing and feature extraction (NumPy + Pillow only).

The same functions run at training time (on OCTMNIST arrays) and at inference
time (on images uploaded through the web app), so the network always sees
identically prepared inputs.

Pipeline for one image:
    any image -> grayscale -> centre square crop -> resize to 28x28 -> [0, 1]
              -> feature vector (raw pixels and/or HOG descriptor)
              -> z-score standardisation with training-set statistics
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = 28


# ------------------------------------------------------------------ preprocessing
def load_image(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image)
    return image


def preprocess_image(image: Image.Image, size: int = IMAGE_SIZE) -> np.ndarray:
    """Match the MedMNIST preparation: centre-crop to a square on the short edge,
    resize, convert to grayscale and scale to [0, 1]. Returns (size, size) float32."""
    if image.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", image.size, (0, 0, 0))
        rgba = image.convert("RGBA")
        background.paste(rgba, mask=rgba.split()[-1])
        image = background
    gray = image.convert("L")
    w, h = gray.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    gray = gray.crop((left, top, left + side, top + side))
    gray = gray.resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(gray, dtype=np.float32) / 255.0


def colourfulness(image: Image.Image) -> float:
    """Mean absolute channel difference; OCT scans are grayscale (~0)."""
    rgb = np.asarray(image.convert("RGB").resize((64, 64)), dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return float((np.abs(r - g) + np.abs(g - b) + np.abs(r - b)).mean() / 3.0)


# ------------------------------------------------------------------ HOG features
def hog(images: np.ndarray, cell: int = 4, bins: int = 9, block: int = 2) -> np.ndarray:
    """Histogram of Oriented Gradients for a batch of (N, H, W) grayscale images.

    1. Gradients gx, gy by central differences.
    2. Magnitude-weighted histogram of unsigned orientations (0-180 deg) per cell.
    3. Blocks of `block x block` cells are L2-Hys normalised and concatenated.
    """
    imgs = images.astype(np.float32)
    n, h, w = imgs.shape
    gx = np.zeros_like(imgs)
    gy = np.zeros_like(imgs)
    gx[:, :, 1:-1] = imgs[:, :, 2:] - imgs[:, :, :-2]
    gy[:, 1:-1, :] = imgs[:, 2:, :] - imgs[:, :-2, :]
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180.0

    # Linear interpolation of each pixel's vote between the two nearest bins.
    bin_width = 180.0 / bins
    pos = ang / bin_width - 0.5
    lo = np.floor(pos).astype(np.int64)
    frac = pos - lo
    lo_bin = lo % bins
    hi_bin = (lo + 1) % bins

    ch, cw = h // cell, w // cell
    mag = mag[:, : ch * cell, : cw * cell]
    lo_bin, hi_bin, frac = (a[:, : ch * cell, : cw * cell] for a in (lo_bin, hi_bin, frac))
    cell_idx = (np.arange(ch * cell) // cell)[:, None] * cw + (np.arange(cw * cell) // cell)[None, :]

    hist = np.zeros((n, ch * cw * bins), dtype=np.float32)
    base = cell_idx[None] * bins
    flat_hist = hist.reshape(-1)
    offset = (np.arange(n) * ch * cw * bins)[:, None, None]
    np.add.at(flat_hist, (offset + base + lo_bin).ravel(), (mag * (1 - frac)).ravel())
    np.add.at(flat_hist, (offset + base + hi_bin).ravel(), (mag * frac).ravel())
    hist = flat_hist.reshape(n, ch, cw, bins)

    by, bx = ch - block + 1, cw - block + 1
    blocks = np.stack(
        [hist[:, i : i + by, j : j + bx, :] for i in range(block) for j in range(block)], axis=3
    ).reshape(n, by, bx, block * block * bins)
    eps = 1e-6
    blocks = blocks / np.sqrt(np.sum(blocks**2, axis=-1, keepdims=True) + eps**2)
    blocks = np.minimum(blocks, 0.2)
    blocks = blocks / np.sqrt(np.sum(blocks**2, axis=-1, keepdims=True) + eps**2)
    return blocks.reshape(n, -1)


# ------------------------------------------------------------------ extractor
@dataclass
class FeatureExtractor:
    """Turns (N, 28, 28) images into standardised feature vectors."""

    kind: str = "pixels+hog"  # "pixels" | "hog" | "pixels+hog"
    mean: np.ndarray | None = field(default=None, repr=False)
    std: np.ndarray | None = field(default=None, repr=False)

    def raw_features(self, images: np.ndarray) -> np.ndarray:
        images = images.astype(np.float32)
        if images.max() > 1.0:
            images = images / 255.0
        parts = []
        if "pixels" in self.kind:
            parts.append(images.reshape(images.shape[0], -1))
        if "hog" in self.kind:
            parts.append(hog(images))
        if not parts:
            raise ValueError(f"Unknown feature kind '{self.kind}'")
        return np.concatenate(parts, axis=1)

    def fit(self, images: np.ndarray) -> "FeatureExtractor":
        feats = self.raw_features(images)
        self.mean = feats.mean(axis=0)
        self.std = feats.std(axis=0) + 1e-6
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("FeatureExtractor must be fitted first")
        return ((self.raw_features(images) - self.mean) / self.std).astype(np.float64)

    def fit_transform(self, images: np.ndarray) -> np.ndarray:
        return self.fit(images).transform(images)

    @property
    def dim(self) -> int:
        return 0 if self.mean is None else int(self.mean.shape[0])
