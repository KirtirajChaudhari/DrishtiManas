"""Image preprocessing and feature extraction (NumPy + Pillow only).

The same functions run at training time (on OCTMNIST arrays) and at inference
time (on images uploaded through the web app), so the network always sees
identically prepared inputs.

Pipeline for one image:
    any image -> grayscale -> centre square crop -> resize to SxS (28 or 64) -> [0, 1]
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
def hog(images: np.ndarray, cell: int = 4, bins: int = 9, block: int = 2, chunk: int = 4096) -> np.ndarray:
    """Chunked wrapper around `_hog` so large datasets do not exhaust memory."""
    if images.shape[0] > chunk:
        return np.concatenate(
            [_hog(images[i : i + chunk], cell, bins, block) for i in range(0, images.shape[0], chunk)]
        )
    return _hog(images, cell, bins, block)


def _hog(images: np.ndarray, cell: int = 4, bins: int = 9, block: int = 2) -> np.ndarray:
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

    size = n * ch * cw * bins
    offset = (np.arange(n) * ch * cw * bins)[:, None, None]
    base = offset + cell_idx[None] * bins
    flat_hist = np.bincount((base + lo_bin).ravel(), weights=(mag * (1 - frac)).ravel(), minlength=size)
    flat_hist += np.bincount((base + hi_bin).ravel(), weights=(mag * frac).ravel(), minlength=size)
    hist = flat_hist.astype(np.float32).reshape(n, ch, cw, bins)

    by, bx = ch - block + 1, cw - block + 1
    blocks = np.stack([hist[:, i : i + by, j : j + bx, :] for i in range(block) for j in range(block)], axis=3).reshape(
        n, by, bx, block * block * bins
    )
    eps = 1e-6
    blocks = blocks / np.sqrt(np.sum(blocks**2, axis=-1, keepdims=True) + eps**2)
    blocks = np.minimum(blocks, 0.2)
    blocks = blocks / np.sqrt(np.sum(blocks**2, axis=-1, keepdims=True) + eps**2)
    return blocks.reshape(n, -1)


# ------------------------------------------------------------------ extractor
def resize_batch(images: np.ndarray, size: int) -> np.ndarray:
    """Bicubic resize of a (N, H, W) batch to (N, size, size) float32 in [0, 1].

    uint8 input is treated as 0-255; float input is assumed to already be in [0, 1]. The
    scale is decided by dtype, not by the pixel values, so a nearly-black uint8 image (max
    pixel <= 1) is still divided by 255.
    """
    if images.shape[1] == size:
        out = images.astype(np.float32)
        return out / 255.0 if images.dtype == np.uint8 else out
    src = images if images.dtype == np.uint8 else np.clip(images * 255, 0, 255).astype(np.uint8)
    out = np.stack(
        [np.asarray(Image.fromarray(im).resize((size, size), Image.Resampling.BICUBIC)) for im in src]
    ).astype(np.float32)
    return out / 255.0


@dataclass
class FeatureExtractor:
    """Turns (N, S, S) grayscale images into standardised feature vectors.

    kind        "pixels" | "hog" | "pixels+hog"
    image_size  side length every image is preprocessed to (28 or 64)
    pixel_size  raw-pixel features are taken after downsampling to this size
    hog_cells   HOG cell sizes in pixels (on the image_size image); more than one
                value concatenates a coarse descriptor (large-scale contrast, e.g.
                the overall retinal layer structure) with a fine one (small-scale
                texture, e.g. individual drusen deposits), which a single cell
                size cannot represent at once
    """

    kind: str = "pixels+hog"
    image_size: int = IMAGE_SIZE
    pixel_size: int = IMAGE_SIZE
    hog_cells: list[int] = field(default_factory=lambda: [4])
    mean: np.ndarray | None = field(default=None, repr=False)
    std: np.ndarray | None = field(default=None, repr=False)

    def raw_features(self, images: np.ndarray) -> np.ndarray:
        images = resize_batch(images, self.image_size)
        parts = []
        if "pixels" in self.kind:
            pix = images if self.pixel_size == self.image_size else resize_batch(images, self.pixel_size)
            parts.append(pix.reshape(pix.shape[0], -1))
        if "hog" in self.kind:
            parts.extend(hog(images, cell=c) for c in self.hog_cells)
        if not parts:
            raise ValueError(f"Unknown feature kind '{self.kind}'")
        return np.concatenate(parts, axis=1).astype(np.float32)

    def fit(self, images: np.ndarray) -> "FeatureExtractor":
        feats = self.raw_features(images)
        self.mean = feats.mean(axis=0)
        self.std = feats.std(axis=0) + 1e-6
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("FeatureExtractor must be fitted first")
        return ((self.raw_features(images) - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, images: np.ndarray) -> np.ndarray:
        return self.fit(images).transform(images)

    @property
    def dim(self) -> int:
        return 0 if self.mean is None else int(self.mean.shape[0])

    def spec(self) -> dict:
        return {
            "kind": self.kind,
            "image_size": self.image_size,
            "pixel_size": self.pixel_size,
            "hog_cells": list(self.hog_cells),
        }
