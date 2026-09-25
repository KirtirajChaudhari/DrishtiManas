"""Tests for preprocessing and feature extraction."""

import numpy as np
from PIL import Image

from ml.features import FeatureExtractor, hog, preprocess_image, resize_batch


def test_preprocess_crops_resizes_and_scales():
    img = Image.fromarray(np.full((300, 500, 3), 255, dtype=np.uint8))
    out = preprocess_image(img)
    assert out.shape == (28, 28)
    assert out.dtype == np.float32
    assert np.allclose(out, 1.0)


def test_preprocess_handles_transparency():
    img = Image.new("RGBA", (40, 40), (255, 255, 255, 0))
    assert np.allclose(preprocess_image(img), 0.0)


def test_hog_shape_and_normalisation():
    feats = hog(np.random.default_rng(0).random((2, 28, 28)))
    assert feats.shape == (2, 6 * 6 * 4 * 9)
    assert feats.max() <= 0.2 + 1e-3 or np.allclose(np.linalg.norm(feats.reshape(2, 36, 36), axis=-1), 1, atol=1e-3)


def test_extractor_standardises_training_features():
    imgs = np.random.default_rng(1).integers(0, 255, (50, 28, 28)).astype(np.uint8)
    fx = FeatureExtractor(kind="pixels+hog")
    X = fx.fit_transform(imgs)
    assert X.shape == (50, 784 + 1296)
    assert np.allclose(X.mean(axis=0), 0, atol=1e-4)


def test_resize_batch_scales_uint8_by_dtype_not_value():
    dark = np.ones((1, 8, 8), dtype=np.uint8)  # max pixel value 1 -- must still be divided by 255
    assert np.allclose(resize_batch(dark, 8), 1 / 255)
    floats = np.full((1, 8, 8), 0.5, dtype=np.float32)  # already in [0, 1]: left unchanged
    assert np.allclose(resize_batch(floats, 8), 0.5)


def test_multiscale_hog_concatenates_each_cell_size():
    imgs = np.random.default_rng(2).integers(0, 255, (20, 32, 32)).astype(np.uint8)
    single = FeatureExtractor(kind="hog", image_size=32, hog_cells=[8]).fit_transform(imgs)
    multi = FeatureExtractor(kind="hog", image_size=32, hog_cells=[8, 4]).fit_transform(imgs)
    fine_only = FeatureExtractor(kind="hog", image_size=32, hog_cells=[4]).fit_transform(imgs)
    assert multi.shape[1] == single.shape[1] + fine_only.shape[1]
