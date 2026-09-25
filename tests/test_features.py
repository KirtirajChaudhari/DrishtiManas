"""Tests for preprocessing and feature extraction."""
import numpy as np
from PIL import Image

from ml.features import FeatureExtractor, hog, preprocess_image


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
