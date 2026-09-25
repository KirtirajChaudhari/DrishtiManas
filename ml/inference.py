"""Load the trained model bundle and classify images. Depends only on NumPy + Pillow."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .data import CLASSES
from .features import FeatureExtractor, colourfulness, preprocess_image
from .nn.mlp import MLP, MLPConfig

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.npz"


def save_bundle(
    path: str | Path,
    models: MLP | list[MLP],
    extractor: FeatureExtractor,
    meta: dict,
    class_bias: np.ndarray | None = None,
) -> None:
    """Save one or more models (an ensemble) plus the shared feature extractor and calibration."""
    members = [models] if isinstance(models, MLP) else list(models)
    data: dict[str, object] = {
        "config": json.dumps(asdict(members[0].config)),
        "ensemble_size": len(members),
        "feature_spec": json.dumps(extractor.spec()),
        "feat_mean": extractor.mean.astype(np.float32),
        "feat_std": extractor.std.astype(np.float32),
        "meta": json.dumps(meta),
        "class_bias": np.zeros(len(CLASSES), dtype=np.float32)
        if class_bias is None
        else np.asarray(class_bias, dtype=np.float32),
    }
    for i, member in enumerate(members):
        for key, value in member.state_dict().items():
            data[f"m{i}_{key}"] = value
    np.savez_compressed(path, **data)


class OCTClassifier:
    """Wraps an ensemble of one or more MLPs that share one feature extractor and one
    class-prior calibration (an additive log-bias tuned on validation; see `ml.train.tune_class_bias`).
    A single-model bundle (ensemble_size == 1) behaves exactly like a plain classifier.
    """

    def __init__(self, path: str | Path = MODEL_PATH) -> None:
        data = np.load(path, allow_pickle=False)
        config = MLPConfig(**json.loads(str(data["config"])))
        n = int(data["ensemble_size"]) if "ensemble_size" in data.files else 1
        self.models: list[MLP] = []
        for i in range(n):
            member = MLP(config)
            prefix = f"m{i}_"
            member.load_state_dict({k[len(prefix) :]: data[k] for k in data.files if k.startswith(prefix)})
            self.models.append(member)
        self.extractor = FeatureExtractor(
            **json.loads(str(data["feature_spec"])), mean=data["feat_mean"], std=data["feat_std"]
        )
        self.class_bias = (
            np.asarray(data["class_bias"], dtype=np.float64) if "class_bias" in data.files else np.zeros(len(CLASSES))
        )
        self.meta = json.loads(str(data["meta"]))
        self.classes = CLASSES

    @property
    def ensemble_size(self) -> int:
        return len(self.models)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Ensemble-averaged, class-bias-calibrated probabilities (what is actually served)."""
        raw = np.mean([m.predict_proba(features) for m in self.models], axis=0)
        # log(p) + bias is the calibrated decision rule; exponentiating and renormalising turns
        # it back into a probability distribution without changing which class has the highest
        # score (argmax is invariant to the per-row rescaling that renormalisation performs).
        adjusted = np.exp(np.log(raw + 1e-12) + self.class_bias)
        return adjusted / adjusted.sum(axis=1, keepdims=True)

    def predict_image(self, image: Image.Image) -> dict:
        t0 = time.perf_counter()
        pixels = preprocess_image(image, size=self.extractor.image_size)
        features = self.extractor.transform(pixels[None])
        probs = self.predict_proba(features)[0]
        top = int(probs.argmax())

        warnings = []
        if colourfulness(image) > 12:
            warnings.append(
                "This image is in colour. The model was trained on grayscale retinal OCT B-scans, "
                "so predictions for other image types are not meaningful."
            )
        if probs[top] < 0.5:
            warnings.append("Low confidence: the model is unsure between several classes.")

        return {
            "prediction": {**self.classes[top], "probability": float(probs[top])},
            "probabilities": [{**c, "probability": float(p)} for c, p in zip(self.classes, probs.tolist())],
            "input_pixels": np.round(pixels * 255).astype(int).tolist(),
            "warnings": warnings,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
