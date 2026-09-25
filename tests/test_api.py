"""API tests against the committed model artifacts."""
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import ARTIFACTS_DIR, app

pytestmark = pytest.mark.skipif(not (ARTIFACTS_DIR / "model.npz").exists(), reason="model not trained")
client = TestClient(app)


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_health_reports_model():
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model"]["loaded"] is True


def test_model_report_has_required_sections():
    body = client.get("/api/model").json()
    for key in ("dataset", "features", "baselines", "tuning", "final", "gradient_check"):
        assert key in body
    test = body["final"]["test"]
    for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "confusion_matrix"):
        assert metric in test


def test_predict_sample_image_returns_distribution():
    sample = client.get("/api/samples").json()[0]
    img = client.get(sample["url"])
    assert img.status_code == 200
    res = client.post("/api/predict", files={"file": ("s.png", img.content, "image/png")})
    assert res.status_code == 200
    body = res.json()
    probs = [p["probability"] for p in body["probabilities"]]
    assert len(probs) == 4
    assert sum(probs) == pytest.approx(1.0, abs=1e-4)
    assert body["prediction"]["code"] in {"CNV", "DME", "DRUSEN", "NORMAL"}
    assert len(body["input_pixels"]) == 28


def test_colour_image_triggers_warning():
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    res = client.post("/api/predict", files={"file": ("red.png", png_bytes(rgb), "image/png")})
    assert res.status_code == 200
    assert res.json()["warnings"]


def test_rejects_non_image_and_empty_uploads():
    assert client.post("/api/predict", files={"file": ("a.txt", b"hello", "text/plain")}).status_code == 400
    assert client.post("/api/predict", files={"file": ("a.png", b"", "image/png")}).status_code == 400


def test_unknown_sample_is_404():
    assert client.get("/api/samples/../../etc/passwd").status_code == 404
    assert client.get("/api/samples/nope.png").status_code == 404
