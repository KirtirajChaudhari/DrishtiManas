"""DrishtiManas API: retinal OCT image classification with a from-scratch MLP.

Endpoints
    GET  /api/health            liveness + model status
    GET  /api/model             training report (metrics, curves, tuning results)
    GET  /api/samples           held-out test images the UI offers as examples
    GET  /api/samples/{file}    one sample image
    POST /api/predict           multipart image upload -> class probabilities

If the React build exists (frontend/dist) it is served from "/", so a single
container hosts the whole application.
"""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.features import load_image  # noqa: E402
from ml.inference import OCTClassifier  # noqa: E402

VERSION = "2.0.0"
ARTIFACTS_DIR = Path(os.environ.get("DRISHTI_ARTIFACTS_DIR", ROOT / "artifacts"))
FRONTEND_DIST = Path(os.environ.get("DRISHTI_FRONTEND_DIST", ROOT / "frontend" / "dist"))
MAX_UPLOAD_BYTES = int(float(os.environ.get("DRISHTI_MAX_UPLOAD_MB", "8")) * 1024 * 1024)
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("DRISHTI_ALLOWED_ORIGINS", "*").split(",") if o.strip()]
Image.MAX_IMAGE_PIXELS = 40_000_000  # refuse decompression bombs

app = FastAPI(
    title="DrishtiManas API",
    version=VERSION,
    description="Retinal OCT image classification (CNV / DME / Drusen / Normal) with a NumPy multilayer perceptron.",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_classifier() -> OCTClassifier:
    path = ARTIFACTS_DIR / "model.npz"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Model not trained yet. Run `python -m ml.train`.")
    return OCTClassifier(path)


@lru_cache(maxsize=1)
def get_report() -> dict:
    path = ARTIFACTS_DIR / "report.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Training report not found. Run `python -m ml.train`.")
    return json.loads(path.read_text())


@app.get("/api/health", tags=["meta"])
def health() -> dict:
    try:
        clf = get_classifier()
        model = {"loaded": True, **clf.meta}
    except HTTPException:
        model = {"loaded": False}
    return {"status": "ok", "version": VERSION, "model": model}


@app.get("/api/model", tags=["model"])
def model_report() -> dict:
    return get_report()


@app.get("/api/samples", tags=["model"])
def samples() -> list[dict]:
    return [{**s, "url": f"/api/samples/{s['file']}"} for s in get_report().get("samples", [])]


@app.get("/api/samples/{name}", tags=["model"], response_class=FileResponse)
def sample_file(name: str) -> FileResponse:
    allowed = {s["file"] for s in get_report().get("samples", [])}
    if name not in allowed:
        raise HTTPException(status_code=404, detail="Unknown sample")
    return FileResponse(ARTIFACTS_DIR / "samples" / name, media_type="image/png")


@app.post("/api/predict", tags=["model"])
async def predict(
    file: UploadFile = File(..., description="Retinal OCT image (PNG / JPEG / BMP / TIFF / WebP)"),
) -> dict:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    try:
        image = load_image(data)
        image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="Could not read the file as an image.") from exc

    result = get_classifier().predict_image(image)
    return {"filename": file.filename, "image_size": list(image.size), **result}


# ---------------------------------------------------------------- single-page app
if FRONTEND_DIST.joinpath("index.html").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        candidate = (FRONTEND_DIST / path).resolve()
        if path and candidate.is_file() and FRONTEND_DIST.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
