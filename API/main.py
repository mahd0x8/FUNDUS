"""
main.py — FastAPI application for fundus disease prediction.

Endpoints:
    GET  /health          → liveness check + model/device info
    POST /predict         → upload a fundus image, get predictions + visualisations
    GET  /classes         → list of all 19 supported disease classes

Run from the project root:
    uvicorn API.main:app --host 0.0.0.0 --port 8000 --reload

Or from inside the API/ folder:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import threading
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Resolve paths relative to this file so the API works from any cwd.
_HERE       = os.path.dirname(os.path.abspath(__file__))
_CHECKPOINT = os.environ.get(
    "CHECKPOINT_PATH",
    os.path.join(_HERE, "..", "EXPERIMENTS", "V1", "best_classifier.pt"),
)
_DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

from predictor import LABEL_COLS, CLASS_NAMES, load_model, predict   # noqa: E402


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class ClassInfo(BaseModel):
    label_code: str
    label_name: str


class PredictionItem(BaseModel):
    label_code: str
    label_name: str
    probability: float = Field(..., ge=0.0, le=1.0)
    predicted: bool


class ClassVisuals(BaseModel):
    label_code:      str
    label_name:      str
    probability:     float
    heatmap:         str = Field(..., description="Base64-encoded PNG — GradCAM heatmap")
    overlay:         str = Field(..., description="Base64-encoded PNG — GradCAM overlaid on fundus")
    polygon_overlay: str = Field(..., description="Base64-encoded PNG — hot-region polygon boundary")
    bounding_box:    str = Field(..., description="Base64-encoded PNG — bounding boxes on fundus")
    panel:           str = Field(..., description="Base64-encoded PNG — 2×2 comparison panel")


class PredictResponse(BaseModel):
    device:         str
    threshold:      float
    predictions:    List[PredictionItem]
    original_image: str   = Field(..., description="Base64-encoded PNG — preprocessed fundus image")
    visuals:        List[ClassVisuals]


class HealthResponse(BaseModel):
    status:      str
    device:      str
    num_classes: int
    checkpoint:  str


# ============================================================
# APP LIFESPAN  (model loaded once at startup)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[startup] Loading model from: {_CHECKPOINT}")
    print(f"[startup] Device: {_DEVICE}")
    app.state.model = load_model(_CHECKPOINT, _DEVICE)
    app.state.lock  = threading.Lock()   # serialise GradCAM (not thread-safe)
    print("[startup] Model ready.")
    yield
    print("[shutdown] Cleaning up.")


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="FUNDUS Disease Prediction API",
    description=(
        "Multi-label fundus image classifier for 19 retinal diseases. "
        "Upload a fundus image and receive per-class probabilities together with "
        "GradCAM heatmaps, overlays, segmentation boundaries, and bounding boxes."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROUTES
# ============================================================

@app.get("/health", response_model=HealthResponse, tags=["Utility"])
def health():
    """Liveness check — confirms the model is loaded and ready."""
    return HealthResponse(
        status="ok",
        device=_DEVICE,
        num_classes=len(LABEL_COLS),
        checkpoint=os.path.abspath(_CHECKPOINT),
    )


@app.get("/classes", response_model=List[ClassInfo], tags=["Utility"])
def list_classes():
    """Return all 19 disease classes the model can detect."""
    return [
        ClassInfo(label_code=code, label_name=CLASS_NAMES[code])
        for code in LABEL_COLS
    ]


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_endpoint(
    file:          UploadFile = File(..., description="Fundus image (JPEG, PNG, BMP, TIFF, WebP)"),
    threshold:     float      = Form(0.5,  description="Probability threshold for positive prediction (0–1)"),
    top_k_if_none: int        = Form(3,    description="Number of top classes to visualise when nothing passes the threshold"),
):
    """
    Upload a fundus image and receive:

    - **predictions** — all 19 disease classes with probabilities (sorted descending)
    - **original_image** — the preprocessed fundus image (base64 PNG)
    - **visuals** — for each predicted class (or top-k if none):
        - `heatmap` — raw GradCAM activation heatmap
        - `overlay` — GradCAM blended on the fundus
        - `polygon_overlay` — hot-region contour/fill boundary
        - `bounding_box` — axis-aligned bounding boxes
        - `panel` — 2×2 comparison panel (original / heatmap / mask / overlay)

    All images are returned as **base64-encoded PNG** strings.
    Decode with: `base64.b64decode(value)`.
    """
    if not (0.0 <= threshold <= 1.0):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="threshold must be between 0.0 and 1.0",
        )
    if not (1 <= top_k_if_none <= len(LABEL_COLS)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"top_k_if_none must be between 1 and {len(LABEL_COLS)}",
        )

    image_bytes = file.file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        with app.state.lock:
            result = predict(
                model=app.state.model,
                image_bytes=image_bytes,
                device=_DEVICE,
                threshold=threshold,
                top_k_if_none=top_k_if_none,
            )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        )

    return PredictResponse(
        device=_DEVICE,
        threshold=threshold,
        predictions=[PredictionItem(**p) for p in result["predictions"]],
        original_image=result["original_image"],
        visuals=[ClassVisuals(**v) for v in result["visuals"]],
    )
