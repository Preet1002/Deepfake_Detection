"""FastAPI backend for the deepfake detector.

    uvicorn app.main:app --reload
    DFNET_CHECKPOINT=runs/dfnet/best.pt uvicorn app.main:app
"""
from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.predict import Detector

CHECKPOINT = os.environ.get("DFNET_CHECKPOINT", "runs/dfnet/best.pt")
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Deepfake Detection API", version="1.0")
_detector: Optional[Detector] = None


def get_detector() -> Detector:
    """Load the model once, on first request rather than at import time."""
    global _detector
    if _detector is None:
        if not Path(CHECKPOINT).exists():
            raise HTTPException(
                status_code=503,
                detail=f"No trained model at '{CHECKPOINT}'. Train one first "
                       f"(python -m src.train --config configs/default.yaml) or set "
                       f"DFNET_CHECKPOINT to an existing checkpoint.",
            )
        _detector = Detector(CHECKPOINT)
    return _detector


def to_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


@app.get("/api/health")
def health():
    checkpoint_exists = Path(CHECKPOINT).exists()
    payload = {
        "status": "ok" if checkpoint_exists else "no_model",
        "checkpoint": CHECKPOINT,
        "model_loaded": _detector is not None,
    }
    if _detector is not None:
        payload["device"] = str(_detector.device)
        payload["val_metrics"] = _detector.checkpoint_metrics
    return payload


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    explain: bool = Query(True, description="return a Grad-CAM overlay"),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported file type '{file.content_type}'. "
                                 f"Upload a JPEG, PNG, WebP or BMP image.")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is larger than 15 MB.")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode that file as an image.")

    detector = get_detector()
    detector.threshold = threshold

    started = time.perf_counter()
    result = detector.predict(image, detect_faces=True, explain=explain)
    elapsed_ms = (time.perf_counter() - started) * 1000

    return {
        "label": result.label,
        "fake_probability": round(result.fake_probability, 4),
        "confidence": round(max(result.fake_probability,
                                1 - result.fake_probability), 4),
        "face_detected": result.face_detected,
        "inference_ms": round(elapsed_ms, 1),
        "threshold": threshold,
        "faces": [
            {
                "box": None if face.box is None else {
                    "x": face.box[0], "y": face.box[1],
                    "w": face.box[2], "h": face.box[3]},
                "fake_probability": round(face.fake_probability, 4),
                "label": face.label,
                "heatmap": to_data_uri(face.heatmap_image) if face.heatmap_image else None,
                "heatmap_target": face.heatmap_target,
            }
            for face in result.faces
        ],
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
