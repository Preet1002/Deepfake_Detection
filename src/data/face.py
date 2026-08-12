"""Face detection and cropping with OpenCV's bundled Haar cascade.

Training images are tight, aligned face crops. An arbitrary photo the user
uploads is not, so inference must crop the face first or the model sees a
distribution it never trained on. We use the cascade that ships with
opencv-python: no extra download and no pretrained deep network, which keeps
the "from scratch" claim about the classifier honest.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

_cascade: cv2.CascadeClassifier | None = None

Box = Tuple[int, int, int, int]     # x, y, w, h


def _get_cascade() -> cv2.CascadeClassifier:
    global _cascade
    if _cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {path}")
        _cascade = cascade
    return _cascade


def detect_faces(image: Image.Image, min_size: int = 60) -> List[Box]:
    """Return face boxes, largest first."""
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _get_cascade().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size)
    )
    boxes = [tuple(int(v) for v in face) for face in faces]
    return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)


def crop_face(image: Image.Image, box: Box, margin: float = 0.3) -> Image.Image:
    """Crop a box with margin, clamped to the image bounds.

    The margin matters: blending seams and hairline artefacts sit just outside
    the detector's tight box, and those are strong deepfake cues.
    """
    x, y, w, h = box
    pad_x, pad_y = int(w * margin), int(h * margin)
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(image.width, x + w + pad_x)
    bottom = min(image.height, y + h + pad_y)
    return image.crop((left, top, right, bottom))


def extract_faces(image: Image.Image, margin: float = 0.3, max_faces: int = 4
                  ) -> List[Tuple[Image.Image, Optional[Box]]]:
    """Detect and crop up to `max_faces` faces.

    Falls back to a centre square crop when nothing is detected, so the pipeline
    still returns a prediction on profile shots and stylised images rather than
    failing outright. That fallback is signalled by a `None` box, and callers
    should treat its prediction as unreliable: the model was trained on face
    crops, not whole scenes.
    """
    boxes = detect_faces(image)[:max_faces]
    if not boxes:
        side = min(image.width, image.height)
        left = (image.width - side) // 2
        top = (image.height - side) // 2
        return [(image.crop((left, top, left + side, top + side)), None)]
    return [(crop_face(image, box, margin), box) for box in boxes]
