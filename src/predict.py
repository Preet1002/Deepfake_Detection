"""Inference: a reusable Detector class plus a small CLI.

    python -m src.predict --checkpoint runs/dfnet/best.pt --image path/to.jpg --cam out.jpg
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image

from .config import Config, ModelConfig
from .data.face import extract_faces
from .data.transforms import build_eval_transform
from .explain import GradCAM, overlay_heatmap
from .models.dfnet import build_model
from .utils.common import load_checkpoint, pick_device


@dataclass
class FaceResult:
    box: Optional[Tuple[int, int, int, int]]    # None = centre-crop fallback
    fake_probability: float
    label: str
    heatmap_image: Optional[Image.Image] = None


@dataclass
class ImageResult:
    fake_probability: float          # aggregated over faces
    label: str
    faces: List[FaceResult]
    face_detected: bool


class Detector:
    """Loads a trained DFNet checkpoint and classifies images end to end."""

    def __init__(self, checkpoint_path: str | Path, device: str = "auto",
                 threshold: float = 0.5):
        self.device = pick_device(device)
        self.threshold = threshold

        checkpoint = load_checkpoint(checkpoint_path, map_location=self.device)
        saved = checkpoint.get("config", {})
        model_config = ModelConfig(**saved["model"]) if "model" in saved else ModelConfig()
        self.img_size = saved.get("data", {}).get("img_size", Config().data.img_size)

        self.model = build_model(model_config).to(self.device).eval()
        self.model.load_state_dict(checkpoint["model_state"])
        self.transform = build_eval_transform(self.img_size)
        self.checkpoint_metrics = checkpoint.get("metrics", {})

    def _label(self, probability: float) -> str:
        return "FAKE" if probability >= self.threshold else "REAL"

    @torch.no_grad()
    def _score(self, face: Image.Image) -> float:
        tensor = self.transform(face).unsqueeze(0).to(self.device)
        return float(torch.sigmoid(self.model(tensor))[0])

    def _cam(self, face: Image.Image) -> Image.Image:
        tensor = self.transform(face).unsqueeze(0).to(self.device)
        with GradCAM(self.model) as cam:
            heatmap = cam(tensor)
        return overlay_heatmap(face, heatmap)

    def predict(self, image: str | Path | Image.Image, *, detect_faces: bool = True,
                explain: bool = False, max_faces: int = 4) -> ImageResult:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")

        if detect_faces:
            crops = extract_faces(image, max_faces=max_faces)
            # A None box means detection found nothing and we fell back to a
            # centre crop, so the caller should discount the result.
            face_detected = any(box is not None for _, box in crops)
        else:
            crops = [(image, (0, 0, image.width, image.height))]
            face_detected = False

        faces = []
        for crop, box in crops:
            probability = self._score(crop)
            faces.append(FaceResult(
                box=box,
                fake_probability=probability,
                label=self._label(probability),
                heatmap_image=self._cam(crop) if explain else None,
            ))

        # Aggregate with max: one convincing fake face is enough to flag the
        # image, and averaging would let extra real faces mask it.
        overall = max(face.fake_probability for face in faces)
        return ImageResult(fake_probability=overall, label=self._label(overall),
                           faces=faces, face_detected=face_detected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify an image as real or fake")
    parser.add_argument("--checkpoint", default="runs/dfnet/best.pt")
    parser.add_argument("--image", required=True, help="image file or directory")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-face-crop", action="store_true",
                        help="classify the whole image instead of detected faces")
    parser.add_argument("--cam", default=None,
                        help="write a Grad-CAM overlay here (single image only)")
    args = parser.parse_args()

    detector = Detector(args.checkpoint, threshold=args.threshold)

    path = Path(args.image)
    paths = sorted(p for p in path.rglob("*")
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}) \
        if path.is_dir() else [path]

    for image_path in paths:
        result = detector.predict(image_path, detect_faces=not args.no_face_crop,
                                  explain=args.cam is not None)
        print(json.dumps({
            "image": str(image_path),
            "label": result.label,
            "fake_probability": round(result.fake_probability, 4),
            "faces": len(result.faces),
            "face_detected": result.face_detected,
        }))
        if args.cam and result.faces[0].heatmap_image is not None:
            Path(args.cam).parent.mkdir(parents=True, exist_ok=True)
            result.faces[0].heatmap_image.save(args.cam)
            print(f"Grad-CAM written to {args.cam}")


if __name__ == "__main__":
    main()
