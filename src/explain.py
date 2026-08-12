"""Grad-CAM for DFNet.

Grad-CAM weights each channel of the last feature map by the gradient of the
output logit with respect to that channel, then sums. The result is a coarse
heatmap of the regions that pushed the decision towards FAKE - the part of the
demo that makes the model's verdict inspectable instead of a bare number.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Usage:  with GradCAM(model) as cam: heatmap = cam(input_tensor)"""

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else model.cam_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = []

    def __enter__(self) -> "GradCAM":
        self._handles.append(self.target_layer.register_forward_hook(self._save_activation))
        self._handles.append(self.target_layer.register_full_backward_hook(self._save_gradient))
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __call__(self, image_tensor: torch.Tensor, target: str = "auto") -> np.ndarray:
        """image_tensor: (1,C,H,W) normalized. Returns HxW heatmap in [0,1].

        `target` picks which verdict to explain:
          "fake" - regions pushing the single logit up
          "real" - regions pushing it down
          "auto" - whichever the model actually predicted (default)

        This matters because the network has one logit, not two. Explaining
        "fake" on a confidently-real image backprops a strongly negative
        gradient, ReLU clamps the whole map to zero, and the overlay comes out
        blank - which is exactly what a user uploading a genuine photo would
        see. Explaining the predicted class always yields a usable map.
        """
        if image_tensor.dim() != 4 or image_tensor.size(0) != 1:
            raise ValueError("GradCAM expects a single image with shape (1,C,H,W)")
        if target not in {"auto", "fake", "real"}:
            raise ValueError(f"target must be auto/fake/real, got {target!r}")

        was_training = self.model.training
        self.model.eval()

        # Grad-CAM needs gradients even at inference time.
        with torch.enable_grad():
            image_tensor = image_tensor.detach().requires_grad_(True)
            logit = self.model(image_tensor).sum()
            if target == "auto":
                sign = 1.0 if logit.item() >= 0 else -1.0
            else:
                sign = 1.0 if target == "fake" else -1.0
            self.model.zero_grad(set_to_none=True)
            (sign * logit).backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("No activations captured - is the target layer in the graph?")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)     # GAP over spatial dims
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image_tensor.shape[-2:],
                            mode="bilinear", align_corners=False)
        cam = cam[0, 0].float().cpu().numpy()

        if was_training:
            self.model.train()

        peak = float(cam.max())
        if peak <= 1e-8:                    # flat map: nothing to normalise
            return np.zeros_like(cam)
        return cam / peak


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45
                    ) -> Image.Image:
    """Blend a [0,1] heatmap over the original image using OpenCV's JET colormap."""
    import cv2

    base = np.array(image.convert("RGB"))
    resized = cv2.resize(heatmap, (base.shape[1], base.shape[0]))
    colored = cv2.applyColorMap(np.uint8(255 * resized), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    blended = np.uint8((1 - alpha) * base + alpha * colored)
    return Image.fromarray(blended)
