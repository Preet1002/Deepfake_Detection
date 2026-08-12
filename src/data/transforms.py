"""Train/eval image transforms.

The degradation augmentations (JPEG, downscale, blur) are the important ones.
A detector trained only on pristine generator output learns the generator's
clean high-frequency signature and collapses the moment an image has been
through a messaging app. Re-compressing during training forces it to rely on
artefacts that survive real-world handling.
"""
from __future__ import annotations

import io
import random

from PIL import Image, ImageFilter
from torchvision import transforms

# Dataset-agnostic normalisation. We deliberately do NOT use ImageNet statistics
# since nothing here is pretrained on ImageNet.
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


class RandomJPEG:
    """Re-encode through JPEG at a random quality."""

    def __init__(self, p: float = 0.5, quality_range=(40, 95)):
        self.p = p
        self.quality_range = tuple(quality_range)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class RandomDownscale:
    """Shrink then upscale back, simulating a low-resolution source."""

    def __init__(self, p: float = 0.3, scale_range=(0.4, 0.9)):
        self.p = p
        self.scale_range = tuple(scale_range)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        w, h = img.size
        scale = random.uniform(*self.scale_range)
        small = img.resize((max(8, int(w * scale)), max(8, int(h * scale))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


class RandomBlur:
    def __init__(self, p: float = 0.2, sigma_range=(0.3, 1.2)):
        self.p = p
        self.sigma_range = tuple(sigma_range)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(random.uniform(*self.sigma_range)))


def build_train_transform(img_size: int, aug) -> transforms.Compose:
    ops = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=aug.hflip),
        RandomDownscale(aug.downscale, aug.downscale_range),
        RandomBlur(aug.blur, aug.blur_sigma),
        RandomJPEG(aug.jpeg, aug.jpeg_quality),
    ]
    if aug.color_jitter > 0:
        ops.append(transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02)],
            p=aug.color_jitter,
        ))
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if aug.random_erasing > 0:
        ops.append(transforms.RandomErasing(p=aug.random_erasing, scale=(0.02, 0.12)))
    return transforms.Compose(ops)


def build_eval_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def denormalize(tensor):
    """Undo Normalize for visualisation. Accepts (C,H,W) or (B,C,H,W)."""
    import torch

    mean = torch.tensor(MEAN, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(STD, device=tensor.device).view(-1, 1, 1)
    if tensor.dim() == 4:
        mean, std = mean.unsqueeze(0), std.unsqueeze(0)
    return (tensor * std + mean).clamp(0, 1)
