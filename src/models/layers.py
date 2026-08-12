"""Building blocks for DFNet, all implemented from scratch (no pretrained weights)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# The three 5x5 Spatial Rich Model high-pass kernels used in image forensics.
# They are *fixed* DSP filters, not learned weights: each one cancels the smooth
# image content and leaves the local noise residual behind. Generative models
# reproduce faces convincingly but leave a different residual/upsampling
# fingerprint than a camera sensor does, which is exactly what we want the
# network to see.
_SRM_KERNELS = [
    (
        [[0, 0, 0, 0, 0],
         [0, -1, 2, -1, 0],
         [0, 2, -4, 2, 0],
         [0, -1, 2, -1, 0],
         [0, 0, 0, 0, 0]], 4.0,
    ),
    (
        [[-1, 2, -2, 2, -1],
         [2, -6, 8, -6, 2],
         [-2, 8, -12, 8, -2],
         [2, -6, 8, -6, 2],
         [-1, 2, -2, 2, -1]], 12.0,
    ),
    (
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 1, -2, 1, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]], 2.0,
    ),
]


class SRMFilter(nn.Module):
    """Fixed high-pass filter bank: 3 RGB channels -> 9 noise-residual channels.

    Applied depthwise so each colour channel keeps its own residual; colour
    channels are informative because most generators upsample in a way that
    correlates chroma noise.
    """

    def __init__(self, in_channels: int = 3, clip: float = 3.0):
        super().__init__()
        self.in_channels = in_channels
        self.clip = clip

        kernels = torch.stack([
            torch.tensor(k, dtype=torch.float32) / q for k, q in _SRM_KERNELS
        ])                                              # (3, 5, 5)
        weight = kernels.repeat(in_channels, 1, 1)      # (3*in_ch, 5, 5)
        weight = weight.unsqueeze(1)                    # (3*in_ch, 1, 5, 5)
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # groups=in_channels with 3 filters each -> depthwise application
        out = F.conv2d(x, self.weight.to(x.dtype), padding=2, groups=self.in_channels)
        # Residuals are heavy-tailed; clipping keeps a few hot pixels from
        # dominating the batch-norm statistics downstream.
        return torch.clamp(out, -self.clip, self.clip)

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.in_channels * 3}, clip={self.clip}"


class SqueezeExcite(nn.Module):
    """Channel attention: lets the net weight residual-sensitive channels up."""

    def __init__(self, channels: int, ratio: float = 0.25):
        super().__init__()
        hidden = max(8, int(channels * ratio))
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean((2, 3), keepdim=True)
        s = F.silu(self.fc1(s))
        return x * torch.sigmoid(self.fc2(s))


class ResidualBlock(nn.Module):
    """Pre-activation residual block with optional stride and SE attention."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 se_ratio: float = 0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcite(out_channels, se_ratio) if se_ratio > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.silu(out + identity)


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.bn(self.conv(x)))
