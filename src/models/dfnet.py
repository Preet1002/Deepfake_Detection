"""DFNet: a two-stream CNN for image deepfake detection, trained from scratch.

Design rationale (useful for the report):

  RGB stream     sees semantic/appearance artefacts - asymmetric eyes, warped
                 teeth, blending seams around the face boundary.
  Noise stream   sees the SRM high-pass residual, where GAN/diffusion
                 upsampling fingerprints and the absence of camera sensor noise
                 show up. This is what keeps the model honest when the fake is
                 semantically perfect.

The two stems are fused early with a 1x1 convolution and share the residual
trunk, which is far cheaper than running two full backbones and still gives the
network access to both signals at every depth.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn

from .layers import ConvBNAct, ResidualBlock, SRMFilter


class DFNet(nn.Module):
    def __init__(
        self,
        stem_channels: int = 32,
        stage_channels: Sequence[int] = (64, 128, 256, 384),
        blocks_per_stage: Sequence[int] = (2, 2, 2, 2),
        use_srm: bool = True,
        se_ratio: float = 0.25,
        dropout: float = 0.3,
        in_channels: int = 3,
    ):
        super().__init__()
        if len(stage_channels) != len(blocks_per_stage):
            raise ValueError("stage_channels and blocks_per_stage must be the same length")

        self.use_srm = use_srm
        self.stage_channels = list(stage_channels)

        # --- stem: RGB branch (stride 2) ---
        self.rgb_stem = ConvBNAct(in_channels, stem_channels, kernel_size=3, stride=2)

        # --- stem: noise-residual branch (stride 2) ---
        if use_srm:
            self.srm = SRMFilter(in_channels)
            self.noise_stem = ConvBNAct(in_channels * 3, stem_channels, kernel_size=3, stride=2)
            fused_in = stem_channels * 2
        else:
            self.srm = None
            self.noise_stem = None
            fused_in = stem_channels

        self.fuse = ConvBNAct(fused_in, stem_channels, kernel_size=1, stride=1)

        # --- residual trunk: each stage halves resolution on its first block ---
        stages: List[nn.Module] = []
        prev = stem_channels
        for channels, n_blocks in zip(stage_channels, blocks_per_stage):
            blocks = [ResidualBlock(prev, channels, stride=2, se_ratio=se_ratio)]
            blocks += [ResidualBlock(channels, channels, stride=1, se_ratio=se_ratio)
                       for _ in range(n_blocks - 1)]
            stages.append(nn.Sequential(*blocks))
            prev = channels
        self.stages = nn.ModuleList(stages)

        # --- head: single logit, BCEWithLogits during training ---
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(prev, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final stage's feature map (used by Grad-CAM)."""
        feat = self.rgb_stem(x)
        if self.use_srm:
            noise = self.noise_stem(self.srm(x))
            feat = torch.cat([feat, noise], dim=1)
        feat = self.fuse(feat)
        for stage in self.stages:
            feat = stage(feat)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        pooled = self.pool(feat).flatten(1)
        return self.classifier(self.dropout(pooled)).squeeze(1)   # (B,) raw logits

    @property
    def cam_layer(self) -> nn.Module:
        """Layer whose activations Grad-CAM explains."""
        return self.stages[-1]


def build_model(model_config) -> DFNet:
    """Construct DFNet from a ModelConfig (or any object with the same fields)."""
    return DFNet(
        stem_channels=model_config.stem_channels,
        stage_channels=model_config.stage_channels,
        blocks_per_stage=model_config.blocks_per_stage,
        use_srm=model_config.use_srm,
        se_ratio=model_config.se_ratio,
        dropout=model_config.dropout,
    )
