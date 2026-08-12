"""Typed configuration objects, loadable from YAML.

Every script takes `--config path/to.yaml`; anything absent from the YAML falls
back to the defaults defined here, so configs stay short and readable.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    root: str = "data/processed"          # expects root/{train,val,test}/{real,fake}
    img_size: int = 160
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = False              # MPS does not benefit from pinned memory
    # Cap images per class for train/val only (test is always evaluated in full).
    # Lets you do a fast shakedown run straight off a read-only dataset mount.
    limit_per_class: Optional[int] = None


@dataclass
class ModelConfig:
    stem_channels: int = 32
    stage_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 384])
    blocks_per_stage: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    use_srm: bool = True                  # fixed high-pass noise-residual branch
    se_ratio: float = 0.25                # 0 disables squeeze-excitation
    dropout: float = 0.3


@dataclass
class AugConfig:
    """Probabilities for the train-time degradations.

    Deepfake detectors overfit hard to the generator's clean output, so the
    compression/rescaling augmentations matter more than the geometric ones.
    """
    hflip: float = 0.5
    jpeg: float = 0.5
    jpeg_quality: List[int] = field(default_factory=lambda: [40, 95])
    downscale: float = 0.3
    downscale_range: List[float] = field(default_factory=lambda: [0.4, 0.9])
    blur: float = 0.2
    blur_sigma: List[float] = field(default_factory=lambda: [0.3, 1.2])
    color_jitter: float = 0.5
    random_erasing: float = 0.25


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    amp: bool = False                     # only meaningful on CUDA
    early_stop_patience: int = 7
    out_dir: str = "runs/dfnet"
    seed: int = 42
    device: str = "auto"                  # auto | mps | cuda | cpu


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


_SECTIONS = {"data": DataConfig, "model": ModelConfig, "aug": AugConfig, "train": TrainConfig}


def load_config(path: str | Path | None) -> Config:
    """Build a Config from YAML, falling back to defaults for missing keys."""
    if path is None:
        return Config()

    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"Unknown config section(s): {sorted(unknown)}")

    sections = {}
    for name, cls in _SECTIONS.items():
        values = raw.get(name) or {}
        valid = {f.name for f in dataclasses.fields(cls)}
        bad = set(values) - valid
        if bad:
            raise ValueError(f"Unknown key(s) in '{name}': {sorted(bad)}")
        sections[name] = cls(**values)

    return Config(**sections)
