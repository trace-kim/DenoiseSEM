"""Validated, serializable settings for SEM noise characterization."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from numbers import Real
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AnalysisConfig:
    expected_frames: int = 128
    min_frames: int = 8
    roi: tuple[int, int, int, int] | None = None  # y0, y1, x0, x1; stop exclusive
    frame_interval_s: float | None = None
    pixel_size_nm: float | None = None
    black_level: float | None = None
    white_level: float | None = None
    registration: str = "fit"          # "fit": the eight-parameter fit per frame; "none": frames taken as aligned
    registration_sigma: float = 1.0    # light blur (px) of both copies before the fit
    sample_pixels: int = 8192
    distribution_samples: int = 100000
    intensity_bins: int = 12
    flat_fraction: float = 0.5
    max_lag: int = 32
    spatial_pairs: int = 16
    spatial_max_side: int = 512
    seed: int = 17

    def __post_init__(self) -> None:
        for name in ("expected_frames", "min_frames", "sample_pixels", "distribution_samples",
                     "intensity_bins", "max_lag", "spatial_pairs", "spatial_max_side"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_frames < 4 or self.intensity_bins < 3:
            raise ValueError("min_frames must be >= 4 and intensity_bins >= 3")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("frame_interval_s", "pixel_size_nm", "registration_sigma"):
            value = getattr(self, name)
            if value is None and name == "registration_sigma":
                raise ValueError(f"{name} must be finite and positive")
            if value is not None and (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        for name in ("black_level", "white_level", "flat_fraction"):
            value = getattr(self, name)
            if value is None and name == "flat_fraction":
                raise ValueError(f"{name} must be finite")
            if value is not None and (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.flat_fraction < 1:
            raise ValueError("flat_fraction must be in (0, 1)")
        if self.black_level is not None and self.white_level is not None:
            if self.black_level >= self.white_level:
                raise ValueError("black_level must be below white_level")
        if self.registration not in {"fit", "none"}:
            raise ValueError("registration must be fit or none")
        if self.roi is not None:
            if len(self.roi) != 4 or any(type(v) is not int for v in self.roi):
                raise ValueError("roi must contain four integers: y0, y1, x0, x1")
            y0, y1, x0, x1 = self.roi
            if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0:
                raise ValueError("roi must have nonnegative starts and increasing stops")
            object.__setattr__(self, "roi", tuple(self.roi))


def load_config(path: str | Path | None) -> AnalysisConfig:
    """Read strict YAML settings; input/output paths belong to the CLI."""
    if path is None:
        return AnalysisConfig()
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("config must be a YAML mapping")
    unknown = set(values) - {f.name for f in fields(AnalysisConfig)}
    if unknown:
        raise ValueError(f"unknown analysis settings: {sorted(unknown)}")
    return AnalysisConfig(**values)
