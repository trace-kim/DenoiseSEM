"""Synthetic image builders for the sem_segment tests.

Images are synthesised into ``tmp_path`` rather than committed: the repository's
``.gitignore`` excludes ``*.png`` globally, and a generator keeps the ground
truth (edge position, radius, blur width) available to the assertions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from sem_segment.config import Config


def gaussian_blurred_step(
    *,
    height: int = 64,
    width: int = 96,
    edge_x: float = 40.3,
    sigma: float = 1.5,
    low: float = 0.2,
    high: float = 0.8,
) -> np.ndarray:
    """A vertical step edge at a known sub-pixel position, blurred by a known sigma.

    The analytic form is used rather than convolving a hard step, so the edge
    position is exact and independent of any filter implementation:
    ``I(x) = low + (high - low) * 0.5 * (1 + erf((x - edge_x) / (sqrt(2) sigma)))``.
    Pixel centres are at integer coordinates.
    """
    from scipy.special import erf

    x = np.arange(width, dtype=np.float64)
    profile = low + (high - low) * 0.5 * (1.0 + erf((x - edge_x) / (np.sqrt(2.0) * sigma)))
    return np.repeat(profile[None, :], height, axis=0)


def blurred_disk(
    *,
    size: int = 128,
    centre: tuple[float, float] = (63.5, 63.5),
    radius: float = 24.0,
    sigma: float = 1.5,
    low: float = 0.2,
    high: float = 0.8,
) -> np.ndarray:
    """A bright disk of known radius with an analytically blurred edge."""
    from scipy.special import erf

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    distance = np.hypot(yy - centre[0], xx - centre[1])
    # Signed distance outward; the edge profile is the same erf as the step.
    return low + (high - low) * 0.5 * (1.0 + erf((radius - distance) / (np.sqrt(2.0) * sigma)))


def write_png(path: Path, image01: np.ndarray, *, bits: int = 8) -> Path:
    """Write a [0, 1] array as an 8- or 16-bit grayscale PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(image01, dtype=np.float64), 0.0, 1.0)
    if bits == 8:
        Image.fromarray(np.round(clipped * 255.0).astype(np.uint8), mode="L").save(path)
    elif bits == 16:
        Image.fromarray(np.round(clipped * 65535.0).astype(np.uint16)).save(path)
    else:
        raise ValueError(f"bits must be 8 or 16, got {bits}")
    return path


def write_rgb_with_border(path: Path, image01: np.ndarray, *, border: int = 2) -> Path:
    """An RGB image that is grayscale only inside ``border``.

    Reproduces the real failure mode of instrument exports: a colored frame or
    databar around an otherwise-grayscale imaging area.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    gray = np.round(np.clip(image01, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    rgb[:border, :, 1] = 255
    rgb[-border:, :, 1] = 255
    rgb[:, :border, 1] = 255
    rgb[:, -border:, 1] = 255
    Image.fromarray(rgb, mode="RGB").save(path)
    return path


def make_config(**overrides) -> Config:
    """A CPU-only config using the no-download backend, with nested overrides."""
    raw: dict = {
        "segmentation": {"backend": "classical", "contrast_stretch": None},
        "report": {"enabled": False},
    }
    for section, values in overrides.items():
        if isinstance(values, dict):
            raw.setdefault(section, {}).update(values)
        else:
            raw[section] = values
    return Config.model_validate(raw)
