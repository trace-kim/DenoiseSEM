"""A saved-uint8 segmentation baseline, without refinement or metrology."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .config import InputConfig
from .image_io import read_native


class OtsuSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    polarity: Literal["dark", "bright"] = "dark"
    sigma_px: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    min_area_px: int = Field(default=25, ge=1, strict=True)


@dataclass
class OtsuResult:
    mask: np.ndarray
    #: Full-image (y, x) coordinates. Closed loops repeat their first vertex;
    #: paths intersecting the measurement border remain open.
    outlines: list[np.ndarray]
    threshold_dn: float
    component_count: int
    retained_count: int
    gaussian_backend: str
    timings_s: dict[str, float]


def otsu_baseline(
    path: str | Path,
    settings: OtsuSettings | None = None,
    *,
    crop: tuple[int, int, int, int] | None = None,
    device: str = "cpu",
) -> OtsuResult:
    """Decode saved uint8 pixels, smooth a copy, threshold, filter, and outline.

    Otsu uses scikit-image's default histogram, independently per image, in DN.
    Dark foreground is <= threshold; bright is > threshold. Constant images
    have no separable foreground and return an empty mask with their constant
    threshold. The returned mask covers the measurement crop, if present.

    Foreground is four-connected both when labeling and tracing. No padding,
    hole filling, splitting, resampling, or coordinate refinement is performed.
    Interpolated binary-mask outlines are not subpixel edge measurements.
    """
    started = time.perf_counter()
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import find_contours

    settings = settings or OtsuSettings()
    if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device):
        raise ValueError("device must be cpu, cuda, or cuda:N")
    crop = InputConfig(crop=crop).crop
    stage = time.perf_counter()
    pixels, _ = read_native(path, crop=crop)
    if pixels.dtype != np.uint8:
        raise ValueError(f"Otsu baseline requires saved uint8 pixels: {path}")
    timings = {"decode": time.perf_counter() - stage}

    stage = time.perf_counter()
    working = pixels.astype(np.float64)
    backend = "disabled"
    if settings.sigma_px > 0:
        if device == "cpu":
            working = ndimage.gaussian_filter(working, settings.sigma_px, mode="reflect")
            backend = "scipy"
        else:
            try:
                import cupy as cp
                from cupyx.scipy.ndimage import gaussian_filter
            except ImportError as error:
                raise RuntimeError(
                    "CUDA smoothing requires CuPy; see sem_segment/README.md "
                    "for installation, or use --device cpu."
                ) from error
            index = int(device.split(":")[1]) if ":" in device else 0
            try:
                # Device indices are relative to scheduler-provided visibility.
                with cp.cuda.Device(index):
                    working = cp.asnumpy(gaussian_filter(
                        cp.asarray(working), settings.sigma_px, mode="reflect"))
                # asnumpy is blocking: timing includes transfers and completion.
            except Exception as error:
                raise RuntimeError(f"CUDA smoothing failed on {device}: {error}") from error
            backend = "cupy"
    timings["gaussian"] = time.perf_counter() - stage

    stage = time.perf_counter()
    threshold = float(threshold_otsu(working))
    foreground = working <= threshold if settings.polarity == "dark" else working > threshold
    if pixels.min() == pixels.max():
        foreground[:] = False
    timings["otsu"] = time.perf_counter() - stage

    stage = time.perf_counter()
    labels, count = ndimage.label(foreground, structure=ndimage.generate_binary_structure(2, 1))
    keep = np.bincount(labels.ravel()) >= settings.min_area_px
    keep[0] = False
    mask = keep[labels]
    timings["components"] = time.perf_counter() - stage

    stage = time.perf_counter()
    # High values are foreground; fully_connected='low' keeps high values
    # four-connected at ambiguous diagonal crossings. Do not pad the mask.
    outlines = find_contours(mask, level=0.5, fully_connected="low")
    if crop is not None:
        offset = np.array([crop[0], crop[2]])
        outlines = [outline + offset for outline in outlines]
    timings["outlines"] = time.perf_counter() - stage
    timings["total"] = time.perf_counter() - started
    return OtsuResult(mask, outlines, threshold, int(count), int(keep.sum()), backend, timings)
