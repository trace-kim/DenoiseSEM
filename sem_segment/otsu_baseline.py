"""A saved-uint8 segmentation baseline, without refinement or metrology."""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import closing
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
    labels: np.ndarray | None = None
    execution: dict = field(default_factory=dict)


def otsu_baseline(
    path: str | Path,
    settings: OtsuSettings | None = None,
    *,
    crop: tuple[int, int, int, int] | None = None,
    device: str = "cpu",
    memory_mb: int = 8192,
    io_workers: int = 2,
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
    if device != "cpu":
        # The CUDA implementation keeps smoothing, thresholding and component
        # filtering on device. The public single-image API shares that path.
        from .otsu_cuda import iter_saved_otsu

        with closing(iter_saved_otsu([Path(path)], settings, crop=crop, device=device, batch_size=1,
                                    memory_mb=memory_mb, io_workers=io_workers)) as results:
            return next(results)
    stage = time.perf_counter()
    pixels, _ = read_native(path, crop=crop)
    if pixels.dtype != np.uint8:
        raise ValueError(f"Otsu baseline requires saved uint8 pixels: {path}")
    timings = {"decode": time.perf_counter() - stage}

    stage = time.perf_counter()
    working = pixels.astype(np.float64)
    backend = "disabled"
    if settings.sigma_px > 0:
        working = ndimage.gaussian_filter(working, settings.sigma_px, mode="reflect")
        backend = "scipy"
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
    remap = np.cumsum(keep, dtype=np.int32)
    remap[~keep] = 0
    labels = remap[labels]
    mask = labels > 0
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
    return OtsuResult(mask, outlines, threshold, int(count), int(keep.sum()), backend, timings,
                      labels=labels, execution={"backend": "scipy", "batch_size": 1})
