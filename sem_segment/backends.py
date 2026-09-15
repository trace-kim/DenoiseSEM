"""The pluggable segmentation seam.

A backend's only job is to turn an image into a list of :class:`InstanceMask`.
Everything downstream - contours, refinement, metrology, reporting - is written
against that one type, so swapping SAM 3 for a classical threshold, or for
whatever replaces SAM 3, touches nothing but this module and its adapters.

That indirection is not theoretical tidiness.  ``facebook/sam3`` is a gated
repository, and the machine this pipeline is meant to run on may have no
internet access at all, so a no-download path has to be a first-class citizen
rather than a fallback bolted on later.

Masks are stored as a bounding box plus a cropped boolean array.  A few hundred
full-frame boolean masks on a 2K image is around a gigabyte of mostly-``False``;
the crop representation also lets the deduplication in ``masks.py`` compute
intersections only where two boxes actually overlap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from .config import Config, SegmentationConfig


@dataclass(frozen=True)
class InstanceMask:
    """One segmented feature, stored as a bounding box plus a cropped mask."""

    #: (y0, y1, x0, x1), half-open, in full-image coordinates.
    bbox: tuple[int, int, int, int]
    #: Boolean array of shape (y1 - y0, x1 - x0).
    crop: np.ndarray
    score: float
    backend: str
    prompt: str | None = None
    #: Which tile produced this instance, when tiling was used.
    source: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def area(self) -> int:
        return int(np.count_nonzero(self.crop))

    @property
    def bbox_shape(self) -> tuple[int, int]:
        y0, y1, x0, x1 = self.bbox
        return (y1 - y0, x1 - x0)

    @property
    def centroid(self) -> tuple[float, float]:
        """Pixel-mass centroid in full-image (y, x) coordinates."""
        ys, xs = np.nonzero(self.crop)
        y0, _, x0, _ = self.bbox
        return (float(ys.mean()) + y0, float(xs.mean()) + x0)

    def full_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Expand to a full-frame boolean mask."""
        mask = np.zeros(shape, dtype=bool)
        y0, y1, x0, x1 = self.bbox
        mask[y0:y1, x0:x1] = self.crop
        return mask

    def touches_border(self, shape: tuple[int, int]) -> bool:
        y0, y1, x0, x1 = self.bbox
        return y0 <= 0 or x0 <= 0 or y1 >= shape[0] or x1 >= shape[1]

    def translated(self, dy: int, dx: int) -> "InstanceMask":
        """Move into another coordinate frame, for mapping a tile to the full image."""
        y0, y1, x0, x1 = self.bbox
        return InstanceMask(
            bbox=(y0 + dy, y1 + dy, x0 + dx, x1 + dx),
            crop=self.crop,
            score=self.score,
            backend=self.backend,
            prompt=self.prompt,
            source=self.source,
            meta=dict(self.meta),
        )

    def with_crop(self, crop: np.ndarray) -> "InstanceMask":
        """Replace the cropped mask, keeping the bounding box and provenance."""
        return InstanceMask(
            bbox=self.bbox,
            crop=np.asarray(crop, dtype=bool),
            score=self.score,
            backend=self.backend,
            prompt=self.prompt,
            source=self.source,
            meta=dict(self.meta),
        )

    @classmethod
    def from_mask(
        cls,
        mask: np.ndarray,
        *,
        score: float,
        backend: str,
        prompt: str | None = None,
        source: str | None = None,
        meta: dict | None = None,
    ) -> "InstanceMask | None":
        """Build from a full-frame boolean mask, or ``None`` when it is empty."""
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
        rows = np.flatnonzero(mask.any(axis=1))
        if rows.size == 0:
            return None
        cols = np.flatnonzero(mask.any(axis=0))
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        x0, x1 = int(cols[0]), int(cols[-1]) + 1
        return cls(
            bbox=(y0, y1, x0, x1),
            crop=np.ascontiguousarray(mask[y0:y1, x0:x1]),
            score=float(score),
            backend=backend,
            prompt=prompt,
            source=source,
            meta=dict(meta or {}),
        )


@runtime_checkable
class Segmenter(Protocol):
    """What every backend must provide."""

    name: str

    def segment(self, model_rgb: np.ndarray) -> list[InstanceMask]:
        """Segment a uint8 RGB image into instance masks in image coordinates."""
        ...

    def describe(self) -> dict:
        """Provenance: model identity, resolution, and the knobs that were used."""
        ...


class BackendUnavailable(RuntimeError):
    """A backend cannot run here, with an actionable explanation."""


#: Populated by the adapter modules at import time.
BACKENDS: dict[str, Callable[[SegmentationConfig], Segmenter]] = {}


def register_backend(name: str, factory: Callable[[SegmentationConfig], Segmenter]) -> None:
    if name in BACKENDS:
        raise ValueError(f"backend {name!r} is already registered")
    BACKENDS[name] = factory


def build_segmenter(config: Config | SegmentationConfig) -> Segmenter:
    """Construct the backend named by the config.

    Imports are deferred to here so that ``--help``, config validation and the
    whole classical path never import torch or transformers.
    """
    segmentation = config.segmentation if isinstance(config, Config) else config
    name = segmentation.backend
    if name == "classical":
        from . import backend_classical  # noqa: F401  (registers on import)
    elif name.startswith("sam3"):
        from . import backend_sam3  # noqa: F401  (registers on import)
    factory = BACKENDS.get(name)
    if factory is None:
        raise BackendUnavailable(
            f"unknown backend {name!r}; available: {sorted(BACKENDS) or ['classical']}"
        )
    return factory(segmentation)


def available_backends() -> list[str]:
    """Names the registry knows about, importing the adapters to find out."""
    from . import backend_classical  # noqa: F401

    try:
        from . import backend_sam3  # noqa: F401
    except Exception:  # pragma: no cover - only when transformers is broken
        pass
    return sorted(BACKENDS)
