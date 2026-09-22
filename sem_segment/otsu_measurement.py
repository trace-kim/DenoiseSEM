"""Adapt saved-uint8 Otsu masks to the existing contour measurement result.

The detector stays identical to the standalone baseline. Only fully contained
regions feed the existing polygon-area measurements; open paths are retained
separately for display. No edge refinement or additional mask cleanup runs.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
import time

import numpy as np

from .backends import InstanceMask
from .config import MetrologyConfig
from .contours import Contour, polygon_area
from .metrology import measure_region, summarise_image
from .otsu_baseline import OtsuResult, OtsuSettings, otsu_baseline
from .pipeline import Diagnostics, SegmentationResult


def measure_saved_otsu(
    path: str | Path,
    settings: OtsuSettings | None = None,
    *,
    crop: tuple[int, int, int, int] | None = None,
    device: str = "cpu",
    metrology: MetrologyConfig | None = None,
    memory_mb: int = 8192,
    io_workers: int = 2,
) -> SegmentationResult:
    """Measure closed baseline masks using the existing area/ECD definitions.

    Returned geometry is relative to the measurement crop, like segment_image.
    The caller restores the crop offset once when exporting full-image contours.
    """
    settings = settings or OtsuSettings()
    metrology = metrology or MetrologyConfig()
    result = otsu_baseline(path, settings, crop=crop, device=device, memory_mb=memory_mb, io_workers=io_workers)
    return measure_otsu_result(result, settings, crop=crop, device=device, metrology=metrology)


def iter_measure_saved_otsu(paths: Sequence[Path], settings: OtsuSettings, *,
                            crop: tuple[int, int, int, int] | None = None,
                            device: str = "cpu", metrology: MetrologyConfig | None = None,
                            batch_size: int = 16, memory_mb: int = 8192,
                            io_workers: int = 2) -> Iterator[SegmentationResult]:
    """Stream measurements while the CUDA producer processes the next batch."""
    if device == "cpu":
        for path in paths:
            yield measure_saved_otsu(path, settings, crop=crop, device=device, metrology=metrology)
        return
    from .otsu_cuda import iter_saved_otsu

    results = iter_saved_otsu(paths, settings, crop=crop, device=device,
                              batch_size=batch_size, memory_mb=memory_mb, io_workers=io_workers)
    try:
        for result in results:
            yield measure_otsu_result(result, settings, crop=crop, device=device, metrology=metrology)
            del result
    finally:
        results.close()


def measure_otsu_result(result: OtsuResult, settings: OtsuSettings, *,
                        crop: tuple[int, int, int, int] | None = None,
                        device: str = "cpu", metrology: MetrologyConfig | None = None) -> SegmentationResult:
    """Adapt a detected label map without decoding or labeling it a second time."""
    from skimage.measure import regionprops

    metrology = metrology or MetrologyConfig()
    started = time.perf_counter()
    labels = result.labels
    if labels is None:
        raise ValueError("Otsu measurement requires the detector's retained label map")
    grouped = defaultdict(list)
    offset = np.array([crop[0], crop[2]]) if crop else np.zeros(2)
    for full_path in result.outlines:
        points = full_path - offset
        # A binary marching-squares vertex lies halfway along a grid edge.
        # Its floor/ceil pixel neighbours contain exactly one foreground label,
        # including at diagonal ambiguities and at the measurement border.
        low, high = np.floor(points[0]).astype(int), np.ceil(points[0]).astype(int)
        region_id = int(labels[low[0]:high[0] + 1, low[1]:high[1] + 1].max())
        grouped[region_id].append(points)

    instances, coarse, holes, open_paths, regions = [], [], [], [], []
    # Preserve deterministic spatial region IDs, as in the existing pipeline.
    properties = sorted(regionprops(labels), key=lambda p: tuple(p.centroid))
    for region_id, prop in enumerate(properties, start=1):
        y0, x0, y1, x1 = prop.bbox
        instance = InstanceMask((y0, y1, x0, x1), prop.image, 1.0, "otsu",
                                prompt=f"otsu:{settings.polarity}")
        border = instance.touches_border(result.mask.shape)
        paths = grouped[prop.label]
        closed = [p[:-1] for p in paths if np.array_equal(p[0], p[-1])]
        opened = [p for p in paths if not np.array_equal(p[0], p[-1])]
        outer = np.empty((0, 2))
        if closed and not border:
            outer_index = int(np.argmax([abs(polygon_area(p)) for p in closed]))
            outer = closed.pop(outer_index)
        # A border region has no complete outer polygon. Its closed loops are
        # interior holes; never mistake one for an independently measured object.
        ring = Contour(outer, region_id=region_id)
        inner = [Contour(p, region_id=region_id, is_hole=True) for p in closed]
        instance.meta.update(touches_border=border, n_holes=len(inner),
                             hole_area_px=sum(abs(polygon_area(h.points)) for h in inner))
        instances.append(instance)
        coarse.append(ring)
        holes.append(inner)
        open_paths.append(opened)
        regions.append(measure_region(region_id, instance, ring, inner, None, metrology))

    timings = dict(result.timings_s)
    timings["mask_metrology"] = time.perf_counter() - started
    timings["total"] += timings["mask_metrology"]
    return SegmentationResult(
        shape=result.mask.shape, instances=instances, coarse=coarse, holes=holes,
        refined=[], regions=regions, open_paths=open_paths,
        image_stats=summarise_image(regions, result.mask.shape, metrology, include_border_regions=False),
        diagnostics=Diagnostics(
            backend={"backend": "otsu", "algorithm": "gaussian + otsu + four-connected components",
                     "settings": settings.model_dump(), "threshold_dn": result.threshold_dn,
                     "gaussian_backend": result.gaussian_backend, "device": device,
                     "component_count": result.component_count, "retained_count": result.retained_count,
                     "execution": result.execution},
            timings_s=timings, refinement={"backend": "disabled", "device": None},
        ),
    )
