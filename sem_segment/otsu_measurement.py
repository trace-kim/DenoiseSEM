"""Adapt saved-uint8 Otsu masks to the existing contour measurement result.

The detector stays identical to the standalone baseline. Only fully contained
regions feed the existing polygon-area measurements; open paths are retained
separately for display. Optional refinement samples the original uint8 crop.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import ExitStack
from pathlib import Path
import time

import numpy as np

from .backends import InstanceMask
from .config import ContoursConfig, MetrologyConfig, RefineConfig
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
    refine: RefineConfig | None = None,
    contours: ContoursConfig | None = None,
) -> SegmentationResult:
    """Measure closed baseline masks using the existing area/ECD definitions.

    Returned geometry is relative to the measurement crop, like segment_image.
    The caller restores the crop offset once when exporting full-image contours.
    """
    settings = settings or OtsuSettings()
    metrology = metrology or MetrologyConfig()
    result = otsu_baseline(path, settings, crop=crop, device=device, memory_mb=memory_mb, io_workers=io_workers)
    return measure_otsu_result(result, settings, crop=crop, device=device, metrology=metrology,
                               refine=refine, contours=contours)


def iter_measure_saved_otsu(paths: Sequence[Path], settings: OtsuSettings, *,
                            crop: tuple[int, int, int, int] | None = None,
                            device: str = "cpu", metrology: MetrologyConfig | None = None,
                            batch_size: int = 16, memory_mb: int = 8192,
                            io_workers: int = 2, refine: RefineConfig | None = None,
                            contours: ContoursConfig | None = None) -> Iterator[SegmentationResult]:
    """Stream measurements while the CUDA producer processes the next batch."""
    if device == "cpu":
        for path in paths:
            yield measure_saved_otsu(path, settings, crop=crop, device=device, metrology=metrology,
                                     refine=refine, contours=contours)
        return
    from .otsu_cuda import iter_saved_otsu

    results = iter_saved_otsu(paths, settings, crop=crop, device=device,
                              batch_size=batch_size, memory_mb=memory_mb, io_workers=io_workers)
    with ExitStack() as stack:
        stack.callback(results.close)
        refiner = None
        if refine is not None:
            from .cuda import CudaRefiner

            refiner = stack.enter_context(CudaRefiner(refine))
        for result in results:
            yield measure_otsu_result(result, settings, crop=crop, device=device, metrology=metrology,
                                       refine=refine, contours=contours, refiner=refiner)
            del result


def measure_otsu_result(result: OtsuResult, settings: OtsuSettings, *,
                        crop: tuple[int, int, int, int] | None = None,
                        device: str = "cpu", metrology: MetrologyConfig | None = None,
                        refine: RefineConfig | None = None, contours: ContoursConfig | None = None,
                        refiner=None) -> SegmentationResult:
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
    timings = dict(result.timings_s)
    refined = []
    refinement = {"backend": "disabled", "device": None}
    if refine is not None:
        from .contours import attach_normals, resample_closed
        from .refine import adaptive_search_px, refine_all

        if not refine.enabled or refine.device != device:
            raise ValueError("Otsu refinement must be enabled on the requested metrology device")
        if result.pixels is None or result.pixels.dtype != np.uint8:
            raise ValueError("Otsu refinement requires the decoded unsmoothed uint8 crop")
        contours = contours or ContoursConfig()
        bases = []
        for ring, instance in zip(coarse, instances):
            base = Contour(resample_closed(ring.points, contours.spacing_px) if len(ring) else ring.points,
                           region_id=ring.region_id)
            if len(base):
                attach_normals(base, instance.full_mask(result.mask.shape), contours)
            else:
                base.normals = np.empty((0, 2))
            bases.append(base)
        radii = [adaptive_search_px(m.crop, refine) for m in instances]
        image01 = result.pixels.astype(np.float64) / 255
        stage = time.perf_counter()
        if device == "cpu":
            refined = refine_all(bases, image01, refine, spacing_px=contours.spacing_px, search_px=radii)
            refinement = {"backend": "scipy", "device": "cpu", "estimator": refine.estimator}
            timings["refine"] = time.perf_counter() - stage
        else:
            from .cuda import CudaRefiner

            with ExitStack() as stack:
                owned = refiner if refiner is not None else stack.enter_context(CudaRefiner(refine))
                refined, _, gpu_times = owned.measure(bases, image01, spacing_px=contours.spacing_px, search_px=radii)
                refinement = {**owned.describe(), "estimator": refine.estimator}
                timings.update(gpu_times)
    for i, (instance, ring, inner) in enumerate(zip(instances, coarse, holes)):
        regions.append(measure_region(i + 1, instance, ring, inner, refined[i] if refined else None, metrology,
                                       spacing_px=(contours or ContoursConfig()).spacing_px))
    timings["mask_metrology"] = time.perf_counter() - started
    timings["mask_metrology"] -= timings.get("refine", 0) + timings.get("edge_strength", 0)
    timings["total"] += time.perf_counter() - started
    return SegmentationResult(
        shape=result.mask.shape, instances=instances, coarse=coarse, holes=holes,
        refined=refined, regions=regions, open_paths=open_paths,
        image_stats=summarise_image(regions, result.mask.shape, metrology, include_border_regions=False),
        diagnostics=Diagnostics(
            backend={"backend": "otsu", "algorithm": "gaussian + otsu + four-connected components",
                     "settings": settings.model_dump(), "threshold_dn": result.threshold_dn,
                     "gaussian_backend": result.gaussian_backend, "device": device,
                     "component_count": result.component_count, "retained_count": result.retained_count,
                     "execution": result.execution},
            timings_s=timings, refinement=refinement,
        ),
    )
