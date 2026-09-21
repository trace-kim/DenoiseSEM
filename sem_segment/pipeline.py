"""The public API: one image in, every measurement out.

``segment_image`` is a pure function.  It takes an array, touches no filesystem,
downloads nothing implicitly, and returns everything it computed.  Persistence
lives in :mod:`sem_segment.writers` and rendering in :mod:`sem_segment.report`,
each callable on its own, so this package can be driven from a notebook, a
wrapper script, or the CLI without any of them inheriting assumptions from the
others.

The load-bearing invariant of the whole package lives here.  Two arrays are kept
side by side:

``measure01``
    Exactly as loaded.  Never resized, stretched, filtered, or denoised.  Every
    reported edge position, area and roughness comes from this array.

``model_rgb``
    What the segmentation backend sees.  May be contrast-stretched, and may be
    downsampled to the model's native resolution.

Because refinement measures against ``measure01``, a transformation applied to
help the model cannot move a reported measurement.  That is what makes it safe
to stretch a low-contrast frame for SAM 3's benefit, and it is why resizing an
oversized frame costs so little: the mask only has to land within a few pixels
of the true edge for the refinement to recover full-resolution precision.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .backends import InstanceMask, Segmenter, build_segmenter
from .config import SAM3_NATIVE_PX, Config
from .contours import Contour, trace_all
from .image_io import to_model_rgb
from .masks import label_map, postprocess
from .metrology import RegionMetrology, measure_region, summarise_image
from .refine import RefinedContour, refine_all


@dataclass
class Diagnostics:
    """Everything needed to explain a result, including what it left out."""

    backend: dict = field(default_factory=dict)
    rejections: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    timings_s: dict = field(default_factory=dict)
    scaling: dict = field(default_factory=dict)
    refine_rejections: dict = field(default_factory=dict)
    valid_fraction: float = float("nan")
    #: Per-region change in mean |grad I| along the ring, coarse -> refined.
    edge_strength_change: dict = field(default_factory=dict)


@dataclass
class SegmentationResult:
    """The complete outcome of measuring one image."""

    shape: tuple[int, int]
    instances: list[InstanceMask]
    coarse: list[Contour]
    holes: list[list[Contour]]
    refined: list[RefinedContour]
    regions: list[RegionMetrology]
    image_stats: dict
    diagnostics: Diagnostics
    provenance: dict = field(default_factory=dict)

    @property
    def region_count(self) -> int:
        return len(self.regions)

    def label_map(self) -> np.ndarray:
        return label_map(self.instances, self.shape)

    def rows(self, *, pixel_size_nm: float | None = None) -> list[dict]:
        return [r.to_row(pixel_size_nm=pixel_size_nm) for r in self.regions]


def _resize_for_model(model_rgb: np.ndarray, target_px: int) -> tuple[np.ndarray, float]:
    from skimage.transform import resize

    height, width = model_rgb.shape[:2]
    scale = target_px / float(max(height, width))
    new_shape = (max(1, int(round(height * scale))), max(1, int(round(width * scale))))
    resized = resize(model_rgb, new_shape, order=1, preserve_range=True, anti_aliasing=True)
    return resized.astype(np.uint8), scale


def _upscale_instance(
    instance: InstanceMask, small_shape: tuple[int, int], full_shape: tuple[int, int]
) -> InstanceMask | None:
    """Map a mask found at reduced resolution back onto the full frame."""
    from skimage.transform import resize

    small = instance.full_mask(small_shape).astype(np.float32)
    grown = resize(small, full_shape, order=1, preserve_range=True, anti_aliasing=False) > 0.5
    return InstanceMask.from_mask(
        grown,
        score=instance.score,
        backend=instance.backend,
        prompt=instance.prompt,
        source=instance.source,
        meta=dict(instance.meta),
    )


def _tile_origins(extent: int, tile: int, stride: int) -> list[int]:
    """Tile starts, with the last one clamped rather than the frame padded.

    Padding would fabricate a hard border that the model happily segments as a
    feature; clamping merely gives the last tile more overlap, which is harmless.
    """
    if extent <= tile:
        return [0]
    stops = list(range(0, extent - tile + 1, stride))
    if stops[-1] != extent - tile:
        stops.append(extent - tile)
    return stops


def _segment_tiled(
    segmenter, model_rgb: np.ndarray, tile: int, overlap: int
) -> tuple[list[InstanceMask], dict]:
    """Segment in overlapping tiles, keeping only whole features.

    Masks are never blended at a seam.  Averaging two binary masks and
    re-thresholding produces a boundary that is a mixture of two different model
    opinions - fabricated geometry, which a metrology package must not emit.
    Instead each tile's instances are kept or discarded whole: anything touching
    a tile's *interior* border is truncated by the tile and is strictly worse
    than the neighbouring tile's view of the same feature, so it is dropped.

    That rule is only safe when the overlap exceeds the largest feature, and a
    feature wider than the overlap is truncated in every tile and lost
    completely.  Those losses are counted and surfaced rather than absorbed.
    """
    height, width = model_rgb.shape[:2]
    stride = tile - overlap
    kept: list[InstanceMask] = []
    truncated = 0

    for y0 in _tile_origins(height, min(tile, height), stride):
        for x0 in _tile_origins(width, min(tile, width), stride):
            y1 = min(y0 + tile, height)
            x1 = min(x0 + tile, width)
            patch = model_rgb[y0:y1, x0:x1]
            tile_shape = patch.shape[:2]
            for instance in segmenter.segment(patch):
                by0, by1, bx0, bx1 = instance.bbox
                touches_interior = (
                    (by0 <= 0 and y0 > 0)
                    or (bx0 <= 0 and x0 > 0)
                    or (by1 >= tile_shape[0] and y1 < height)
                    or (bx1 >= tile_shape[1] and x1 < width)
                )
                if touches_interior:
                    truncated += 1
                    continue
                moved = instance.translated(y0, x0)
                moved.meta["tile"] = f"{y0}_{x0}"
                # Distance from the tile's own border decides ties later: the
                # most interior observation is the least edge-affected one.
                centre = moved.centroid
                moved.meta["tile_margin"] = float(
                    min(centre[0] - y0, y1 - centre[0], centre[1] - x0, x1 - centre[1])
                )
                kept.append(moved)
    return kept, {"tiles_truncated_instances": truncated}


def _plan_scaling(shape: tuple[int, int], config: Config) -> tuple[str, dict, list[str]]:
    """Decide how an oversized frame reaches the model."""
    large = config.segmentation.large_frame
    longest = max(shape)
    warnings: list[str] = []
    info = {"mode": large.mode, "input_longest_px": longest, "model_native_px": SAM3_NATIVE_PX}

    if not config.needs_model_weights or longest <= SAM3_NATIVE_PX or large.mode == "none":
        info["applied"] = "none"
        return "none", info, warnings

    if large.mode == "resize":
        scale = SAM3_NATIVE_PX / float(longest)
        info["applied"] = "resize"
        info["scale"] = scale
        warnings.append(
            f"Frame is {longest} px across and the model runs at {SAM3_NATIVE_PX} px; it was "
            f"resized by {scale:.3f} for segmentation only. Edge positions were still measured "
            "on the original pixels, so precision is unaffected, but a feature smaller than "
            f"{large.min_feature_px_after_resize / scale:.0f} px in the original may not be "
            "detected. Set segmentation.large_frame.mode to 'tile' if features are being missed."
        )
        return "resize", info, warnings

    info["applied"] = "tile"
    info["tile_px"] = large.tile_px
    info["overlap_px"] = large.overlap_px
    warnings.append(
        f"Tiled segmentation at {large.tile_px} px with {large.overlap_px} px overlap. A feature "
        f"wider than the overlap is truncated in every tile and will be missed entirely; check "
        "tiles_truncated_instances in the diagnostics."
    )
    return "tile", info, warnings


def segment_image(image01: np.ndarray, config: Config, *, segmenter: Segmenter | None = None) -> SegmentationResult:
    """Segment, contour, refine and measure one image.

    ``image01`` is a float array in [0, 1] as produced by
    :func:`sem_segment.image_io.load_image`.  It is the measurement array and is
    never modified.
    A caller processing a series may supply a backend built from this config
    to keep its model loaded; all image-specific measurements are recomputed.
    """
    total_started = time.perf_counter()
    measure01 = np.asarray(image01, dtype=np.float64)
    if measure01.ndim != 2:
        raise ValueError(f"expected a 2-D image in [0, 1], got shape {measure01.shape}")
    shape = (int(measure01.shape[0]), int(measure01.shape[1]))
    timings: dict = {}
    warnings: list[str] = []

    model_rgb = to_model_rgb(measure01, contrast_stretch=config.segmentation.contrast_stretch)
    mode, scaling, scale_warnings = _plan_scaling(shape, config)
    warnings.extend(scale_warnings)

    if segmenter is None:
        segmenter = build_segmenter(config)
    timings["prepare"] = time.perf_counter() - total_started

    start = time.perf_counter()
    extra: dict = {}
    if mode == "resize":
        small_rgb, _ = _resize_for_model(model_rgb, SAM3_NATIVE_PX)
        small_shape = small_rgb.shape[:2]
        raw = segmenter.segment(small_rgb)
        raw = [m for m in (_upscale_instance(i, small_shape, shape) for i in raw) if m is not None]
    elif mode == "tile":
        raw, extra = _segment_tiled(
            segmenter,
            model_rgb,
            config.segmentation.large_frame.tile_px,
            config.segmentation.large_frame.overlap_px,
        )
    else:
        raw = segmenter.segment(model_rgb)
    timings["segment"] = time.perf_counter() - start

    raw = [m for m in raw if m.score >= config.segmentation.score_threshold]

    start = time.perf_counter()
    instances, tally = postprocess(raw, shape, config.masks)
    instances = instances[: config.segmentation.max_instances]
    timings["masks"] = time.perf_counter() - start

    start = time.perf_counter()
    coarse, holes = trace_all(instances, shape, config.contours)
    timings["contours"] = time.perf_counter() - start

    refined: list[RefinedContour] = []
    if config.refine.enabled:
        start = time.perf_counter()
        from .refine import adaptive_search_px

        radii = [adaptive_search_px(m.crop, config.refine) for m in instances]
        refined = refine_all(
            coarse,
            measure01,
            config.refine,
            spacing_px=config.contours.spacing_px,
            search_px=radii,
        )
        timings["refine"] = time.perf_counter() - start

    start = time.perf_counter()
    regions: list[RegionMetrology] = []
    for index, instance in enumerate(instances):
        regions.append(
            measure_region(
                index + 1,
                instance,
                coarse[index],
                holes[index],
                refined[index] if index < len(refined) else None,
                config.metrology,
                spacing_px=config.contours.spacing_px,
            )
        )
    image_stats = summarise_image(
        regions,
        shape,
        config.metrology,
        include_border_regions=config.masks.include_border_regions,
        pixel_size_nm=config.input.pixel_size_nm,
    )
    timings["metrology"] = time.perf_counter() - start

    # Did refinement actually improve each boundary? Without ground truth the
    # honest check is whether the ring ends up on a stronger gradient. A contour
    # dragged onto a neighbour or into flat background lands on a weaker one, and
    # a summary that only reports what was fixed would never show it.
    strength: dict = {}
    start = time.perf_counter()
    if refined:
        from scipy.ndimage import gaussian_gradient_magnitude
        from .refine import edge_strength_along

        magnitude = gaussian_gradient_magnitude(measure01, 1.0)
        changes = []
        for base, ref in zip(coarse, refined):
            before = edge_strength_along(base.points, measure01, magnitude=magnitude)
            after = edge_strength_along(ref.polygon, measure01, magnitude=magnitude)
            if np.isfinite(before) and np.isfinite(after) and before > 0:
                changes.append((after - before) / before)
        if changes:
            values = np.asarray(changes)
            strength = {
                "regions_improved": int((values > 0.02).sum()),
                "regions_unchanged": int((np.abs(values) <= 0.02).sum()),
                "regions_degraded": int((values < -0.02).sum()),
                "median_change": float(np.median(values)),
                "worst_change": float(values.min()),
            }
    timings["edge_strength"] = time.perf_counter() - start

    refine_rejections: dict = {}
    for contour in refined:
        for reason, count in contour.reason_counts().items():
            refine_rejections[reason] = refine_rejections.get(reason, 0) + count
    valid_fraction = (
        float(np.mean([c.valid_fraction for c in refined])) if refined else float("nan")
    )

    if config.refine.enabled and refined and valid_fraction < config.refine.min_valid_fraction:
        warnings.append(
            f"Only {valid_fraction:.0%} of contour vertices produced a usable edge fit "
            f"(threshold {config.refine.min_valid_fraction:.0%}). Refined geometry is based on "
            "the surviving vertices; see refine_rejections for why the rest failed."
        )
    if strength.get("regions_degraded"):
        warnings.append(
            f"Refinement moved {strength['regions_degraded']} region(s) onto a WEAKER edge than "
            f"the segmentation boundary they started from (worst {strength['worst_change']:.1%}). "
            "Those contours are worse than the mask they came from and should not be trusted; "
            "inspect them before using their numbers."
        )
    if not regions:
        warnings.append(
            "No regions survived segmentation and filtering. Check masks.min_area_px and "
            "segmentation.score_threshold, or try a different backend."
        )
    border = sum(1 for r in regions if r.touches_border)
    if border and not config.masks.include_border_regions:
        warnings.append(
            f"{border} of {len(regions)} regions touch the image border. They are measured and "
            "listed individually but excluded from image-level aggregates, because their true "
            "extent is unknown."
        )

    timings["total"] = time.perf_counter() - total_started
    diagnostics = Diagnostics(
        backend=segmenter.describe(),
        rejections={**tally.as_dict(), **extra},
        warnings=warnings,
        timings_s={k: round(v, 4) for k, v in timings.items()},
        scaling=scaling,
        refine_rejections=refine_rejections,
        valid_fraction=valid_fraction,
        edge_strength_change=strength,
    )
    return SegmentationResult(
        shape=shape,
        instances=instances,
        coarse=coarse,
        holes=holes,
        refined=refined,
        regions=regions,
        image_stats=image_stats,
        diagnostics=diagnostics,
        provenance={
            "schema_version": 1,
            "config": config.model_dump(mode="json"),
            "backend": segmenter.describe(),
        },
    )
