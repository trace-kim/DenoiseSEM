"""Turning raw backend output into a set of measurable regions.

Three of the policies here are deliberately not the usual shortcut, because the
usual shortcut quietly corrupts measurements:

*Containment is checked separately from IoU.*  A mask nested inside another has
a **low** IoU with it, so IoU-only deduplication keeps both and every feature is
measured twice.  ``sam3_auto`` produces exactly this pattern - "the hole" and
"the hole's dark core" - as a matter of course.

*The larger of two nested masks wins by default.*  This is the opposite of the
usual detection convention of keeping the higher-scoring mask, and it is the
right choice here: the tight inner core frequently scores higher, but the
feature is the outer boundary.

*Holes are filled only when small.*  Always-filling silently turns a genuine
annulus - a ring, a via seen through a dielectric - into a disk.  Speckle holes
are filled; larger ones are kept and counted.

Nothing is dropped silently.  Every rejection is tallied by reason and travels
into the report, because a region count that quietly halved is indistinguishable
from a specimen that genuinely had half as many features.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .backends import InstanceMask
from .config import MasksConfig


@dataclass
class RejectionTally:
    """Why instances did not survive, so a count change is always explainable."""

    counts: Counter = field(default_factory=Counter)
    total_in: int = 0

    def reject(self, reason: str, n: int = 1) -> None:
        if n:
            self.counts[reason] += n

    @property
    def total_rejected(self) -> int:
        return int(sum(self.counts.values()))

    def as_dict(self) -> dict:
        return {
            "instances_in": self.total_in,
            "instances_rejected": self.total_rejected,
            "rejected_by_reason": dict(sorted(self.counts.items())),
        }


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2])


def intersection_area(a: InstanceMask, b: InstanceMask) -> int:
    """Overlapping pixel count, evaluated only on the shared sub-rectangle."""
    if not _bboxes_overlap(a.bbox, b.bbox):
        return 0
    y0 = max(a.bbox[0], b.bbox[0])
    y1 = min(a.bbox[1], b.bbox[1])
    x0 = max(a.bbox[2], b.bbox[2])
    x1 = min(a.bbox[3], b.bbox[3])
    a_sub = a.crop[y0 - a.bbox[0] : y1 - a.bbox[0], x0 - a.bbox[2] : x1 - a.bbox[2]]
    b_sub = b.crop[y0 - b.bbox[0] : y1 - b.bbox[0], x0 - b.bbox[2] : x1 - b.bbox[2]]
    return int(np.count_nonzero(a_sub & b_sub))


def mask_iou(a: InstanceMask, b: InstanceMask) -> float:
    """Intersection over union."""
    inter = intersection_area(a, b)
    if inter == 0:
        return 0.0
    union = a.area + b.area - inter
    return float(inter / union) if union else 0.0


def containment(a: InstanceMask, b: InstanceMask) -> float:
    """Intersection over the *smaller* area: how nested the pair is.

    Unlike IoU this stays near 1.0 when a small mask sits wholly inside a large
    one, which is precisely the case IoU cannot see.
    """
    inter = intersection_area(a, b)
    if inter == 0:
        return 0.0
    smaller = min(a.area, b.area)
    return float(inter / smaller) if smaller else 0.0


def filter_by_area(
    instances: list[InstanceMask],
    shape: tuple[int, int],
    config: MasksConfig,
    tally: RejectionTally,
) -> list[InstanceMask]:
    """Drop regions too small to measure and regions large enough to be substrate."""
    max_area = config.max_area_fraction * float(shape[0] * shape[1])
    kept = []
    for instance in instances:
        if instance.area < config.min_area_px:
            tally.reject("area_below_min")
        elif instance.area > max_area:
            tally.reject("area_above_max")
        else:
            kept.append(instance)
    return kept


def deduplicate(
    instances: list[InstanceMask],
    config: MasksConfig,
    tally: RejectionTally,
) -> list[InstanceMask]:
    """Remove duplicate and nested detections in two single-rule passes."""
    # Pass 1: ordinary NMS on IoU, highest score first.
    by_score = sorted(instances, key=lambda m: (-m.score, -m.area, m.bbox))
    kept: list[InstanceMask] = []
    for candidate in by_score:
        if any(mask_iou(candidate, other) > config.iou_dedupe for other in kept):
            tally.reject("duplicate_iou")
        else:
            kept.append(candidate)

    # Pass 2: containment. Ordering by area descending means the survivor is
    # always considered before anything nested inside it, so the rule reduces to
    # "drop the later one" and no replacement bookkeeping is needed.
    if config.containment_keep == "larger":
        ordered = sorted(kept, key=lambda m: (-m.area, -m.score, m.bbox))
    else:
        ordered = sorted(kept, key=lambda m: (-m.score, -m.area, m.bbox))
    survivors: list[InstanceMask] = []
    for candidate in ordered:
        if any(containment(candidate, other) > config.containment_dedupe for other in survivors):
            tally.reject("duplicate_containment")
        else:
            survivors.append(candidate)
    return survivors


def _largest_component(crop: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    labels, count = ndimage.label(crop)
    if count <= 1:
        return crop
    sizes = ndimage.sum_labels(crop, labels, index=np.arange(1, count + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def fill_small_holes(crop: np.ndarray, max_fill_area: float) -> tuple[np.ndarray, int, int]:
    """Fill speckle holes, keep real ones.

    Returns the updated crop, the number of holes left unfilled, and their total
    area.  A hole cannot touch its own region's bounding box border by
    construction, so working on the crop rather than the full frame is safe -
    do not "fix" this back to a full-frame fill, it only costs memory.
    """
    from scipy import ndimage

    filled = ndimage.binary_fill_holes(crop)
    holes = filled & ~crop
    if not holes.any():
        return crop, 0, 0
    labels, count = ndimage.label(holes)
    result = crop.copy()
    kept_holes = 0
    kept_area = 0
    for index in range(1, count + 1):
        hole = labels == index
        area = int(np.count_nonzero(hole))
        if area <= max_fill_area:
            result |= hole
        else:
            kept_holes += 1
            kept_area += area
    return result, kept_holes, kept_area


def postprocess(
    instances: list[InstanceMask],
    shape: tuple[int, int],
    config: MasksConfig,
) -> tuple[list[InstanceMask], RejectionTally]:
    """Filter, clean and deduplicate raw backend output.

    Border-touching regions are **never dropped** here.  Their visible geometry
    is well defined and worth reporting; what is not defined is their true size,
    so they are flagged and excluded from image-level aggregates downstream.
    """
    tally = RejectionTally(total_in=len(instances))
    cleaned: list[InstanceMask] = []
    for instance in instances:
        crop = instance.crop
        if config.keep_largest_component:
            crop = _largest_component(crop)
        max_fill = config.max_fill_area_fraction * float(np.count_nonzero(crop))
        crop, n_holes, hole_area = fill_small_holes(crop, max_fill)
        updated = instance.with_crop(crop)
        updated.meta["n_holes"] = n_holes
        updated.meta["hole_area_px"] = hole_area
        updated.meta["touches_border"] = updated.touches_border(shape)
        cleaned.append(updated)

    kept = filter_by_area(cleaned, shape, config, tally)
    kept = deduplicate(kept, config, tally)
    # Deterministic region ids: sort by position, not by score, so two runs on
    # the same image produce identical CSV row order.
    kept.sort(key=lambda m: (m.centroid[0], m.centroid[1]))
    return kept, tally


def label_map(instances: list[InstanceMask], shape: tuple[int, int]) -> np.ndarray:
    """A uint16 label image, 0 = background, region ids starting at 1."""
    if len(instances) >= np.iinfo(np.uint16).max:
        raise ValueError(f"too many regions for a uint16 label map: {len(instances)}")
    labels = np.zeros(shape, dtype=np.uint16)
    for index, instance in enumerate(instances, start=1):
        y0, y1, x0, x1 = instance.bbox
        window = labels[y0:y1, x0:x1]
        window[instance.crop] = index
    return labels
