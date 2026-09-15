"""Mask post-processing: the policies that stop features being double-counted."""

from __future__ import annotations

import numpy as np
import pytest

from sem_segment.backends import InstanceMask
from sem_segment.config import MasksConfig
from sem_segment.masks import (
    RejectionTally,
    containment,
    deduplicate,
    fill_small_holes,
    intersection_area,
    label_map,
    mask_iou,
    postprocess,
)


def square(y0, x0, size, *, score=1.0, shape=(128, 128)):
    mask = np.zeros(shape, dtype=bool)
    mask[y0 : y0 + size, x0 : x0 + size] = True
    return InstanceMask.from_mask(mask, score=score, backend="test")


def disk(cy, cx, radius, *, score=1.0, shape=(128, 128)):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    return InstanceMask.from_mask(mask, score=score, backend="test")


def test_from_mask_returns_none_for_an_empty_mask():
    assert InstanceMask.from_mask(np.zeros((16, 16), dtype=bool), score=1.0, backend="t") is None


def test_bbox_and_area_are_tight():
    instance = square(10, 20, 8)
    assert instance.bbox == (10, 18, 20, 28)
    assert instance.area == 64
    np.testing.assert_allclose(instance.centroid, (13.5, 23.5))


def test_full_mask_round_trips():
    instance = square(10, 20, 8)
    restored = InstanceMask.from_mask(instance.full_mask((128, 128)), score=1.0, backend="t")
    assert restored.bbox == instance.bbox
    np.testing.assert_array_equal(restored.crop, instance.crop)


def test_disjoint_masks_have_no_intersection():
    assert intersection_area(square(0, 0, 10), square(50, 50, 10)) == 0
    assert mask_iou(square(0, 0, 10), square(50, 50, 10)) == 0.0


def test_iou_and_containment_disagree_on_nested_masks():
    """The reason containment is checked at all.

    A small mask wholly inside a large one has a low IoU, so IoU-only dedupe
    keeps both and the feature is measured twice.
    """
    outer = square(10, 10, 40)
    inner = square(25, 25, 10)
    assert mask_iou(outer, inner) < 0.1
    assert containment(outer, inner) == pytest.approx(1.0)


def test_identical_masks_are_deduplicated_by_iou():
    config = MasksConfig()
    tally = RejectionTally()
    kept = deduplicate([square(10, 10, 20), square(10, 10, 20, score=0.5)], config, tally)
    assert len(kept) == 1
    assert tally.counts["duplicate_iou"] == 1


def test_nested_masks_keep_the_larger_by_default():
    """SAM 3 often scores a tight inner core higher than the true boundary."""
    outer = square(10, 10, 40, score=0.6)
    inner = square(25, 25, 10, score=0.99)
    tally = RejectionTally()
    kept = deduplicate([outer, inner], MasksConfig(), tally)
    assert len(kept) == 1
    assert kept[0].area == outer.area
    assert tally.counts["duplicate_containment"] == 1


def test_containment_keep_higher_score_is_selectable():
    outer = square(10, 10, 40, score=0.6)
    inner = square(25, 25, 10, score=0.99)
    kept = deduplicate(
        [outer, inner], MasksConfig(containment_keep="higher_score"), RejectionTally()
    )
    assert len(kept) == 1
    assert kept[0].area == inner.area


def test_distinct_features_survive_deduplication():
    kept = deduplicate([disk(30, 30, 10), disk(90, 90, 10)], MasksConfig(), RejectionTally())
    assert len(kept) == 2


def test_area_filters_tally_their_reasons():
    instances = [square(0, 0, 2), disk(64, 64, 10), square(0, 0, 120)]
    config = MasksConfig(min_area_px=25.0, max_area_fraction=0.25)
    kept, tally = postprocess(instances, (128, 128), config)
    assert len(kept) == 1
    assert tally.counts["area_below_min"] == 1
    assert tally.counts["area_above_max"] == 1
    assert tally.total_in == 3


def test_small_holes_are_filled_and_large_holes_are_kept():
    """An annulus must not be silently turned into a disk."""
    crop = np.ones((60, 60), dtype=bool)
    crop[5, 5] = False  # speckle
    crop[20:40, 20:40] = False  # a real 400 px hole
    filled, n_holes, hole_area = fill_small_holes(crop, max_fill_area=10.0)
    assert filled[5, 5]  # speckle filled
    assert not filled[30, 30]  # annulus preserved
    assert n_holes == 1 and hole_area == 400


def test_postprocess_records_holes_and_border_contact():
    ring = np.zeros((128, 128), dtype=bool)
    yy, xx = np.mgrid[0:128, 0:128]
    radius = np.hypot(yy - 64, xx - 64)
    ring[(radius < 30) & (radius > 15)] = True
    kept, _ = postprocess(
        [InstanceMask.from_mask(ring, score=1.0, backend="t")],
        (128, 128),
        MasksConfig(max_fill_area_fraction=0.1),
    )
    assert kept[0].meta["n_holes"] == 1
    assert kept[0].meta["touches_border"] is False


def test_border_regions_are_flagged_but_never_dropped():
    edge = square(0, 0, 30)
    kept, tally = postprocess([edge], (128, 128), MasksConfig())
    assert len(kept) == 1
    assert kept[0].meta["touches_border"] is True
    assert tally.total_rejected == 0


def test_region_order_is_positional_and_deterministic():
    """Two runs on the same image must produce identical CSV row order."""
    instances = [disk(90, 20, 8, score=0.3), disk(10, 60, 8, score=0.9), disk(10, 20, 8, score=0.5)]
    first, _ = postprocess(list(instances), (128, 128), MasksConfig())
    second, _ = postprocess(list(reversed(instances)), (128, 128), MasksConfig())
    assert [m.bbox for m in first] == [m.bbox for m in second]
    centroids = [m.centroid for m in first]
    assert centroids == sorted(centroids)


def test_label_map_assigns_one_id_per_region():
    instances = [disk(30, 30, 10), disk(90, 90, 10)]
    labels = label_map(instances, (128, 128))
    assert labels.dtype == np.uint16
    assert sorted(np.unique(labels)) == [0, 1, 2]
    assert int((labels == 1).sum()) == instances[0].area


def test_largest_component_removes_detached_speckle():
    mask = np.zeros((128, 128), dtype=bool)
    mask[20:60, 20:60] = True
    mask[100:103, 100:103] = True  # detached fragment
    kept, _ = postprocess(
        [InstanceMask.from_mask(mask, score=1.0, backend="t")],
        (128, 128),
        MasksConfig(keep_largest_component=True),
    )
    assert kept[0].area == 40 * 40
