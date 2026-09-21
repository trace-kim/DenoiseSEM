"""The public API: purity, the measurement invariant, and end-to-end accuracy."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic_images import make_config

from sem_segment.config import Config
from sem_segment.pipeline import SegmentationResult, _tile_origins, segment_image


def disks(size=160, width=260, truth=((80.0, 50.0, 18.0), (80.0, 130.0, 24.0), (80.0, 215.0, 12.0))):
    from scipy.special import erf

    image = np.full((size, width), 0.15)
    yy, xx = np.mgrid[0:size, 0:width].astype(float)
    for cy, cx, radius in truth:
        distance = np.hypot(yy - cy, xx - cx)
        image = np.maximum(image, 0.15 + 0.7 * 0.5 * (1 + erf((radius - distance) / (np.sqrt(2) * 1.5))))
    return image, truth


def test_segment_image_returns_a_populated_result():
    image, truth = disks()
    result = segment_image(image, make_config())
    assert isinstance(result, SegmentationResult)
    assert result.region_count == len(truth)
    assert len(result.coarse) == len(result.refined) == len(result.regions)
    assert result.shape == image.shape


def test_segment_image_writes_nothing(tmp_path, monkeypatch):
    """The API is pure: a caller that only wants numbers never touches disk."""
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())
    segment_image(disks()[0], make_config())
    assert set(tmp_path.iterdir()) == before


def test_segment_image_does_not_modify_the_input_array():
    image, _ = disks()
    original = image.copy()
    segment_image(image, make_config())
    np.testing.assert_array_equal(image, original)


def test_refinement_beats_the_mask_boundary_on_known_radii():
    """End to end, against ground truth: this is the point of the package."""
    image, truth = disks()
    result = segment_image(image, make_config())
    by_x = sorted(result.regions, key=lambda r: r.refined.centroid_x)
    coarse_errors, refined_errors = [], []
    for region, (_, _, radius) in zip(by_x, sorted(truth, key=lambda t: t[1])):
        coarse_errors.append(np.sqrt(region.coarse.area_px2 / np.pi) - radius)
        refined_errors.append(np.sqrt(region.refined.area_px2 / np.pi) - radius)
    assert np.abs(refined_errors).max() < 0.05
    assert np.abs(refined_errors).max() < np.abs(coarse_errors).max()


def test_contrast_stretch_cannot_move_a_measurement():
    """The load-bearing invariant of the whole package.

    A stretch changes what the model sees. It must not change a measured edge,
    because measurement always reads the untouched array.
    """
    image, _ = disks()
    plain = segment_image(image, make_config(segmentation={"contrast_stretch": None}))
    stretched = segment_image(image, make_config(segmentation={"contrast_stretch": (2.0, 98.0)}))
    assert plain.region_count == stretched.region_count
    for a, b in zip(plain.regions, stretched.regions):
        assert a.refined.cd_px == pytest.approx(b.refined.cd_px, abs=1e-9)
        assert a.refined.centroid_x == pytest.approx(b.refined.centroid_x, abs=1e-9)


def test_disabling_refinement_still_produces_coarse_metrology():
    result = segment_image(disks()[0], make_config(refine={"enabled": False}))
    assert result.refined == []
    assert all(r.coarse is not None and r.refined is None for r in result.regions)
    assert np.isnan(result.diagnostics.valid_fraction)


def test_diagnostics_explain_every_dropped_mask():
    image, _ = disks()
    result = segment_image(image, make_config(masks={"min_area_px": 800.0}))
    rejections = result.diagnostics.rejections
    assert rejections["instances_in"] == 3
    assert rejections["rejected_by_reason"]["area_below_min"] >= 1
    assert result.region_count == rejections["instances_in"] - rejections["instances_rejected"]


def test_empty_result_warns_rather_than_failing():
    result = segment_image(np.full((64, 64), 0.5), make_config())
    assert result.region_count == 0
    assert any("No regions survived" in w for w in result.diagnostics.warnings)
    assert result.image_stats["region_count"] == 0


def test_border_regions_are_flagged_in_the_warnings():
    from scipy.special import erf

    image = np.full((120, 120), 0.15)
    yy, xx = np.mgrid[0:120, 0:120].astype(float)
    distance = np.hypot(yy - 10.0, xx - 10.0)
    image = np.maximum(image, 0.15 + 0.7 * 0.5 * (1 + erf((22.0 - distance) / (np.sqrt(2) * 1.5))))
    result = segment_image(image, make_config())
    assert any("touch the image border" in w for w in result.diagnostics.warnings)


def test_rows_carry_both_methods_and_optional_nanometres():
    result = segment_image(disks()[0], make_config())
    rows = result.rows()
    assert "cd_px_coarse" in rows[0] and "cd_px_refined" in rows[0]
    assert not any(k.endswith("_nm") for k in rows[0])
    scaled = result.rows(pixel_size_nm=2.0)
    assert scaled[0]["cd_nm_refined"] == pytest.approx(rows[0]["cd_px_refined"] * 2.0)


def test_label_map_ids_match_region_ids():
    result = segment_image(disks()[0], make_config())
    labels = result.label_map()
    assert sorted(np.unique(labels)) == [0] + [r.region_id for r in result.regions]


def test_results_are_deterministic():
    image, _ = disks()
    first = segment_image(image, make_config())
    second = segment_image(image, make_config())
    assert [r.region_id for r in first.regions] == [r.region_id for r in second.regions]
    for a, b in zip(first.regions, second.regions):
        assert a.refined.cd_px == b.refined.cd_px


def test_image_preparations_are_shared_without_changing_measurements(monkeypatch):
    from dataclasses import asdict
    from scipy import ndimage
    from sem_segment import pipeline, refine

    config = make_config()
    image, _ = disks()
    images = [image, image * .9 + .03]
    original_strength = refine.edge_strength_along

    # Compute the previous per-contour path as an independent numerical baseline.
    with monkeypatch.context() as old:
        old.setattr(pipeline, "refine_all", lambda contours, pixels, settings, **kw: [
            refine.refine_contour(c, pixels, settings, spacing_px=kw["spacing_px"], search_px=r)
            for c, r in zip(contours, kw["search_px"])])
        old.setattr(refine, "edge_strength_along", lambda points, pixels, **kw: original_strength(points, pixels))
        expected = [segment_image(pixels, config) for pixels in images]

    calls = []
    original_gradient = ndimage.gaussian_gradient_magnitude

    def counted(*args, **kwargs):
        calls.append(1)
        return original_gradient(*args, **kwargs)

    monkeypatch.setattr(ndimage, "gaussian_gradient_magnitude", counted)
    for pixels, baseline in zip(images, expected):
        result = segment_image(pixels, config)
        for a, b in zip(result.regions, baseline.regions):
            for method in ("coarse", "refined"):
                np.testing.assert_allclose(list(asdict(getattr(a, method)).values()),
                                           list(asdict(getattr(b, method)).values()), rtol=0, atol=1e-10)
        assert result.region_count == baseline.region_count
        assert result.diagnostics.warnings == baseline.diagnostics.warnings
        assert result.diagnostics.edge_strength_change == pytest.approx(baseline.diagnostics.edge_strength_change)
        timings = result.diagnostics.timings_s
        assert "edge_strength" in timings and "prepare" in timings
        assert timings["total"] >= sum(v for k, v in timings.items() if k != "total") - .001
    assert len(calls) == len(images)  # One per image; never retain another image's gradient.


def test_series_can_reuse_backend_without_reusing_image_results(monkeypatch):
    from sem_segment import pipeline
    from sem_segment.backends import build_segmenter

    config = make_config()
    backend = build_segmenter(config)
    image, _ = disks()

    def unexpected_build(*args):
        pytest.fail("A supplied backend must not be reloaded")

    monkeypatch.setattr(pipeline, "build_segmenter", unexpected_build)
    assert segment_image(image, config, segmenter=backend).region_count == 3
    assert segment_image(np.full_like(image, .5), config, segmenter=backend).region_count == 0


def test_oversized_frames_are_resized_not_tiled_by_default():
    """Resize has no seams and costs nothing measurable, so it is the default."""
    config = Config.model_validate({"segmentation": {"backend": "sam3_auto"}})
    from sem_segment.pipeline import _plan_scaling

    mode, info, warnings = _plan_scaling((2048, 2048), config)
    assert mode == "resize"
    assert info["scale"] == pytest.approx(1008 / 2048)
    assert any("measured on the original pixels" in w for w in warnings)


def test_small_frames_are_left_alone():
    from sem_segment.pipeline import _plan_scaling

    config = Config.model_validate({"segmentation": {"backend": "sam3_auto"}})
    mode, _, warnings = _plan_scaling((512, 512), config)
    assert mode == "none" and warnings == []


def test_classical_backend_never_triggers_resizing():
    """Only a fixed-resolution model needs the frame scaled."""
    from sem_segment.pipeline import _plan_scaling

    mode, _, _ = _plan_scaling((4096, 4096), make_config())
    assert mode == "none"


def test_tile_origins_clamp_the_last_tile_instead_of_padding():
    """Padding fabricates a hard border the model segments as a feature."""
    origins = _tile_origins(2048, 1008, 752)
    assert origins[0] == 0
    assert origins[-1] == 2048 - 1008
    assert all(o + 1008 <= 2048 for o in origins)


def test_tile_origins_of_a_small_frame_is_a_single_tile():
    assert _tile_origins(500, 1008, 752) == [0]


def test_tiling_drops_instances_truncated_by_an_interior_seam():
    """Masks are never blended at a seam; whole instances are selected instead."""
    from sem_segment.backends import InstanceMask
    from sem_segment.pipeline import _segment_tiled

    class FakeSegmenter:
        def segment(self, patch):
            height, width = patch.shape[:2]
            whole = np.zeros((height, width), dtype=bool)
            whole[20:60, 20:60] = True
            clipped = np.zeros((height, width), dtype=bool)
            clipped[0:30, 40:80] = True  # runs off the top of the tile
            return [
                InstanceMask.from_mask(whole, score=0.9, backend="fake"),
                InstanceMask.from_mask(clipped, score=0.9, backend="fake"),
            ]

    rgb = np.zeros((300, 300, 3), dtype=np.uint8)
    kept, extra = _segment_tiled(FakeSegmenter(), rgb, tile=200, overlap=100)
    assert extra["tiles_truncated_instances"] > 0
    # Everything kept records which tile produced it and how interior it was.
    assert all("tile" in m.meta and "tile_margin" in m.meta for m in kept)
