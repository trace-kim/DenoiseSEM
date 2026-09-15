"""Method 2 against ground truth.

The synthetic edges here are generated analytically rather than by convolving a
hard step, so the true edge position and blur width are exact and independent of
any filter implementation.  That makes these the only tests in the package that
can assert an absolute accuracy rather than a self-consistency.
"""

from __future__ import annotations

import numpy as np
import pytest
from synthetic_images import blurred_disk, gaussian_blurred_step

from sem_segment.backends import InstanceMask
from sem_segment.config import ContoursConfig, RefineConfig
from sem_segment.contours import Contour, polygon_area, trace_instance
from sem_segment.refine import (
    REJECT_LOW_CONTRAST,
    REJECT_OK,
    REJECT_OUT_OF_BOUNDS,
    _interpolate_short_gaps,
    refine_contour,
    sample_profiles,
)

ESTIMATORS = ["threshold", "gradient_peak", "erf"]


def vertical_edge_contour(*, guess_x: float, height: int = 40, n: int = 24) -> Contour:
    """Vertices on a vertical line at ``guess_x``, normals pointing +x."""
    ys = np.linspace(4.0, height - 5.0, n)
    points = np.column_stack([ys, np.full(n, guess_x)])
    normals = np.tile(np.array([[0.0, 1.0]]), (n, 1))
    return Contour(points=points, normals=normals)


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("true_x", [40.0, 40.25, 40.5, 40.75])
def test_edge_position_is_recovered_to_a_hundredth_of_a_pixel(estimator, true_x):
    """The measurement the whole package exists to produce."""
    image = gaussian_blurred_step(edge_x=true_x, sigma=1.5)
    # Start from a boundary that is wrong by ~2 px, the scale of SAM 3's
    # quantisation on a downsampled frame.
    contour = vertical_edge_contour(guess_x=true_x - 2.0)
    refined = refine_contour(contour, image, RefineConfig(estimator=estimator))

    assert refined.valid.all(), refined.reason_counts()
    measured = refined.refined_points[:, 1]
    assert np.mean(measured) == pytest.approx(true_x, abs=0.01)
    # And it is consistent along a straight edge, not merely right on average.
    assert np.std(measured) < 0.01


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_refinement_is_insensitive_to_the_starting_guess(estimator):
    """A region proposal that is off by several pixels must still converge."""
    true_x = 40.3
    image = gaussian_blurred_step(edge_x=true_x, sigma=1.5)
    measured = []
    for offset in (-3.0, -1.0, 0.0, 1.0, 3.0):
        refined = refine_contour(
            vertical_edge_contour(guess_x=true_x + offset),
            image,
            RefineConfig(estimator=estimator, search_px=6.0),
        )
        measured.append(float(np.mean(refined.refined_points[:, 1])))
    assert np.allclose(measured, true_x, atol=0.02)


@pytest.mark.parametrize("sigma", [0.8, 1.5, 2.5])
def test_erf_recovers_the_true_edge_width(sigma):
    """``sigma`` is a real measurement - beam blur plus edge slope - not a nuisance."""
    image = gaussian_blurred_step(edge_x=40.3, sigma=sigma)
    refined = refine_contour(
        vertical_edge_contour(guess_x=39.0), image, RefineConfig(estimator="erf", search_px=8.0)
    )
    assert np.nanmean(refined.edge_width) == pytest.approx(sigma, rel=0.05)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_dark_features_measure_the_same_as_bright_ones(estimator):
    """SEM polarity varies by detector; the result must not."""
    true_x = 40.3
    bright = gaussian_blurred_step(edge_x=true_x, low=0.2, high=0.8)
    dark = gaussian_blurred_step(edge_x=true_x, low=0.8, high=0.2)
    config = RefineConfig(estimator=estimator)
    a = refine_contour(vertical_edge_contour(guess_x=39.0), bright, config)
    b = refine_contour(vertical_edge_contour(guess_x=39.0), dark, config)
    assert np.mean(a.refined_points[:, 1]) == pytest.approx(true_x, abs=0.02)
    assert np.mean(b.refined_points[:, 1]) == pytest.approx(true_x, abs=0.02)


def test_refinement_corrects_a_biased_mask_boundary_on_a_disk():
    """End to end: a thresholded disk is biased; refinement removes the bias."""
    radius, centre = 24.0, 63.5
    image = blurred_disk(size=128, centre=(centre, centre), radius=radius, sigma=1.5)
    # A deliberately biased mask: thresholding high shrinks the region.
    instance = InstanceMask.from_mask(image > 0.62, score=1.0, backend="t")
    outer, _ = trace_instance(instance, ContoursConfig())
    from sem_segment.contours import attach_normals

    attach_normals(outer, instance.full_mask(image.shape), ContoursConfig())

    coarse_radius = np.sqrt(abs(polygon_area(outer.points)) / np.pi)
    refined = refine_contour(outer, image, RefineConfig(estimator="threshold"))
    refined_radius = np.sqrt(abs(polygon_area(refined.polygon)) / np.pi)

    assert refined.valid_fraction > 0.95
    # The coarse boundary is biased small; the refined one lands on the truth.
    assert coarse_radius < radius - 0.3
    assert refined_radius == pytest.approx(radius, abs=0.15)
    assert abs(refined_radius - radius) < abs(coarse_radius - radius)


def test_curvature_bias_on_a_small_disk_is_measured_and_bounded():
    """A normal-profile fit on a tight curve is biased; the doc must state how much."""
    results = {}
    for radius in (8.0, 16.0, 32.0):
        image = blurred_disk(size=128, centre=(63.5, 63.5), radius=radius, sigma=1.5)
        instance = InstanceMask.from_mask(image > 0.5, score=1.0, backend="t")
        outer, _ = trace_instance(instance, ContoursConfig())
        from sem_segment.contours import attach_normals

        attach_normals(outer, instance.full_mask(image.shape), ContoursConfig())
        refined = refine_contour(outer, image, RefineConfig(estimator="threshold"))
        measured = np.sqrt(abs(polygon_area(refined.polygon)) / np.pi)
        results[radius] = measured - radius
    # Bias shrinks as curvature does, and stays well under a tenth of a pixel.
    assert abs(results[8.0]) < 0.25
    assert abs(results[32.0]) < 0.1
    assert abs(results[32.0]) <= abs(results[8.0]) + 1e-9


def test_flat_region_is_rejected_for_low_contrast_not_silently_measured():
    flat = np.full((40, 80), 0.5)
    refined = refine_contour(vertical_edge_contour(guess_x=40.0), flat, RefineConfig())
    assert not refined.valid.any()
    assert refined.reason_counts() == {"low_contrast": len(refined.valid)}
    assert refined.valid_fraction == 0.0


def test_window_leaving_the_image_is_rejected_as_out_of_bounds():
    image = gaussian_blurred_step(height=40, width=80, edge_x=3.0)
    contour = vertical_edge_contour(guess_x=1.0, height=40)
    refined = refine_contour(contour, image, RefineConfig(search_px=6.0))
    assert (refined.reasons == REJECT_OUT_OF_BOUNDS).all()
    # Rejected, not clipped to the search limit.
    assert np.isnan(refined.displacement).all()


def test_polygon_excludes_invalid_vertices_rather_than_zero_filling():
    """Zero-filling would plant a notch far larger than the roughness signal."""
    image = gaussian_blurred_step(edge_x=40.3)
    contour = vertical_edge_contour(guess_x=38.3, n=20)
    refined = refine_contour(contour, image, RefineConfig())
    refined.valid[5] = False
    refined.displacement[5] = np.nan
    assert refined.polygon.shape[0] == len(refined.valid) - 1
    assert np.isfinite(refined.polygon).all()


def test_short_gaps_are_interpolated_and_long_gaps_are_not():
    displacement = np.arange(20, dtype=float)
    valid = np.ones(20, dtype=bool)
    valid[5:7] = False  # a 2-vertex gap
    valid[12:18] = False  # a 6-vertex gap
    displacement[~valid] = np.nan

    filled, repaired = _interpolate_short_gaps(
        displacement, valid, spacing_px=1.0, max_gap_px=3.0
    )
    assert repaired[5] and repaired[6]
    np.testing.assert_allclose(filled[5:7], [5.0, 6.0])
    assert not repaired[12:18].any()


def test_sample_profiles_shape_and_offsets():
    image = gaussian_blurred_step()
    contour = vertical_edge_contour(guess_x=40.0, n=7)
    profiles, offsets, in_bounds = sample_profiles(
        image, contour.points, contour.normals, search_px=6.0, step_px=0.25
    )
    assert profiles.shape == (7, 49)
    assert offsets[0] == -6.0 and offsets[-1] == 6.0
    assert in_bounds.all()


def test_cubic_sampling_beats_bilinear_on_subpixel_positions():
    """Why interp_order defaults to 3: bilinear has a once-per-pixel systematic."""
    errors = {1: [], 3: []}
    for true_x in np.linspace(40.0, 41.0, 11):
        image = gaussian_blurred_step(edge_x=true_x, sigma=1.2)
        for order in (1, 3):
            refined = refine_contour(
                vertical_edge_contour(guess_x=39.0),
                image,
                RefineConfig(estimator="threshold", interp_order=order),
            )
            errors[order].append(float(np.mean(refined.refined_points[:, 1])) - true_x)
    assert np.std(errors[3]) <= np.std(errors[1])


def test_reason_counts_only_reports_failures():
    image = gaussian_blurred_step(edge_x=40.3)
    refined = refine_contour(vertical_edge_contour(guess_x=39.0), image, RefineConfig())
    assert refined.reason_counts() == {}
    assert (refined.reasons == REJECT_OK).all()


def test_low_contrast_code_is_distinct_from_ok():
    assert REJECT_LOW_CONTRAST != REJECT_OK
