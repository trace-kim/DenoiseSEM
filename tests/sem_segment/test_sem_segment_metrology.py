"""Metrology against shapes whose measures have closed forms.

Every assertion here compares to an exact analytic value rather than a
previously recorded output, so the suite cannot drift along with a regression.
"""

from __future__ import annotations

import numpy as np
import pytest

from sem_segment.config import MetrologyConfig
from sem_segment.contours import Contour
from sem_segment.metrology import (
    chord_widths,
    edge_roughness,
    equivalent_ellipse,
    feret_diameters,
    measure_shape,
    polygon_moments,
    summarise_image,
)
from sem_segment.refine import REJECT_OK, RefinedContour


def circle(radius=10.0, n=720, centre=(0.0, 0.0)):
    angle = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([centre[0] + radius * np.sin(angle), centre[1] + radius * np.cos(angle)])


def ellipse(a=20.0, b=10.0, n=720, rotation_deg=0.0):
    """Semi-axis ``a`` along x, ``b`` along y, rotated counter-clockwise."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x, y = a * np.cos(t), b * np.sin(t)
    phi = np.radians(rotation_deg)
    return np.column_stack(
        [x * np.sin(phi) + y * np.cos(phi), x * np.cos(phi) - y * np.sin(phi)]
    )


def rectangle(height=10.0, width=30.0, n_per_side=200):
    t = np.linspace(0.0, 1.0, n_per_side, endpoint=False)
    top = np.column_stack([np.zeros(n_per_side), t * width])
    right = np.column_stack([t * height, np.full(n_per_side, width)])
    bottom = np.column_stack([np.full(n_per_side, height), width - t * width])
    left = np.column_stack([height - t * height, np.zeros(n_per_side)])
    return np.vstack([top, right, bottom, left])


CONFIG = MetrologyConfig()


def test_polygon_moments_of_a_circle_match_the_closed_form():
    radius = 10.0
    area, cy, cx, mu_yy, mu_xx, _ = polygon_moments(circle(radius, n=2000))
    assert area == pytest.approx(np.pi * radius**2, rel=1e-5)
    assert (cy, cx) == pytest.approx((0.0, 0.0), abs=1e-9)
    # For a disk the second moment about any centroidal axis is r^2 / 4.
    assert mu_xx == pytest.approx(radius**2 / 4.0, rel=1e-5)
    assert mu_yy == pytest.approx(radius**2 / 4.0, rel=1e-5)


def test_equivalent_ellipse_recovers_known_axes_and_orientation():
    major, minor, orientation, eccentricity = equivalent_ellipse(ellipse(a=20.0, b=10.0))
    assert major == pytest.approx(40.0, rel=1e-4)  # axis length = 2a
    assert minor == pytest.approx(20.0, rel=1e-4)
    assert abs(orientation) % 180.0 == pytest.approx(0.0, abs=0.5)
    assert eccentricity == pytest.approx(np.sqrt(1 - (10.0 / 20.0) ** 2), rel=1e-3)


@pytest.mark.parametrize("rotation", [0.0, 30.0, 75.0])
def test_ellipse_axes_are_rotation_invariant(rotation):
    major, minor, _, _ = equivalent_ellipse(ellipse(a=20.0, b=10.0, rotation_deg=rotation))
    assert major == pytest.approx(40.0, rel=1e-3)
    assert minor == pytest.approx(20.0, rel=1e-3)


def test_feret_of_a_circle_is_its_diameter_in_every_direction():
    feret_min, feret_max, feret_mean = feret_diameters(circle(10.0, n=2000))
    assert feret_min == pytest.approx(20.0, rel=1e-3)
    assert feret_max == pytest.approx(20.0, rel=1e-3)
    assert feret_mean == pytest.approx(20.0, rel=1e-3)


def test_feret_of_a_rectangle_matches_its_sides_and_diagonal():
    feret_min, feret_max, _ = feret_diameters(rectangle(height=10.0, width=30.0))
    assert feret_min == pytest.approx(10.0, rel=1e-3)
    assert feret_max == pytest.approx(np.hypot(10.0, 30.0), rel=1e-3)


def test_feret_mean_satisfies_cauchys_formula():
    """Mean caliper width of a convex body equals its perimeter over pi."""
    from sem_segment.contours import polygon_perimeter
    from sem_segment.metrology import convex_hull_points

    points = ellipse(a=20.0, b=10.0)
    _, _, feret_mean = feret_diameters(points)
    hull_perimeter = polygon_perimeter(convex_hull_points(points))
    assert feret_mean == pytest.approx(hull_perimeter / np.pi, rel=1e-9)


def test_circle_measures_are_exact():
    radius = 10.0
    measures = measure_shape(Contour(points=circle(radius, n=2000)), CONFIG)
    assert measures.area_px2 == pytest.approx(np.pi * radius**2, rel=1e-4)
    assert measures.perimeter_px == pytest.approx(2 * np.pi * radius, rel=1e-4)
    assert measures.equivalent_diameter_px == pytest.approx(2 * radius, rel=1e-4)
    assert measures.circularity == pytest.approx(1.0, rel=1e-3)
    assert measures.solidity == pytest.approx(1.0, rel=1e-3)
    assert measures.convexity == pytest.approx(1.0, rel=1e-3)
    assert measures.aspect_ratio == pytest.approx(1.0, rel=1e-3)


def test_rectangle_measures_are_exact():
    measures = measure_shape(Contour(points=rectangle(10.0, 30.0)), CONFIG)
    assert measures.area_px2 == pytest.approx(300.0, rel=1e-3)
    assert measures.perimeter_px == pytest.approx(80.0, rel=1e-3)
    assert measures.bbox_height_px == pytest.approx(10.0, rel=1e-2)
    assert measures.bbox_width_px == pytest.approx(30.0, rel=1e-2)
    assert measures.bbox_fill == pytest.approx(1.0, rel=1e-2)
    # 4 pi A / P^2 for a 10x30 rectangle
    assert measures.circularity == pytest.approx(4 * np.pi * 300.0 / 80.0**2, rel=1e-3)


def test_solidity_detects_a_non_convex_shape():
    """A cross is far less solid than its hull."""
    from sem_segment.metrology import convex_hull_points

    cross = np.array(
        [
            [-10.0, -3.0], [-3.0, -3.0], [-3.0, -10.0], [3.0, -10.0], [3.0, -3.0],
            [10.0, -3.0], [10.0, 3.0], [3.0, 3.0], [3.0, 10.0], [-3.0, 10.0],
            [-3.0, 3.0], [-10.0, 3.0],
        ]
    )
    measures = measure_shape(Contour(points=cross), CONFIG)
    # Cross area = 2 * 20 * 6 - 6^2 = 204. Its hull is a 20x20 square with four
    # 7x7 corner triangles removed = 400 - 98 = 302. Both are exact.
    assert measures.area_px2 == pytest.approx(204.0)
    assert measures.solidity == pytest.approx(204.0 / 302.0, rel=1e-3)
    assert measures.solidity < 0.7
    assert convex_hull_points(cross).shape[0] < cross.shape[0]


def test_area_subtracts_holes():
    outer = circle(20.0, n=1000)
    hole = circle(10.0, n=1000)
    measures = measure_shape(Contour(points=outer), CONFIG, holes=[hole])
    assert measures.area_px2 == pytest.approx(np.pi * (400.0 - 100.0), rel=1e-3)


def test_chord_width_of_a_circle_is_its_diameter():
    widths = chord_widths(circle(10.0, n=1000), (0.0, 0.0), 36)
    np.testing.assert_allclose(widths, 20.0, rtol=1e-3)


def test_chord_width_of_a_rectangle_spans_side_to_diagonal():
    points = rectangle(height=10.0, width=30.0)
    widths = chord_widths(points, (5.0, 15.0), 180)
    assert widths.min() == pytest.approx(10.0, rel=0.02)
    assert widths.max() == pytest.approx(np.hypot(10.0, 30.0), rel=0.02)


def test_cd_definition_selects_the_reported_column():
    points = rectangle(height=10.0, width=30.0)
    equivalent = measure_shape(Contour(points=points), MetrologyConfig()).cd_px
    feret = measure_shape(
        Contour(points=points), MetrologyConfig(cd_definition="feret_min")
    ).cd_px
    assert equivalent == pytest.approx(2 * np.sqrt(300.0 / np.pi), rel=1e-3)
    assert feret == pytest.approx(10.0, rel=1e-2)
    assert equivalent != feret


def test_degenerate_polygon_returns_none_rather_than_nonsense():
    assert measure_shape(Contour(points=np.zeros((2, 2))), CONFIG) is None
    collinear = np.column_stack([np.zeros(5), np.arange(5.0)])
    assert measure_shape(Contour(points=collinear), CONFIG) is None


def _refined_from(points, normals, displacement):
    n = points.shape[0]
    return RefinedContour(
        base_points=points,
        normals=normals,
        displacement=displacement,
        valid=np.ones(n, dtype=bool),
        reasons=np.full(n, REJECT_OK, dtype=np.int16),
        edge_width=np.full(n, np.nan),
        contrast=np.full(n, 0.6),
    )


def test_roughness_measures_the_specimen_not_the_segmentation_error():
    """A boundary displaced by a large constant is smooth; roughness must be ~0."""
    n = 512
    angle = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    base = np.column_stack([30.0 * np.sin(angle), 30.0 * np.cos(angle)])
    normals = np.column_stack([np.sin(angle), np.cos(angle)])

    smooth = edge_roughness(_refined_from(base, normals, np.full(n, 3.0)), CONFIG)
    assert smooth["ler_3sigma_px"] == pytest.approx(0.0, abs=0.02)

    # Now a genuinely rough boundary with zero mean displacement.
    rng = np.random.default_rng(0)
    rough_amplitude = 0.4
    rough = edge_roughness(
        _refined_from(base, normals, rng.normal(0.0, rough_amplitude, n)), CONFIG
    )
    assert rough["ler_3sigma_px"] == pytest.approx(3.0 * rough_amplitude, rel=0.2)
    assert rough["ler_3sigma_px"] > smooth["ler_3sigma_px"] * 10


def test_roughness_recovers_a_known_sinusoidal_amplitude():
    n = 1024
    angle = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    base = np.column_stack([100.0 * np.sin(angle), 100.0 * np.cos(angle)])
    normals = np.column_stack([np.sin(angle), np.cos(angle)])
    amplitude = 0.5
    # 120 cycles around the ring is ~8.5 samples per cycle, far above the 8 px
    # cutoff, so the high-pass passes it essentially undistorted. At 40 cycles
    # the Gaussian transfer function keeps only ~85% and the closed form below
    # would not apply - the filter response is part of the measurement.
    wobble = amplitude * np.sin(120.0 * angle)
    result = edge_roughness(_refined_from(base, normals, wobble), CONFIG, spacing_px=1.0)
    # 3 sigma of a sine of amplitude A is 3 * A / sqrt(2).
    assert result["ler_3sigma_px"] == pytest.approx(3.0 * amplitude / np.sqrt(2.0), rel=0.05)


def test_roughness_is_nan_when_detrending_is_disabled():
    n = 64
    base = circle(20.0, n=n)
    normals = base / np.linalg.norm(base, axis=1, keepdims=True)
    result = edge_roughness(
        _refined_from(base, normals, np.zeros(n)), MetrologyConfig(roughness_detrend="none")
    )
    assert np.isnan(result["ler_3sigma_px"])


def test_row_adds_nanometre_columns_only_when_a_pixel_size_is_given():
    from sem_segment.metrology import RegionMetrology

    measures = measure_shape(Contour(points=circle(10.0)), CONFIG)
    region = RegionMetrology(
        region_id=1, coarse=measures, refined=measures, score=1.0, backend="t", prompt=None,
        touches_border=False, n_holes=0, hole_area_px2=0.0, valid_fraction=1.0,
        shift_mean_px=0.1, shift_abs_mean_px=0.1, shift_3sigma_px=0.3,
        edge_width_px=1.5, edge_contrast=0.6,
    )
    plain = region.to_row()
    assert not any(k.endswith("_nm") for k in plain)
    assert "area_px2_coarse" in plain and "area_px2_refined" in plain

    scaled = region.to_row(pixel_size_nm=5.82812)
    assert scaled["shift_mean_nm"] == pytest.approx(0.1 * 5.82812)
    assert scaled["area_nm2_coarse"] == pytest.approx(plain["area_px2_coarse"] * 5.82812**2)


def test_summary_excludes_border_regions_from_aggregates_by_default():
    from sem_segment.metrology import RegionMetrology

    def region(rid, radius, border):
        measures = measure_shape(Contour(points=circle(radius, centre=(50.0, 50.0))), CONFIG)
        return RegionMetrology(
            region_id=rid, coarse=measures, refined=measures, score=1.0, backend="t", prompt=None,
            touches_border=border, n_holes=0, hole_area_px2=0.0, valid_fraction=1.0,
            shift_mean_px=0.0, shift_abs_mean_px=0.0, shift_3sigma_px=0.0,
            edge_width_px=1.5, edge_contrast=0.6,
        )

    regions = [region(1, 10.0, False), region(2, 10.0, False), region(3, 2.0, True)]
    summary = summarise_image(regions, (128, 128), CONFIG)
    assert summary["region_count"] == 3
    assert summary["region_count_aggregated"] == 2
    assert summary["region_count_border"] == 1
    # The truncated region would have dragged the mean down.
    assert summary["cd_px_refined_mean"] == pytest.approx(20.0, rel=1e-3)

    included = summarise_image(regions, (128, 128), CONFIG, include_border_regions=True)
    assert included["cd_px_refined_mean"] < summary["cd_px_refined_mean"]


def test_summary_is_empty_but_valid_with_no_regions():
    summary = summarise_image([], (64, 64), CONFIG)
    assert summary["region_count"] == 0
    assert summary["image_height_px"] == 64
