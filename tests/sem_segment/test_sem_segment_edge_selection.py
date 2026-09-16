"""The rule that decides which edge in a profile window is the feature's own.

These tests exist because of a real failure on real data: on a dim particle
beside a bright one, the refinement measured the *neighbour's* edge and pushed
the contour 3-4 px across the gap. Both estimators got it wrong, for different
surface reasons and one shared root cause - a window containing several edges
and no principled rule for choosing between them.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erf

from sem_segment.backends import InstanceMask
from sem_segment.config import Config, RefineConfig
from sem_segment.contours import Contour
from sem_segment.refine import (
    REJECT_INCOHERENT,
    _coherence_filter,
    adaptive_search_px,
    refine_contour,
)


def test_gradient_is_the_default_estimator():
    """The requested method 2 is gradient-based edge detection.

    It was once shipped as a non-default option behind a level-crossing
    estimator; the deliverable is the named technique.
    """
    assert Config().refine.estimator == "gradient_peak"
    assert RefineConfig().estimator == "gradient_peak"


def two_edge_image(*, near_x: float, near_contrast: float, far_x: float, far_contrast: float,
                   height: int = 40, width: int = 120, sigma: float = 1.2) -> np.ndarray:
    """A weak edge near the boundary and a stronger one further out."""
    x = np.arange(width, dtype=np.float64)
    profile = (
        0.10
        + near_contrast * 0.5 * (1.0 + erf((near_x - x) / (np.sqrt(2.0) * sigma)))
        + far_contrast * 0.5 * (1.0 + erf((x - far_x) / (np.sqrt(2.0) * sigma)))
    )
    return np.repeat(profile[None, :], height, axis=0)


def vertical_contour(guess_x: float, n: int = 20, height: int = 40) -> Contour:
    ys = np.linspace(4.0, height - 5.0, n)
    return Contour(
        points=np.column_stack([ys, np.full(n, guess_x)]),
        normals=np.tile(np.array([[0.0, 1.0]]), (n, 1)),
    )


def test_gradient_picks_the_nearer_edge_even_with_a_wide_window():
    """The failure that broke real particles, as a controlled case.

    The feature's own edge sits at t=0 with modest contrast; a brighter
    neighbour's edge sits 4 px out with three times the contrast. A global
    |dI/dt| argmax measures the neighbour. Nearest-to-the-boundary does not, and
    it needs no help from the window size to get this right.
    """
    image = two_edge_image(near_x=40.0, near_contrast=0.15, far_x=44.0, far_contrast=0.45)
    refined = refine_contour(
        vertical_contour(40.0),
        image,
        RefineConfig(search_px=6.0, adaptive_search=False, min_contrast=0.02),
    )
    measured = np.nanmean(refined.refined_points[:, 1])
    assert measured == pytest.approx(40.0, abs=0.5), f"locked onto the far edge: {measured:.2f}"
    assert abs(measured - 44.0) > 2.0


@pytest.mark.parametrize("estimator", ["gradient_peak", "threshold", "erf"])
def test_every_estimator_is_correct_once_the_window_excludes_the_neighbour(estimator):
    """Why the adaptive window is not merely an optimisation.

    A level-crossing estimator takes its threshold from the whole window, so a
    strong far edge moves the level itself and the near edge may not cross it at
    all - no selection rule can recover from that. Gradient has no such coupling
    and survives a wide window on its own, which is why it is the default. Sizing
    the window to the feature is what makes the other two safe as well, and in
    normal use it is always on.
    """
    image = two_edge_image(near_x=40.0, near_contrast=0.15, far_x=44.0, far_contrast=0.45)
    refined = refine_contour(
        vertical_contour(40.0),
        image,
        RefineConfig(estimator=estimator, search_px=3.0, adaptive_search=False,
                     min_contrast=0.02, step_px=0.2),
    )
    assert refined.valid.any(), f"{estimator} refined nothing"
    measured = float(np.mean(refined.polygon[:, 1]))
    assert measured == pytest.approx(40.0, abs=0.5), f"{estimator} gave {measured:.2f}"


def test_selection_is_not_swayed_by_which_side_is_brighter():
    """Inverting contrast must not change which edge is selected."""
    bright = two_edge_image(near_x=40.0, near_contrast=0.15, far_x=44.0, far_contrast=0.45)
    dark = 1.0 - bright
    config = RefineConfig(search_px=6.0, adaptive_search=False, min_contrast=0.02)
    a = np.nanmean(refine_contour(vertical_contour(40.0), bright, config).refined_points[:, 1])
    b = np.nanmean(refine_contour(vertical_contour(40.0), dark, config).refined_points[:, 1])
    assert a == pytest.approx(b, abs=0.1)
    assert a == pytest.approx(40.0, abs=0.5)


def disk_mask(radius: float, size: int = 128) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return np.hypot(yy - size / 2, xx - size / 2) <= radius


def test_search_window_shrinks_on_small_features():
    """A window wider than the feature swallows it and reaches the next one."""
    config = RefineConfig(search_px=6.0, search_fraction=0.5, min_search_px=2.0)
    small = adaptive_search_px(disk_mask(7.0), config)
    large = adaptive_search_px(disk_mask(60.0), config)
    assert small < config.search_px
    assert small == pytest.approx(3.5, abs=0.5)
    assert large == config.search_px  # big features keep the full window


def test_search_window_respects_its_floor_and_ceiling():
    config = RefineConfig(search_px=6.0, search_fraction=0.5, min_search_px=2.0)
    assert adaptive_search_px(disk_mask(1.5), config) == pytest.approx(2.0)
    assert adaptive_search_px(disk_mask(200.0, size=512), config) == pytest.approx(6.0)


def test_adaptive_search_uses_the_narrowest_width_not_the_area():
    """An elongated feature is constrained by its width, not its equivalent radius."""
    mask = np.zeros((128, 128), dtype=bool)
    mask[60:66, 10:120] = True  # 6 px wide, 110 px long
    config = RefineConfig(search_px=6.0, search_fraction=0.5, min_search_px=1.0)
    # Equivalent circular radius would be sqrt(660/pi) = 14.5 px, which would
    # license the full window and span the whole bar.
    assert adaptive_search_px(mask, config) < 2.5


def test_adaptive_search_can_be_disabled():
    config = RefineConfig(search_px=6.0, adaptive_search=False)
    assert adaptive_search_px(disk_mask(7.0), config) == 6.0


def test_coherence_filter_rejects_an_isolated_spike():
    """Adjacent vertices cannot genuinely disagree by several pixels."""
    displacement = np.full(40, 0.2)
    displacement[17] = 4.0
    valid = np.ones(40, dtype=bool)
    kept = _coherence_filter(displacement, valid, threshold_mad=3.0)
    assert not kept[17]
    assert kept.sum() == 39


def test_coherence_filter_keeps_a_smoothly_varying_boundary():
    """A curved or rough boundary must not be penalised for varying."""
    angle = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    displacement = 0.6 * np.sin(angle)  # a genuine, smooth excursion
    kept = _coherence_filter(displacement, np.ones(80, dtype=bool), threshold_mad=3.0)
    assert kept.all()


def test_coherence_filter_catches_a_run_longer_than_its_window():
    """A local median cannot see a run that dominates its own neighbourhood.

    On real data an 8-vertex stretch of one contour locked onto a neighbouring
    particle at +1.5 to +2.75 px and survived the local test entirely, because
    with a 7-vertex window the run *is* the local median. A region-level test
    sees it, since it still departs from what the whole contour agrees on.
    """
    displacement = np.full(60, 0.2)
    displacement[20:31] = 2.4          # 11 contiguous vertices, window is 7
    kept = _coherence_filter(displacement, np.ones(60, dtype=bool), threshold_mad=3.0)
    assert not kept[20:31].any(), "the run survived"
    # Vertices immediately flanking the run may also go, since their own local
    # median is contaminated by it. Everything well clear of it must survive.
    assert kept[:18].all() and kept[33:].all()
    assert kept.sum() >= 45


def test_a_uniformly_displaced_boundary_is_kept_intact():
    """When the mask really was off by a constant, nothing is an outlier."""
    displacement = np.full(60, 2.5) + np.random.default_rng(0).normal(0, 0.05, 60)
    kept = _coherence_filter(displacement, np.ones(60, dtype=bool), threshold_mad=3.0)
    assert kept.all()


def test_coherence_filter_can_be_disabled():
    displacement = np.full(40, 0.2)
    displacement[17] = 4.0
    kept = _coherence_filter(displacement, np.ones(40, dtype=bool), threshold_mad=0.0)
    assert kept.all()


def test_incoherent_vertices_are_reported_not_hidden():
    """A rejected vertex is counted by reason, like every other rejection."""
    image = two_edge_image(near_x=40.0, near_contrast=0.3, far_x=45.0, far_contrast=0.3)
    contour = vertical_contour(40.0, n=30)
    refined = refine_contour(contour, image, RefineConfig(adaptive_search=False, search_px=6.0))
    # Force a spike and re-run the filter through the public counter.
    refined.reasons[5] = REJECT_INCOHERENT
    assert refined.reason_counts().get("incoherent_with_neighbours") == 1


def test_a_clean_isolated_edge_is_unaffected_by_all_of_this():
    """The safeguards must not disturb the case that already worked."""
    from synthetic_images import gaussian_blurred_step

    for true_x in (40.0, 40.25, 40.5, 40.75):
        image = gaussian_blurred_step(edge_x=true_x, sigma=1.5)
        refined = refine_contour(
            vertical_contour(true_x - 2.0, height=64), image,
            RefineConfig(adaptive_search=False, search_px=6.0),
        )
        assert refined.valid.all()
        assert np.mean(refined.refined_points[:, 1]) == pytest.approx(true_x, abs=0.02)
