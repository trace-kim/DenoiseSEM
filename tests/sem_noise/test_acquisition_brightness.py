from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")

from sem_noise.pair_diagnostics import acquisition_brightness
from sem_noise.pair_matching import (identity_transform, measure_brightness, measure_quantile_brightness,
                                    select_brightness_regions, warp_target)


def test_fixed_reference_recovers_gain_offset_and_exports_actual_corrected_images(tmp_path) -> None:
    reference = 20 + np.arange(64 * 64).reshape(64, 64) / 100
    gains, offsets = [1, 0.7, 1.3, 0.9], [0, 10, -8, 5]
    stack = np.stack([(reference - offset) / gain for gain, offset in zip(gains, offsets)])
    before = stack.copy()
    stack.flags.writeable = False
    regions = select_brightness_regions(reference, np.ones(reference.shape, bool))
    rows = acquisition_brightness(stack, np.ones(4, bool), np.arange(4) * 3, (None, None),
                                 {i: identity_transform() for i in range(4)}, regions, "",
                                 tmp_path / "acquisition_brightness")
    assert len(rows) == 4
    for i, row in enumerate(rows):
        assert row["reference_index"] == 0
        assert row["mean_pixels"] == reference.size
        assert row["raw_mean_dn"] == stack[i].mean()
        for prefix in ("two_region", "quantile"):
            assert row[f"{prefix}_gain"] == pytest.approx(gains[i])
            assert row[f"{prefix}_offset_dn"] == pytest.approx(offsets[i])
            assert row[f"{prefix}_mean_dn"] == pytest.approx(reference.mean())
            if "example_arrays" in row:
                with np.load(tmp_path / row["example_arrays"]) as saved:
                    np.testing.assert_allclose(saved[f"{prefix}_corrected"], reference)
                    assert row[f"{prefix}_mean_dn"] == saved[f"{prefix}_corrected"].mean()
                    np.testing.assert_array_equal(saved["raw"], stack[i])
                    np.testing.assert_array_equal(saved["reference"], reference)
    assert [r["frame_index"] for r in rows if "example_arrays" in r] == [0, 6, 9]
    np.testing.assert_array_equal(stack, before)


def test_full_image_means_retain_outliers_and_do_not_use_fit_crop_or_force_flatness(tmp_path) -> None:
    reference = 20 + np.tile(np.arange(64), (64, 1)).astype(float)
    target = reference * 2 + 7
    target[0] = 1e6  # outside geometric support, but MUST contribute to plotted means
    regions = select_brightness_regions(reference, np.ones(reference.shape, bool))
    transform = identity_transform()
    transform[0, 2] = 0.5
    rows = acquisition_brightness(np.stack([reference, target]), np.ones(2, bool), np.arange(2),
                                 (None, 1e6), {0: identity_transform(), 1: transform}, regions, "",
                                 tmp_path / "acquisition_brightness")
    row = rows[1]
    aligned, valid = warp_target(target, transform, invalid=target >= 1e6)
    measured = measure_brightness(reference, aligned, regions, valid)
    quantile = measure_quantile_brightness(reference, target)
    for prefix, estimate in (("two_region", measured), ("quantile", quantile)):
        assert row[f"{prefix}_gain"] == estimate["gain"]
        assert row[f"{prefix}_offset_dn"] == estimate["offset_dn"]
        corrected = estimate["gain"] * target + estimate["offset_dn"]
        assert row[f"{prefix}_mean_dn"] == corrected.mean()
        assert row[f"{prefix}_mean_dn"] > reference.mean() + 1000
        assert row[f"{prefix}_mean_dn"] != pytest.approx(corrected[valid].mean())
    assert row["quantile_input_pixels"] == row["quantile_target_pixels"] == target.size


def test_acquisition_failures_are_independent_and_exclusions_keep_raw_means(tmp_path) -> None:
    reference = 20 + np.tile(np.arange(64), (64, 1)).astype(float)
    stack = np.stack([reference - 7, reference, reference + 4, np.full_like(reference, 30)])
    labels = select_brightness_regions(reference, np.ones(reference.shape, bool))
    rows = acquisition_brightness(stack, np.array([False, True, True, True]), np.arange(4) * 5,
                                 (None, None), {1: identity_transform(), 3: identity_transform()},
                                 labels, "", tmp_path / "acquisition_brightness")
    assert [r["frame_index"] for r in rows] == [0, 5, 10, 15]
    assert all(r["reference_index"] == 5 for r in rows)
    assert rows[0]["raw_mean_dn"] == stack[0].mean()
    assert rows[0]["two_region_status"] == rows[0]["quantile_status"] == "excluded"
    assert rows[2]["two_region_status"] == "failed"
    assert "affine" in rows[2]["two_region_error"]
    assert "two_region_mean_dn" not in rows[2]
    assert rows[2]["quantile_mean_dn"] == pytest.approx(reference.mean())
    assert rows[3]["two_region_status"] == rows[3]["quantile_status"] == "failed"
    assert "equal" in rows[3]["two_region_error"] and "equal" in rows[3]["quantile_error"]
