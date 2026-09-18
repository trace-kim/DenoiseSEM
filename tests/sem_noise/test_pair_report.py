from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("matplotlib")
pytest.importorskip("scipy")

import matplotlib.pyplot as plt

from sem_noise.pair_matching import measure_quantile_brightness
from sem_noise.pair_report import _distribution_comparison, _pair_section, _pixel_scatter


def test_scatter_uses_each_corresponding_pixel_with_shared_linear_axes() -> None:
    fixed = np.arange(64, dtype=float).reshape(8, 8)
    valid = np.ones(fixed.shape, dtype=bool)
    valid[0, 0] = False
    arrays = {"input": fixed, "target": fixed[::-1].copy(),
              "translated_target": fixed[:, ::-1].copy(),
              "aligned_target": fixed * 2 + 10, "difference_valid": valid}
    arrays["target"][0, 0] = 99999  # excluded in all three comparisons
    row = {"input_index": 3, "target_index": 7, "gain": 0.5, "offset_dn": -5,
           "target_low_dn": 30, "target_high_dn": 110, "input_low_dn": 10, "input_high_dn": 50}
    original = row.copy()
    fig = _pixel_scatter(arrays, row)
    try:
        assert len(fig.axes) == 3
        for ax, key in zip(fig.axes, ("target", "translated_target", "aligned_target")):
            points = ax.collections[0].get_offsets()
            np.testing.assert_array_equal(points, np.column_stack((arrays[key][valid], fixed[valid])))
            assert ax.get_xscale() == ax.get_yscale() == "linear"
            assert ax.get_xlim() == fig.axes[0].get_xlim()
            assert ax.get_ylim() == fig.axes[0].get_ylim()
            assert ax.get_xlim()[1] < 99999
        # Raw and translation stages have an identity line, but no new fit.
        assert len(fig.axes[0].lines) == len(fig.axes[1].lines) == 1
        line = fig.axes[2].lines[1]
        np.testing.assert_allclose(line.get_ydata(), row["gain"] * line.get_xdata() + row["offset_dn"])
        np.testing.assert_array_equal(fig.axes[2].collections[1].get_offsets(), [[30, 10], [110, 50]])
        assert row == original
    finally:
        plt.close(fig)


def test_distribution_plot_compares_both_mappings_on_every_native_pixel() -> None:
    target = np.arange(256, dtype=np.uint8).reshape(16, 16)
    fixed = 2.0 * target + 5
    measured = measure_quantile_brightness(fixed, target)
    points = [{"target_dn": float(x), "input_dn": float(y)} for x, y in
              zip(measured["target_quantiles_dn"], measured["input_quantiles_dn"])]
    row = {"input_index": 0, "target_index": 4, "status": "complete", "gain": 1.5, "offset_dn": 10,
           "quantile_gain": measured["gain"], "quantile_offset_dn": measured["offset_dn"]}
    original = row.copy()
    fig = _distribution_comparison(fixed, target, row, points)
    try:
        assert len(fig.axes) == 3
        np.testing.assert_array_equal(fig.axes[0].collections[0].get_offsets(),
                                      np.column_stack((measured["target_quantiles_dn"], measured["input_quantiles_dn"])))
        assert len(fig.axes[1].patches) == 2 and len(fig.axes[2].patches) == 3
        distributions = ((fixed, target), (fixed, 1.5 * target + 10, 2.0 * target + 5))
        for ax, arrays in zip(fig.axes[1:], distributions):
            for patch, values in zip(ax.patches, arrays):
                histogram = patch.get_data()
                expected, _ = np.histogram(values, bins=histogram.edges)
                np.testing.assert_array_equal(histogram.values, expected)
                assert histogram.values.sum() == target.size
                assert histogram.edges[-1] == 515  # corrected values are not clipped to uint8
        assert row == original
    finally:
        plt.close(fig)


def test_every_pair_links_to_an_interactive_viewer_but_only_examples_embed_it() -> None:
    row = {"input_index": 0, "target_index": 4, "quantile_status": "complete",
           "difference_status": "complete", "difference_image": "pairs/example.png",
           "difference_viewer": "pairs/example.html", "colour_limit_dn": 220}
    for interactive in (False, True):
        html = _pair_section(row, [], interactive=interactive)
        assert 'href="pairs/example.html"' in html
        assert ('<iframe ' in html) == interactive
        assert '95% ?' not in html and '<select' not in html


def test_intermediate_examples_show_both_saved_brightness_corrections(tmp_path, monkeypatch) -> None:
    from sem_noise import pair_report

    fixed = np.arange(64, dtype=float).reshape(8, 8)
    valid = np.ones(fixed.shape, dtype=bool)
    arrays = {key: fixed.copy() for key in ("input", "target", "input_blurred", "target_blurred",
                                           "translated_target", "aligned_target")}
    arrays.update(corrected_target=fixed + 2, quantile_corrected_target=fixed * 1.1 - 3,
                  regions=np.zeros_like(fixed), difference_valid=valid,
                  input_blur_valid=valid, target_blur_valid=valid)
    np.savez(tmp_path / "example.npz", **arrays)
    seen = []

    def capture(out, name, fig, caption):
        if name.endswith("_images.png"):
            images = {ax.get_title(): ax.images[0] for ax in fig.axes}
            for title, key in (("Raw input A — untouched", "input"),
                               ("Target after affine + two-region", "corrected_target"),
                               ("Target after affine + percentiles", "quantile_corrected_target")):
                np.testing.assert_array_equal(images[title].get_array(), arrays[key])
                assert images[title].get_clim() == images["Raw input A — untouched"].get_clim()
            seen.append(name)
        plt.close(fig)
        return ""

    monkeypatch.setattr(pair_report, "_figure", capture)
    pair_report._example(tmp_path, {"input_index": 0, "target_index": 1, "example_arrays": "example.npz"})
    assert seen == ["example_images.png"]
