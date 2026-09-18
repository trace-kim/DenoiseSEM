from __future__ import annotations

from html.parser import HTMLParser
import json

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("matplotlib")
pytest.importorskip("scipy")

import matplotlib.pyplot as plt

from sem_noise.pair_matching import measure_quantile_brightness
from sem_noise.pair_diagnostics import PAIR_PANELS
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


def test_difference_range_controls_keep_options_separate_from_saturation_cells() -> None:
    class Elements(HTMLParser):
        def __init__(self):
            super().__init__()
            self.items = []

        def handle_starttag(self, tag, attrs):
            self.items.append((tag, dict(attrs)))

    row = {"input_index": 0, "target_index": 4, "quantile_status": "complete"}
    for suffix, limit, saturated in (("", 20, 4.5), ("_p99", 30, 0.7), ("_full", 220, 0)):
        row[f"difference_image{suffix}"] = f"pairs/example{suffix}.png"
        row[f"colour_limit{suffix}_dn"] = limit
        row.update({f"{panel}_saturated{suffix}_pct": saturated for panel in PAIR_PANELS})
    original = row.copy()
    parsed = Elements()
    parsed.feed(_pair_section(row, []))
    options = [attrs for tag, attrs in parsed.items if tag == "option"]
    assert len(options) == 3
    for option, suffix, limit, saturated in zip(options, ("", "_p99", "_full"), (20, 30, 220), (4.5, 0.7, 0)):
        assert option["data-image"] == row[f"difference_image{suffix}"]
        assert float(option["data-limit"]) == limit
        assert list(map(float, json.loads(option["data-saturation-values"]))) == [saturated] * 5
    # The update targets only the five metric cells, never the option labels.
    cells = [(tag, attrs) for tag, attrs in parsed.items if "data-saturation" in attrs]
    assert len(cells) == 5 and all(tag == "td" for tag, _ in cells)
    assert row == original
