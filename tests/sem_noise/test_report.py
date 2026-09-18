from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("scipy")

from sem_noise.report import _acquisition_figure, _errorbar, _with_error, raw_histogram_examples


def test_corner_error_is_used_in_both_text_and_plot() -> None:
    row = {"corner_max_px": 2.0, "corner_max_se": 0.3,
           "corner_top_left_dy_px": -1.0, "corner_top_left_dy_se": 0.2}
    assert "0.3" in _with_error(row, "corner_max_px")
    assert "0.2" in _with_error(row, "corner_top_left_dy_px")

    class Axis:
        def errorbar(self, x, y, **kwargs):
            np.testing.assert_array_equal(kwargs["yerr"], [0.3])

    _errorbar(Axis(), np.array([0]), [row], "corner_max_px")


def test_raw_histograms_count_every_uint8_value_including_clipping(tmp_path: Path) -> None:
    raw = np.arange(256, dtype=np.uint8).reshape(16, 16)
    stack = np.stack([raw] * 6).astype(np.float32)  # same lossless cache conversion as PNG pipeline
    stack[3, 4:8] = 255
    stack[5, 8:12] = 0
    before = stack.copy()
    indices = np.arange(6) * 4
    html = raw_histogram_examples(tmp_path, stack, np.array([True, False, True, True, True, True]),
                                  indices, "uint8", (0, 255))
    with (tmp_path / "raw_histograms.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {int(row["frame_index"]) for row in rows} == {0, 12, 20}
    for i in (0, 3, 5):
        bins = [row for row in rows if int(row["frame_index"]) == indices[i]]
        assert len(bins) == 256
        assert float(bins[0]["bin_left_dn"]) == -0.5
        assert float(bins[-1]["bin_right_dn"]) == 255.5
        counts = np.array([int(row["pixel_count"]) for row in bins])
        np.testing.assert_array_equal(counts, np.bincount(stack[i].astype(np.uint8).ravel(), minlength=256))
        assert counts.sum() == 256
    assert "Clipped pixels are included" in html and 'id="raw-image-histograms"' in html
    assert (tmp_path / "raw_histograms.png").is_file()
    np.testing.assert_array_equal(stack, before)


def test_acquisition_plots_use_frame_order_and_show_failed_estimates_as_gaps() -> None:
    import matplotlib.pyplot as plt
    frames = [{"frame_index": i, "included": i != 3, "raw_mean_dn": 70 + 2 * i,
               "native_mean_dn": 60 + i, "aligned_mean_dn": 61 + i} for i in range(5)]
    geometry = [{"frame_index": i, "dx_px": i * 0.2, "dy_px": -i * 0.1,
                 "affine_dx_px": i * 0.3, "affine_dy_px": -i * 0.4, "corner_max_px": i * 0.05}
                for i in (0, 1, 3, 4)]  # frame 2 failed; frame 3 explicitly excluded
    fig = _acquisition_figure(frames, geometry, {"enabled": True, "method": "affine"})
    try:
        assert len(fig.axes) == 4
        np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), [70, 72, 74, np.nan, 78])
        np.testing.assert_allclose(fig.axes[1].lines[0].get_ydata(), [0, 0.2, np.nan, np.nan, 0.8])
        np.testing.assert_allclose(fig.axes[2].lines[0].get_ydata(), [0, 0.3, np.nan, np.nan, 1.2])
        np.testing.assert_allclose(fig.axes[3].lines[0].get_ydata(), [0, 0.05, np.nan, np.nan, 0.2])
        assert all(ax.get_xlabel() == "Acquisition index" for ax in fig.axes)
    finally:
        plt.close(fig)
