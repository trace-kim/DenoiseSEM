from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("scipy")

from sem_noise.report import _errorbar, _with_error


def test_corner_error_is_used_in_both_text_and_plot() -> None:
    row = {"corner_max_px": 2.0, "corner_max_se": 0.3,
           "corner_top_left_dy_px": -1.0, "corner_top_left_dy_se": 0.2}
    assert "0.3" in _with_error(row, "corner_max_px")
    assert "0.2" in _with_error(row, "corner_top_left_dy_px")

    class Axis:
        def errorbar(self, x, y, **kwargs):
            np.testing.assert_array_equal(kwargs["yerr"], [0.3])

    _errorbar(Axis(), np.array([0]), [row], "corner_max_px")
