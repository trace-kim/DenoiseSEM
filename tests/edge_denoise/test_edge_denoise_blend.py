"""Regression tests for the full-frame tile blend (``denoise_full_frame``).

An identity denoiser round-trips through ANY normalized weighting, so it
cannot see the seams the blend exists to prevent.  These tests use denoisers
whose output depends on the position inside the tile: a corrupted border ring
(the zero-padded / never-supervised outer pixels) and per-tile constant
offsets (the disagreement between overlapping tiles).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from edge_denoise.distill import _positions, _window1d, denoise_full_frame


def _ring_corrupter(width: int, value: float = 1.0):
    """Identity except the outermost ``width`` px of every tile are ``value``
    (model range; 1.0 -> intensity 1.0)."""

    def fn(tiles: torch.Tensor) -> torch.Tensor:
        out = tiles.clone()
        out[:, :, :width, :] = value
        out[:, :, -width:, :] = value
        out[:, :, :, :width] = value
        out[:, :, :, -width:] = value
        return out

    return fn


def _frame(shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    # Intensities in [0.25, 0.75]: a corrupted value of 1.0 is at least 0.25 off.
    return 0.25 + 0.5 * np.random.default_rng(seed).random(shape)


def test_window_vanishes_at_the_tile_edges_but_stays_positive() -> None:
    window = _window1d(512)
    assert window.min() > 0.0
    assert window[0] == window[-1] < 1e-4
    assert abs(window.max() - 1.0) < 1e-4  # even length: the peak sits between two samples
    # Constant overlap-add: two windows half a tile apart cross-fade to ~1.
    assert np.allclose(window[:256] + window[256:], 1.0, atol=1e-2)


def test_tile_edge_pixels_barely_leak_into_the_blend() -> None:
    """A tile's outermost pixel enters the blend with ~0 weight wherever another
    tile overlaps.  Regression for the 0.1 window floor, which let a corrupted
    ring in at 0.1/1.1 of the vote and drew lines along every tile boundary."""
    frame = _frame((96, 160))
    out = denoise_full_frame(_ring_corrupter(1), frame, tile=64, stride=32, tile_batch=7)
    assert np.abs(out - frame)[1:-1, 1:-1].max() < 0.01
    # The frame's own border ring is covered by tile edges only: the network's
    # estimate there is used as is (the frame is never padded).
    assert np.allclose(out[0, :], 1.0) and np.allclose(out[:, -1], 1.0)


def test_margin_excludes_the_ring_exactly_except_at_the_frame_border() -> None:
    frame = _frame((96, 150), seed=1)  # both axes end with a snapped last tile
    assert _positions(150, 64, 30)[-1] == 86 and _positions(96, 64, 30)[-1] == 32
    out = denoise_full_frame(_ring_corrupter(2), frame, tile=64, stride=30, tile_batch=64, margin=2)
    assert np.allclose(out[2:-2, 2:-2], frame[2:-2, 2:-2], atol=1e-9)
    assert np.allclose(out[:2, :], 1.0) and np.allclose(out[:, -2:], 1.0)


def test_identity_round_trips_with_a_margin() -> None:
    frame = np.random.default_rng(3).random((50, 70))
    out = denoise_full_frame(lambda t: t, frame, tile=16, stride=6, tile_batch=5, margin=3)
    assert np.allclose(out, frame, atol=1e-6)


def test_per_tile_offsets_fade_instead_of_stepping() -> None:
    """Overlapping tiles never agree exactly (tile-global statistics); the
    blend must turn that disagreement into a slow fade, never a step."""
    frame = _frame((64, 192), seed=2)  # one row of five tiles at stride 32
    offsets = torch.tensor([1.0, -1.0, -1.0, 1.0, 1.0]) * 0.2  # +-0.1 in intensity units

    def offset_per_tile(tiles: torch.Tensor) -> torch.Tensor:
        return tiles + offsets[: tiles.shape[0]].view(-1, 1, 1, 1)

    out = denoise_full_frame(offset_per_tile, frame, tile=64, stride=32, tile_batch=64)
    residual = out - frame
    assert residual.max() - residual.min() > 0.1  # the offsets are really there ...
    # ... but never as a step: the steepest cross-fade slope of a 64-px Hann is
    # ~0.0097 per pixel here; the 0.1 floor produced 0.018 where one tile leaves
    # and another enters at the same column.
    assert np.abs(np.diff(residual, axis=1)).max() < 0.012


def test_stride_wider_than_the_valid_region_is_rejected() -> None:
    frame = _frame((64, 64))
    with pytest.raises(ValueError, match="valid region"):
        denoise_full_frame(lambda t: t, frame, tile=64, stride=61, margin=2)
    with pytest.raises(ValueError, match="margin"):
        denoise_full_frame(lambda t: t, frame, tile=64, stride=32, margin=-1)
