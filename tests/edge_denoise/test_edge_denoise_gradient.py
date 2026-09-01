from __future__ import annotations

import numpy as np
import pytest
import torch

from edge_denoise.gradient import (
    SOBEL_NOISE_GAIN,
    reconstruct_from_sobel,
    sobel,
    sobel_kernels,
)


def _smooth(image: torch.Tensor, sigma: float = 2.0, taps: int = 11) -> torch.Tensor:
    offsets = torch.arange(taps, dtype=torch.float32) - (taps - 1) / 2.0
    kernel = torch.exp(-(offsets**2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    half = taps // 2
    out = torch.nn.functional.conv2d(
        torch.nn.functional.pad(image, (half, half, 0, 0), mode="reflect"),
        kernel.view(1, 1, 1, taps),
    )
    return torch.nn.functional.conv2d(
        torch.nn.functional.pad(out, (0, 0, half, half), mode="reflect"),
        kernel.view(1, 1, taps, 1),
    )


# ---------------------------------------------------------------------------
# forward operator


def test_kernel_normalization_gives_unit_ramp_response() -> None:
    ramp_x = torch.arange(32, dtype=torch.float32).expand(32, 32)[None, None]
    g = sobel(ramp_x)
    gx = g[:, 0, 4:-4, 4:-4]
    gy = g[:, 1, 4:-4, 4:-4]
    assert torch.allclose(gx, torch.ones_like(gx))
    assert gy.abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_constant_image_has_zero_gradient() -> None:
    g = sobel(torch.full((1, 1, 16, 16), 0.37))
    assert g.abs().max().item() == 0.0


def test_sobel_is_linear() -> None:
    torch.manual_seed(0)
    a = torch.randn(2, 1, 16, 16)
    b = torch.randn(2, 1, 16, 16)
    combined = sobel(2.0 * a - 3.0 * b)
    assert torch.allclose(combined, 2.0 * sobel(a) - 3.0 * sobel(b), atol=1e-5)


def test_white_noise_gain_matches_kernel_energy() -> None:
    kernels = sobel_kernels()
    assert float((kernels[0] ** 2).sum()) == pytest.approx(SOBEL_NOISE_GAIN)
    assert float((kernels[1] ** 2).sum()) == pytest.approx(SOBEL_NOISE_GAIN)
    torch.manual_seed(1)
    noise = torch.randn(64, 1, 64, 64)
    interior = sobel(noise)[:, :, 4:-4, 4:-4]
    assert interior.var(unbiased=False).item() == pytest.approx(SOBEL_NOISE_GAIN, rel=0.02)


def test_sobel_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="B, 1, H, W"):
        sobel(torch.zeros(1, 2, 8, 8))
    with pytest.raises(ValueError, match="B, 1, H, W"):
        sobel(torch.zeros(8, 8))


# ---------------------------------------------------------------------------
# reconstruction


def test_round_trip_recovers_a_smooth_image() -> None:
    torch.manual_seed(0)
    image = _smooth(torch.randn(2, 1, 48, 48))
    recovered = reconstruct_from_sobel(sobel(image), mean=image.mean(dim=(1, 2, 3)))
    assert (recovered - image).abs().max().item() < 5e-3


def test_round_trip_preserves_subpixel_edge_positions() -> None:
    """The metrology property: a bar's threshold crossings survive
    sobel -> reconstruct to well under a hundredth of a pixel."""
    cols = torch.arange(64, dtype=torch.float64)
    coverage = torch.clamp(
        torch.minimum(cols + 1.0, torch.tensor(40.7))
        - torch.maximum(cols, torch.tensor(20.3)),
        0.0,
        1.0,
    )
    bar = (0.2 + 0.6 * coverage).expand(64, 64).clone()[None, None].float()
    recovered = reconstruct_from_sobel(sobel(bar), mean=bar.mean(dim=(1, 2, 3)))

    def crossings(profile: np.ndarray) -> np.ndarray:
        threshold = 0.5 * (profile.min() + profile.max())
        delta = profile - threshold
        indices = np.nonzero(delta[:-1] * delta[1:] < 0.0)[0]
        return indices + delta[indices] / (delta[indices] - delta[indices + 1])

    original = crossings(bar[0, 0, 32].numpy())
    roundtrip = crossings(recovered[0, 0, 32].numpy())
    assert len(original) == len(roundtrip) == 2
    np.testing.assert_allclose(roundtrip, original, atol=0.01)


def test_reconstruction_restores_the_requested_mean() -> None:
    torch.manual_seed(2)
    image = _smooth(torch.randn(3, 1, 32, 32))
    means = torch.tensor([0.1, -0.4, 2.0])
    recovered = reconstruct_from_sobel(sobel(image), mean=means)
    assert torch.allclose(recovered.mean(dim=(1, 2, 3)), means, atol=1e-5)
    scalar = reconstruct_from_sobel(sobel(image), mean=0.25)
    assert torch.allclose(scalar.mean(dim=(1, 2, 3)), torch.full((3,), 0.25), atol=1e-5)


def test_reconstruction_of_a_non_integrable_field_is_finite_and_stable() -> None:
    torch.manual_seed(3)
    garbage = torch.randn(1, 2, 32, 32)
    recovered = reconstruct_from_sobel(garbage, mean=0.0)
    assert torch.isfinite(recovered).all()
    # The null-mode cutoff bounds the amplification: a unit-scale random field
    # must not reconstruct into an arbitrarily large image.
    assert recovered.abs().max().item() < 50.0


def test_reconstruction_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="B, 2, H, W"):
        reconstruct_from_sobel(torch.zeros(1, 1, 8, 8))
    with pytest.raises(ValueError, match="mean"):
        reconstruct_from_sobel(torch.zeros(2, 2, 8, 8), mean=torch.zeros(3))
