"""Sobel gradient operator and its exact least-squares inverse.

The whole package rests on two linear-algebra facts, stated here and verified
by tests:

1. **Forward.** ``sobel`` applies the normalized Sobel pair with REFLECT
   padding.  Normalization is 1/8, chosen so a unit-slope ramp responds with
   exactly 1 -- the output is "intensity change per pixel", the unit CD-SEM
   edge slopes are naturally expressed in.  For i.i.d. pixel noise of variance
   ``sigma^2`` each output channel carries ``(12/64) sigma^2``
   (:data:`SOBEL_NOISE_GAIN`); the noise is attenuated but becomes spatially
   correlated (the [1, 2, 1] smoothing).

2. **Inverse.** Reflect padding makes the forward operator EXACTLY a circular
   convolution on the whole-sample symmetric extension of the image (period
   ``2H - 2`` by ``2W - 2``): the even extension's neighbor of sample 0 is
   sample 1, which is precisely what reflect padding supplies.  On that
   extended torus the normal equations of

        min_u  || S_x * u - g_x ||^2 + || S_y * u - g_y ||^2

   diagonalize in the DFT basis, so :func:`reconstruct_from_sobel` solves them
   exactly in O(N log N).  The null space is DC plus the TWO FULL NYQUIST
   LINES (kx = pi for every ky, and ky = pi for every kx): on those lines the
   [1, 2, 1] smoothing factor of one Sobel component and the central-difference
   factor of the other BOTH vanish, so no Sobel data constrains them.  They
   are unrecoverable BY ANY method; the solve zeroes them -- together with a
   small neighborhood controlled by ``null_threshold``, because next to the
   lines the operator gain ``1/|S(w)|`` diverges and would amplify any
   prediction error there -- and the caller supplies the mean (typically the
   noisy input's crop mean; Poisson noise is mean-preserving).  A predicted
   gradient field need not be integrable (curl-free): the least-squares solve
   is then the orthogonal projection onto realizable fields, which is the
   right way to discard the inconsistent part.

Conventions: images are ``[B, 1, H, W]`` tensors; gradient fields are
``[B, 2, H, W]`` with channel 0 = d/dx (x = width axis, increasing rightward)
and channel 1 = d/dy (y = height axis, increasing downward).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

#: Per-channel white-noise variance gain of the normalized Sobel operator:
#: sum of squared kernel coefficients, (1+4+1+1+4+1) / 8^2 = 12/64.
SOBEL_NOISE_GAIN = 12.0 / 64.0

_KERNEL_X = (
    torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
    )
    / 8.0
)
_KERNEL_Y = _KERNEL_X.t().contiguous()


def sobel_kernels(device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """The stacked correlation kernels as a ``[2, 1, 3, 3]`` conv2d weight."""
    return torch.stack([_KERNEL_X, _KERNEL_Y]).unsqueeze(1).to(device=device, dtype=dtype)


def sobel(image: torch.Tensor) -> torch.Tensor:
    """Normalized Sobel gradient of a ``[B, 1, H, W]`` image -> ``[B, 2, H, W]``.

    Reflect padding keeps the output the same size and, critically, makes the
    operator exactly invertible (up to its null space) by
    :func:`reconstruct_from_sobel` -- see the module docstring.
    """
    if image.dim() != 4 or image.shape[1] != 1:
        raise ValueError(f"expected [B, 1, H, W], got shape {tuple(image.shape)}")
    if image.shape[2] < 2 or image.shape[3] < 2:
        raise ValueError(f"image sides must be >= 2 for reflect padding, got {tuple(image.shape)}")
    padded = F.pad(image, (1, 1, 1, 1), mode="reflect")
    return F.conv2d(padded, sobel_kernels(image.device, image.dtype))


def _even_extend(field: torch.Tensor) -> torch.Tensor:
    """Whole-sample even extension in both axes: [H, W] -> [2H-2, 2W-2]."""
    field = torch.cat([field, field.flip(-1)[..., 1:-1]], dim=-1)
    return torch.cat([field, field.flip(-2)[..., 1:-1, :]], dim=-2)


def _odd_even_extend(field: torch.Tensor, *, odd_axis: int) -> torch.Tensor:
    """Odd extension along ``odd_axis`` (-1 or -2), even along the other.

    A gradient component measured through reflect padding is automatically odd
    about the mirror samples of its own axis (the reflected neighborhood is
    symmetric, so the derivative there is zero); extending a *predicted* field
    the same way simply projects it onto the class of fields the forward
    operator can produce, which is what the least-squares solve does anyway.
    """
    if odd_axis == -1:
        field = torch.cat([field, -field.flip(-1)[..., 1:-1]], dim=-1)
        return torch.cat([field, field.flip(-2)[..., 1:-1, :]], dim=-2)
    if odd_axis == -2:
        field = torch.cat([field, field.flip(-1)[..., 1:-1]], dim=-1)
        return torch.cat([field, -field.flip(-2)[..., 1:-1, :]], dim=-2)
    raise ValueError(f"odd_axis must be -1 or -2, got {odd_axis}")


def _kernel_spectrum(kernel: torch.Tensor, shape: tuple[int, int], device: torch.device) -> torch.Tensor:
    """DFT of the 3x3 CORRELATION kernel embedded circularly on ``shape``.

    conv2d computes cross-correlation; correlation with K equals convolution
    with K flipped, so the flipped kernel is embedded with its center at the
    origin and transformed.
    """
    height, width = shape
    embedded = torch.zeros(height, width, dtype=torch.float64, device=device)
    flipped = kernel.flip(0, 1).to(dtype=torch.float64, device=device)
    embedded[:3, :3] = flipped
    embedded = torch.roll(embedded, shifts=(-1, -1), dims=(0, 1))
    return torch.fft.fft2(embedded)


def reconstruct_from_sobel(
    gradients: torch.Tensor,
    *,
    mean: torch.Tensor | float = 0.0,
    null_threshold: float = 1e-3,
) -> torch.Tensor:
    """Least-squares image from a (possibly non-integrable) Sobel field.

    ``gradients`` is ``[B, 2, H, W]`` (channels d/dx, d/dy as produced by
    :func:`sobel`); ``mean`` is the DC value the result cannot know -- a float
    or a ``[B]`` tensor (e.g. the noisy input's crop mean).  Returns
    ``[B, 1, H, W]`` float32.  Exact inverse of :func:`sobel` up to the
    operator's null space; for an arbitrary field it is the (regularized)
    orthogonal projection described in the module docstring.

    ``null_threshold`` is RELATIVE to the largest spectral gain: modes with
    ``|Sx|^2 + |Sy|^2 < null_threshold * max`` are zeroed.  The default 1e-3
    keeps the worst error amplification below ~30x while touching only the
    immediate neighborhood of the (exactly null) Nyquist lines; low
    frequencies -- whose 1/|w| gain is the unavoidable physics of integrating
    a gradient -- stay untouched at any realistic crop size.
    """
    if gradients.dim() != 4 or gradients.shape[1] != 2:
        raise ValueError(f"expected [B, 2, H, W], got shape {tuple(gradients.shape)}")
    batch, _, height, width = gradients.shape
    if height < 3 or width < 3:
        raise ValueError(f"reconstruction needs H, W >= 3, got {height}x{width}")
    device = gradients.device
    work = gradients.to(torch.float64)
    gx_ext = _odd_even_extend(work[:, 0], odd_axis=-1)
    gy_ext = _odd_even_extend(work[:, 1], odd_axis=-2)
    shape = (2 * height - 2, 2 * width - 2)

    spectrum_x = _kernel_spectrum(_KERNEL_X, shape, device)
    spectrum_y = _kernel_spectrum(_KERNEL_Y, shape, device)
    denominator = spectrum_x.abs() ** 2 + spectrum_y.abs() ** 2
    # Null space of the operator pair (DC + both full Nyquist lines) and its
    # ill-conditioned neighborhood: excluded from the solve and set to zero.
    invertible = denominator > denominator.max() * null_threshold

    numerator = spectrum_x.conj() * torch.fft.fft2(gx_ext) + spectrum_y.conj() * torch.fft.fft2(gy_ext)
    solution = torch.zeros_like(numerator)
    solution[..., invertible] = numerator[..., invertible] / denominator[invertible]
    image = torch.fft.ifft2(solution).real[..., :height, :width]

    image = image - image.mean(dim=(-2, -1), keepdim=True)
    offset = torch.as_tensor(mean, dtype=torch.float64, device=device).reshape(-1)
    if offset.numel() == 1:
        offset = offset.expand(batch)
    elif offset.numel() != batch:
        raise ValueError(f"mean must be a scalar or [B]={batch} values, got {offset.numel()}")
    image = image + offset.view(batch, 1, 1)
    return image.unsqueeze(1).to(torch.float32)
