"""Drift registration of raw burst frames -- no clean image, no averaged target.

The question this module answers is the one that decides whether burst
fusion is deployable at all: *can the frames of a drifting burst be aligned
to sub-pixel precision from the noisy frames themselves?*  On the MIIC
peak-10 data the stock global-argmax cross-correlation cannot (errors of
several pixels: the line patterns are periodic, so the noisy correlation
locks onto a neighbouring period), which is precisely the objection that a
noisy-input registration "may not be available".  Two things fix it:

1. **Bounded search.**  Drift is continuous, so the shift between
   neighbouring frames of a burst is small and the correlation peak is
   searched only within ``radius`` pixels of the previous frame's estimate.
   The period of the pattern is then irrelevant as long as it exceeds
   ``2 * radius``.
2. **Least-squares refinement.**  The correlation peak of a soft-edged image
   is broad, so its sub-pixel position is a poor estimator (0.3-0.5 px on
   full 512x512 frames).  A Gauss-Newton solve of
   ``min_d sum_p (b(p) - a(p - d))^2`` on Gaussian-smoothed frames reaches
   0.01-0.04 px in every direction the image constrains -- the Cramer-Rao
   level for two Poisson frames -- because it uses every edge pixel with the
   weight its gradient deserves.  Smoothing (``sigma``) trades the noise of
   the gradient images against edge sharpness; sigma 2 px is right for
   peak-10 frames, sigma 1 for pre-denoised ones (``predenoise``).

Directions the image does *not* constrain (a shift along a field of
parallel lines) come out with a large covariance; the constant-velocity
Kalman/RTS smoother (:func:`smooth_trajectory`) then falls back on the
drift-continuity prior there, which is also where the estimate cannot
matter for the fused image.  The drift *within* a frame (the raster scans
rows over the frame time, so a moving sample shears the frame) is not
re-estimated per frame: it is the trajectory's velocity, position(k+1) -
position(k), which the smoother supplies for free.

Conventions (shared with :mod:`edge_denoise.drift` and the fusion module):

- ``position[k] = (dy, dx)``: the content of frame ``k`` is the scene shifted
  by ``+position`` relative to frame 0's mid-frame coordinates, i.e.
  ``y_k(r, c) ~ S(r - dy, c - dx)``.
- ``velocity[k]``: drift across frame ``k`` in px/frame; row ``r`` of the
  frame sits at ``position + velocity * (r / (H - 1) - 0.5)``.
- :func:`warp_frame` brings a frame INTO frame-0 coordinates by sampling it
  at ``(r + dy(r), c + dx(r))``; :func:`warp_scene` does the opposite (what
  the generator uses to *create* a drifted frame).

torch only (GPU-capable); numpy for the small linear algebra of the smoother.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

#: A per-frame full-frame denoiser ``[H, W]`` float in [0, 1] -> same, used to
#: register on denoised content instead of raw frames.
PredenoiseFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class Trajectory:
    """Registered drift of one burst (``N`` frames) relative to frame 0."""

    position: np.ndarray  # [N, 2] (dy, dx) mid-frame, smoothed
    velocity: np.ndarray  # [N, 2] px/frame across the frame, smoothed
    raw_position: np.ndarray  # [N, 2] the per-frame least-squares estimate
    covariance: np.ndarray  # [N, 2, 2] measurement covariance of raw_position

    def __post_init__(self) -> None:
        n = len(self.position)
        for name in ("position", "velocity", "raw_position"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (n, 2):
                raise ValueError(f"{name} must be [N, 2], got {value.shape}")
            setattr(self, name, value)
        self.covariance = np.asarray(self.covariance, dtype=np.float64)
        if self.covariance.shape != (n, 2, 2):
            raise ValueError(f"covariance must be [N, 2, 2], got {self.covariance.shape}")

    def __len__(self) -> int:
        return len(self.position)

    def row_shift(self, index: int, rows: np.ndarray, height: int) -> np.ndarray:
        """``[len(rows), 2]`` shift of frame ``index`` at the given rows."""
        frac = rows.astype(np.float64) / max(height - 1, 1) - 0.5
        return self.position[index][None, :] + self.velocity[index][None, :] * frac[:, None]

    def to_json(self) -> dict:
        return {
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "raw_position": self.raw_position.tolist(),
            "covariance": self.covariance.tolist(),
        }

    @classmethod
    def from_json(cls, payload: dict) -> "Trajectory":
        return cls(
            position=np.asarray(payload["position"], dtype=np.float64),
            velocity=np.asarray(payload["velocity"], dtype=np.float64),
            raw_position=np.asarray(payload["raw_position"], dtype=np.float64),
            covariance=np.asarray(payload["covariance"], dtype=np.float64),
        )

    @classmethod
    def identity(cls, count: int) -> "Trajectory":
        return cls(
            position=np.zeros((count, 2)),
            velocity=np.zeros((count, 2)),
            raw_position=np.zeros((count, 2)),
            covariance=np.zeros((count, 2, 2)),
        )


# ---------------------------------------------------------------------------
# image operators


def _as_batch(frames: torch.Tensor | np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(frames) if not torch.is_tensor(frames) else frames)
    tensor = tensor.to(device=device, dtype=torch.float32)
    if tensor.dim() == 2:
        tensor = tensor[None, None]
    elif tensor.dim() == 3:
        tensor = tensor[:, None]
    if tensor.dim() != 4 or tensor.shape[1] != 1:
        raise ValueError(f"expected [H, W], [N, H, W] or [N, 1, H, W], got {tuple(tensor.shape)}")
    return tensor


def gaussian_smooth(frames: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of ``[N, 1, H, W]`` with reflect padding."""
    if sigma <= 0.0:
        return frames
    radius = max(1, int(math.ceil(3.0 * sigma)))
    taps = torch.arange(-radius, radius + 1, device=frames.device, dtype=frames.dtype)
    kernel = torch.exp(-0.5 * (taps / sigma) ** 2)
    kernel = kernel / kernel.sum()
    padded = F.pad(frames, (radius, radius, radius, radius), mode="reflect")
    out = F.conv2d(padded, kernel.view(1, 1, 1, -1))
    return F.conv2d(out, kernel.view(1, 1, -1, 1))


def _sample_grid(height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ys = torch.arange(height, device=device, dtype=torch.float32)
    xs = torch.arange(width, device=device, dtype=torch.float32)
    return ys, xs


def _grid_sample(
    image: torch.Tensor, sample_y: torch.Tensor, sample_x: torch.Tensor, *, mode: str
) -> torch.Tensor:
    """Sample ``[1, 1, H, W]`` at float coordinates ``[H, W]`` (reflection outside)."""
    height, width = image.shape[-2:]
    grid = torch.stack(
        [sample_x / max(width - 1, 1) * 2.0 - 1.0, sample_y / max(height - 1, 1) * 2.0 - 1.0],
        dim=-1,
    )[None]
    return F.grid_sample(image, grid, mode=mode, padding_mode="reflection", align_corners=True)


def warp_scene(
    scene: torch.Tensor | np.ndarray,
    position: Sequence[float],
    velocity: Sequence[float] = (0.0, 0.0),
    *,
    mode: str = "bicubic",
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Create a drifted frame from a scene: ``out(r, c) = S(r - dy(r), c - dx(r))``
    with the row-dependent shift of the module conventions."""
    device = torch.device(device)
    image = _as_batch(scene, device)
    height, width = image.shape[-2:]
    ys, xs = _sample_grid(height, width, device)
    frac = ys / max(height - 1, 1) - 0.5
    dy = float(position[0]) + float(velocity[0]) * frac
    dx = float(position[1]) + float(velocity[1]) * frac
    sample_y = (ys - dy)[:, None].expand(height, width)
    sample_x = (xs[None, :] - dx[:, None]).expand(height, width)
    out = _grid_sample(image, sample_y, sample_x, mode=mode)
    return out[0, 0].cpu().numpy().astype(np.float64)


def warp_frame(
    frame: torch.Tensor | np.ndarray,
    position: Sequence[float],
    velocity: Sequence[float] = (0.0, 0.0),
    *,
    mode: str = "bicubic",
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Bring a drifted frame INTO frame-0 coordinates (the inverse of
    :func:`warp_scene`); returns ``(aligned, valid)`` where ``valid`` marks
    pixels whose sample point lies inside the frame."""
    device = torch.device(device)
    image = _as_batch(frame, device)
    height, width = image.shape[-2:]
    ys, xs = _sample_grid(height, width, device)
    frac = ys / max(height - 1, 1) - 0.5
    dy = float(position[0]) + float(velocity[0]) * frac
    dx = float(position[1]) + float(velocity[1]) * frac
    sample_y = (ys + dy)[:, None].expand(height, width)
    sample_x = (xs[None, :] + dx[:, None]).expand(height, width)
    out = _grid_sample(image, sample_y, sample_x, mode=mode)
    valid = (sample_y >= 0) & (sample_y <= height - 1) & (sample_x >= 0) & (sample_x <= width - 1)
    # Bicubic overshoot on noise-like frames is an interpolation artefact:
    # keep aligned frames inside the stored range (training crops do the same).
    return out[0, 0].clamp(0.0, 1.0).cpu().numpy().astype(np.float64), valid.cpu().numpy()


# ---------------------------------------------------------------------------
# per-frame shift estimation


def coarse_shift(
    reference: torch.Tensor,
    moving: torch.Tensor,
    *,
    radius: int,
    guess: tuple[int, int] = (0, 0),
) -> tuple[int, int]:
    """Integer ``(dy, dx)`` with ``moving ~ reference shifted by (dy, dx)``,
    searched within ``radius`` of ``guess`` on the windowed cross-correlation."""
    height, width = reference.shape[-2:]
    window = torch.outer(torch.hann_window(height, periodic=False), torch.hann_window(width, periodic=False))
    window = window.to(reference.device, reference.dtype)
    a = (reference[0, 0] - reference.mean()) * window
    b = (moving[0, 0] - moving.mean()) * window
    cross = torch.fft.fft2(a) * torch.conj(torch.fft.fft2(b))
    corr = torch.fft.ifft2(cross).real
    # corr[l] peaks at lag l = -shift; wrap so the search window is contiguous.
    corr = torch.roll(corr, shifts=(height // 2, width // 2), dims=(0, 1))
    cy, cx = height // 2, width // 2
    gy, gx = -int(guess[0]), -int(guess[1])
    y0, y1 = max(0, cy + gy - radius), min(height, cy + gy + radius + 1)
    x0, x1 = max(0, cx + gx - radius), min(width, cx + gx + radius + 1)
    patch = corr[y0:y1, x0:x1]
    flat = int(torch.argmax(patch))
    py, px = divmod(flat, patch.shape[1])
    lag_y, lag_x = y0 + py - cy, x0 + px - cx
    return -lag_y, -lag_x


def refine_shift(
    reference: torch.Tensor,
    moving: torch.Tensor,
    initial: Sequence[float],
    *,
    iterations: int = 12,
    margin: int = 8,
    tolerance: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Newton sub-pixel refinement of ``moving ~ reference shifted by d``.

    Returns ``(d, covariance)``.  The covariance is the least-squares one
    (residual variance times the inverse normal matrix) with an
    errors-in-variables correction: the Jacobian is built from a NOISY image,
    so the noise's own gradient energy inflates the normal matrix and would
    report false confidence along directions the scene does not constrain
    (a field of horizontal lines "constrains" x only through noise).  Half
    the gradient energy of the fitted residual estimates the reference
    frame's noise-gradient energy and is subtracted; a direction left with
    no signal energy gets a near-infinite variance, which is what lets the
    trajectory smoother fall back on the drift prior there.
    """
    height, width = reference.shape[-2:]
    device = reference.device
    ys, xs = _sample_grid(height, width, device)
    d = torch.tensor([float(initial[0]), float(initial[1])], device=device, dtype=torch.float64)
    margin = max(1, min(margin, (min(height, width) - 4) // 4))
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    mask[margin : height - margin, margin : width - margin] = True
    b = moving[0, 0]
    normal = torch.eye(2, device=device, dtype=torch.float64)
    residual_var = torch.tensor(0.0, device=device, dtype=torch.float64)
    residual_image = torch.zeros_like(b)

    def gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gy = torch.zeros_like(image)
        gx = torch.zeros_like(image)
        gy[1:-1] = (image[2:] - image[:-2]) * 0.5
        gx[:, 1:-1] = (image[:, 2:] - image[:, :-2]) * 0.5
        return gy, gx

    for _ in range(iterations):
        sample_y = (ys - d[0].float())[:, None].expand(height, width)
        sample_x = (xs - d[1].float())[None, :].expand(height, width)
        moved = _grid_sample(reference, sample_y, sample_x, mode="bicubic")[0, 0]
        # d moved / d(dy) = -(d reference / dy) at the sample point ~ -grad(moved)
        gy, gx = gradients(moved)
        residual_image = b - moved
        residual = residual_image[mask].double()
        jac = torch.stack([-gy[mask], -gx[mask]], dim=1).double()
        normal = jac.t() @ jac
        rhs = jac.t() @ residual
        try:
            step = torch.linalg.solve(normal + 1e-12 * torch.eye(2, device=device, dtype=torch.float64), rhs)
        except RuntimeError:
            break
        d = d + step
        residual_var = (residual**2).mean()
        if float(step.abs().max()) < tolerance:
            break
    # Errors-in-variables correction (see the docstring): the residual is the
    # difference of two independent noise fields (plus misfit), so half of its
    # gradient energy estimates the reference's noise-gradient energy.
    ry, rx = gradients(residual_image)
    noise_jac = torch.stack([ry[mask], rx[mask]], dim=1).double()
    signal_normal = normal - 0.5 * (noise_jac.t() @ noise_jac)
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (signal_normal + signal_normal.t()))
    floor = 1e-6 * float(torch.trace(normal)) + 1e-12
    eigenvalues = torch.clamp(eigenvalues, min=floor)
    covariance = residual_var * (eigenvectors @ torch.diag(1.0 / eigenvalues) @ eigenvectors.t())
    return d.cpu().numpy(), covariance.cpu().numpy()


# ---------------------------------------------------------------------------
# trajectory smoothing (constant-velocity Kalman filter + RTS smoother)


def smooth_trajectory(
    measurements: np.ndarray,
    covariances: np.ndarray,
    *,
    walk_sigma: float = 0.3,
    velocity_sigma: float = 0.05,
    anchor_sigma: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth per-frame shift measurements with a constant-velocity drift prior.

    State per frame: ``(py, vy, px, vx)``; ``p' = p + v + w_p``, ``v' = v + w_v``.
    Frame 0 is anchored at 0 (it defines the coordinate origin).  Returns
    ``(position [N, 2], velocity [N, 2])``.
    """
    z = np.asarray(measurements, dtype=np.float64)
    r = np.asarray(covariances, dtype=np.float64)
    n = len(z)
    if n == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    A = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
    Q = np.diag([walk_sigma**2, velocity_sigma**2, walk_sigma**2, velocity_sigma**2])
    Hm = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    x = np.zeros(4)
    P = np.diag([anchor_sigma**2, 10.0, anchor_sigma**2, 10.0])
    xs_pred, Ps_pred, xs_filt, Ps_filt = [], [], [], []
    for k in range(n):
        if k > 0:
            x = A @ x
            P = A @ P @ A.T + Q
        xs_pred.append(x.copy())
        Ps_pred.append(P.copy())
        Rk = r[k] if k > 0 else np.eye(2) * anchor_sigma**2
        zk = z[k] if k > 0 else np.zeros(2)
        # Guard: a singular / enormous covariance means "no information".
        Rk = 0.5 * (Rk + Rk.T) + np.eye(2) * 1e-9
        S = Hm @ P @ Hm.T + Rk
        K = P @ Hm.T @ np.linalg.inv(S)
        x = x + K @ (zk - Hm @ x)
        P = (np.eye(4) - K @ Hm) @ P
        xs_filt.append(x.copy())
        Ps_filt.append(P.copy())
    # RTS backward pass
    xs_smooth = [None] * n
    xs_smooth[-1] = xs_filt[-1]
    P_smooth = Ps_filt[-1]
    for k in range(n - 2, -1, -1):
        C = Ps_filt[k] @ A.T @ np.linalg.pinv(Ps_pred[k + 1])
        xs_smooth[k] = xs_filt[k] + C @ (xs_smooth[k + 1] - xs_pred[k + 1])
        P_smooth = Ps_filt[k] + C @ (P_smooth - Ps_pred[k + 1]) @ C.T
    smooth = np.stack(xs_smooth)
    position = smooth[:, [0, 2]]
    velocity = smooth[:, [1, 3]]
    position = position - position[0][None, :]
    return position, velocity


# ---------------------------------------------------------------------------
# burst registration


def register_burst(
    frames01: Sequence[np.ndarray] | np.ndarray,
    *,
    sigma: float = 2.0,
    radius: int = 6,
    device: torch.device | str = "cpu",
    predenoise: PredenoiseFn | None = None,
    smooth: bool = True,
    walk_sigma: float = 0.3,
    velocity_sigma: float = 0.05,
) -> Trajectory:
    """Register every frame of a burst to frame 0 (see the module docstring).

    ``frames01``: ``N`` full frames ``[H, W]`` in [0, 1].  Each frame is
    optionally pre-denoised, Gaussian-smoothed, coarsely matched within
    ``radius`` of the previous frame's estimate, then refined by Gauss-Newton.
    """
    device = torch.device(device)
    stack = [np.asarray(frame, dtype=np.float64) for frame in frames01]
    if not stack:
        raise ValueError("register_burst needs at least one frame")
    if predenoise is not None:
        stack = [np.asarray(predenoise(frame), dtype=np.float64) for frame in stack]
    tensor = _as_batch(np.stack(stack), device)
    smoothed = gaussian_smooth(tensor, sigma)
    reference = smoothed[0:1]
    n = len(stack)
    raw = np.zeros((n, 2))
    cov = np.zeros((n, 2, 2))
    guess = (0, 0)
    with torch.no_grad():
        for k in range(1, n):
            moving = smoothed[k : k + 1]
            integer = coarse_shift(reference, moving, radius=radius, guess=guess)
            estimate, covariance = refine_shift(reference, moving, integer)
            raw[k] = estimate
            cov[k] = covariance
            guess = (int(round(estimate[0])), int(round(estimate[1])))
    if smooth:
        position, velocity = smooth_trajectory(
            raw, cov, walk_sigma=walk_sigma, velocity_sigma=velocity_sigma
        )
    else:
        position = raw.copy()
        velocity = np.zeros_like(raw)
        if n > 1:
            velocity[:-1] = raw[1:] - raw[:-1]
            velocity[-1] = velocity[-2] if n > 2 else velocity[0]
    return Trajectory(position=position, velocity=velocity, raw_position=raw, covariance=cov)


def align_burst(
    frames01: Sequence[np.ndarray],
    trajectory: Trajectory,
    *,
    indices: Sequence[int] | None = None,
    device: torch.device | str = "cpu",
    mode: str = "bicubic",
) -> tuple[np.ndarray, np.ndarray]:
    """Warp the selected frames into frame-0 coordinates.

    Returns ``(aligned [K, H, W], valid [K, H, W])``.  Frame 0 is also
    de-sheared by its own velocity so every frame refers to frame 0's
    mid-frame coordinates.
    """
    chosen = list(range(len(frames01))) if indices is None else [int(i) for i in indices]
    aligned = []
    valid = []
    for index in chosen:
        image, mask = warp_frame(
            frames01[index],
            trajectory.position[index],
            trajectory.velocity[index],
            mode=mode,
            device=device,
        )
        aligned.append(image)
        valid.append(mask)
    return np.stack(aligned), np.stack(valid)


def fuse_mean(aligned: np.ndarray, valid: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Per-pixel mean over the frames that cover the pixel; pixels no frame
    covers (never happens for drift far below the frame size) take
    ``fallback``."""
    weight = valid.astype(np.float64)
    count = weight.sum(axis=0)
    total = (aligned * weight).sum(axis=0)
    out = np.where(count > 0, total / np.maximum(count, 1.0), fallback)
    return out


# ---------------------------------------------------------------------------
# registration tables (per dataset) and accuracy against a drift truth


REGISTRATION_FORMAT = 1


@dataclass
class RegistrationTable:
    """Trajectories of every burst of every source of a dataset."""

    method: str
    sigma: float
    radius: int
    frames_per_burst: int
    bursts: dict[int, list[Trajectory]]  # source_index -> one Trajectory per burst

    def trajectory(self, source_index: int, replica_index: int) -> tuple[Trajectory, int]:
        """``(trajectory, index within the burst)`` of a replica."""
        burst, offset = divmod(int(replica_index), self.frames_per_burst)
        return self.bursts[int(source_index)][burst], offset

    def to_json(self) -> dict:
        return {
            "format": REGISTRATION_FORMAT,
            "method": self.method,
            "sigma": self.sigma,
            "radius": self.radius,
            "frames_per_burst": self.frames_per_burst,
            "sources": {
                str(source): [trajectory.to_json() for trajectory in trajectories]
                for source, trajectories in sorted(self.bursts.items())
            },
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RegistrationTable":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != REGISTRATION_FORMAT:
            raise ValueError(f"{path} is not a format-{REGISTRATION_FORMAT} registration table")
        return cls(
            method=str(payload["method"]),
            sigma=float(payload["sigma"]),
            radius=int(payload["radius"]),
            frames_per_burst=int(payload["frames_per_burst"]),
            bursts={
                int(source): [Trajectory.from_json(item) for item in trajectories]
                for source, trajectories in payload["sources"].items()
            },
        )


def registration_errors(
    estimated: Trajectory, truth_position: np.ndarray, truth_velocity: np.ndarray
) -> dict:
    """Per-burst error statistics of a registered trajectory against the truth."""
    dp = estimated.position - np.asarray(truth_position, dtype=np.float64)
    dv = estimated.velocity - np.asarray(truth_velocity, dtype=np.float64)
    draw = estimated.raw_position - np.asarray(truth_position, dtype=np.float64)
    return {
        "position_rms": np.sqrt((dp**2).mean(axis=0)).tolist(),
        "position_max": np.abs(dp).max(axis=0).tolist(),
        "raw_position_rms": np.sqrt((draw**2).mean(axis=0)).tolist(),
        "velocity_rms": np.sqrt((dv**2).mean(axis=0)).tolist(),
        "position_errors": dp.tolist(),
    }
