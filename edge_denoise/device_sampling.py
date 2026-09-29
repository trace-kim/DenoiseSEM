"""Bounded, batched native-frame sampling with the training sampler's semantics."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def sample_native_crops(frames: Sequence[np.ndarray], matrices: np.ndarray, size: int,
                        black: float, white: float, device: str | torch.device,
                        *, memory_bytes: int = 256 * 1024**2, normalize: bool = True,
                        require_overlap: bool = True, interpolation: str = "bicubic"
                        ) -> Iterator[tuple[np.ndarray, torch.Tensor, torch.Tensor]]:
    """Yield indices, normalized crops and cubic-support masks on ``device``.

    Translation retains sample_region's edge padding, float32 grid and exact
    integer crops. Affine retains OpenCV's zero border and 1/32-pixel table
    coordinates (10-bit intermediate rounding), not an unquantized warp.
    Power-of-two affine grid denominators keep those coordinates exact in FP32.
    The byte bound includes conservative grid/normalization scratch space.
    """
    device = torch.device(device)
    if interpolation not in ("bicubic", "nearest"):
        raise ValueError("sampling interpolation must be bicubic or nearest")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA sampling requested but unavailable: {device}")
    matrices = np.asarray(matrices, dtype=np.float64)
    if not len(frames) or matrices.shape != (len(frames), 2, 3) or not np.isfinite(matrices).all():
        raise ValueError("sampling requires frames and finite 2x3 matrices")
    if size < 1 or not np.isfinite([black, white]).all() or white <= black:
        raise ValueError("invalid crop size or normalization")
    groups: dict[str, list[tuple]] = {"integer": [], "translation": [], "affine": []}
    for i, (frame, matrix) in enumerate(zip(frames, matrices)):
        h, w = frame.shape
        shift = np.array_equal(matrix[:, :2], np.eye(2))
        top, left = matrix[1, 2], matrix[0, 2]
        if shift and top == int(top) and left == int(left) and 0 <= top <= h - size and 0 <= left <= w - size:
            groups["integer"].append((i, int(top), int(left), size, size))
        elif shift:
            groups["translation"].append((i, int(np.floor(top)) - 2, int(np.floor(left)) - 2, size + 5, size + 5))
        else:
            corners = matrix @ np.array([[0, size - 1, 0, size - 1], [0, 0, size - 1, size - 1], [1, 1, 1, 1]])
            x0, y0 = np.maximum(0, np.floor(corners.min(axis=1)).astype(int) - 2)
            x1, y1 = np.minimum([w, h], np.ceil(corners.max(axis=1)).astype(int) + 3)
            if x1 <= x0 or y1 <= y0:
                if require_overlap:
                    raise ValueError("target has no valid overlap with this input crop")
                y0, x0, y1, x1 = 0, 0, 1, 1  # Only zero-support pixels can sample this placeholder.
            groups["affine"].append((i, y0, x0, y1 - y0, x1 - x0))
    for kind, rows in groups.items():
        if not rows:
            continue
        ph, pw = max(r[3] for r in rows), max(r[4] for r in rows)
        if kind == "affine":
            ph, pw = (2 ** (int(v) - 1).bit_length() + 1 for v in (ph, pw))
        per_frame = ph * pw * 20 + size * size * 64
        count = memory_bytes // per_frame
        if count < 1:
            raise ValueError("one warped crop exceeds the sampling memory budget")
        for start in range(0, len(rows), count):
            chunk = rows[start:start + count]
            indices = np.array([r[0] for r in chunk])
            dtype = torch.from_numpy(np.empty(0, dtype=frames[indices[0]].dtype)).dtype
            host = torch.zeros((len(chunk), 1, ph, pw), dtype=dtype, pin_memory=device.type == "cuda")
            destination = host.numpy()
            for k, (i, y0, x0, height, width) in enumerate(chunk):
                frame = frames[i]
                h, w = frame.shape
                if kind == "translation":
                    # Edge padding includes out-of-frame crops; no frame-sized copy.
                    ys = np.clip(np.arange(y0, y0 + height), 0, h - 1)
                    xs = np.clip(np.arange(x0, x0 + width), 0, w - 1)
                    destination[k, 0] = frame[np.ix_(ys, xs)]
                else:
                    destination[k, 0, :height, :width] = frame[y0:y0 + height, x0:x0 + width]
            patches = host.to(device, non_blocking=device.type == "cuda").float()
            if normalize:
                patches.sub_(black).div_(white - black).clamp_(0, 1)
            matrix = torch.as_tensor(matrices[indices], device=device, dtype=torch.float64)
            base = torch.arange(size, device=device, dtype=torch.float64)
            xx, yy = base[None, None, :], base[None, :, None]
            xs = matrix[:, 0, 0, None, None] * xx + matrix[:, 0, 1, None, None] * yy + matrix[:, 0, 2, None, None]
            ys = matrix[:, 1, 0, None, None] * xx + matrix[:, 1, 1, None, None] * yy + matrix[:, 1, 2, None, None]
            shapes = torch.tensor([frames[i].shape for i in indices], device=device)
            valid = (ys >= 2) & (ys <= shapes[:, 0, None, None] - 3) & (xs >= 2) & (xs <= shapes[:, 1, None, None] - 3)
            if require_overlap and not valid.flatten(1).any(1).all():
                raise ValueError("target has no valid overlap with this input crop")
            if kind == "integer":
                sampled = patches
            else:
                if kind == "translation":
                    # Match the operation order of sample_region, including its
                    # float32 grid roundoff; integer crops bypass interpolation.
                    base32 = torch.arange(size, dtype=torch.float32)
                    top = torch.tensor([matrices[i, 1, 2] for i in indices], dtype=torch.float32)
                    left = torch.tensor([matrices[i, 0, 2] for i in indices], dtype=torch.float32)
                    gy = ((base32[None] + 2 + top[:, None]) - top.floor()[:, None]) / (ph - 1) * 2 - 1
                    gx = ((base32[None] + 2 + left[:, None]) - left.floor()[:, None]) / (pw - 1) * 2 - 1
                    # CPU and CUDA grid_sample differ in FP32 coordinate FMA.
                    # Recover the reference's pixel coordinates on these tiny
                    # 1-D axes, then sample in double to preserve them on both.
                    gy = (((gy + 1) / 2) * (ph - 1)).double().to(device) * (2 / (ph - 1)) - 1
                    gx = (((gx + 1) / 2) * (pw - 1)).double().to(device) * (2 / (pw - 1)) - 1
                    patches = patches.double()
                    gx, gy = gx[:, None, :].expand(-1, size, -1), gy[:, :, None].expand(-1, -1, size)
                else:
                    origins = torch.tensor([[r[2], r[1]] for r in chunk], device=device)
                    local = matrix.clone()
                    local[:, :, 2] -= origins
                    # OpenCV imgwarp.cpp: AB_BITS=10, INTER_BITS=5.
                    coords = []
                    for axis, extent in ((0, pw), (1, ph)):
                        delta = torch.round(local[:, axis, 0, None, None] * xx * 1024)
                        origin = torch.round((local[:, axis, 1, None, None] * yy + local[:, axis, 2, None, None]) * 1024)
                        pixel = torch.floor((delta + origin + 16) / 32).float() / 32
                        coords.append(pixel * (2 / (extent - 1)) - 1)
                    gx, gy = coords
                    # Padding must be zero AFTER normalization, including black < 0.
                    for k, (_, _, _, height, width) in enumerate(chunk):
                        patches[k, :, height:, :] = 0
                        patches[k, :, :, width:] = 0
                sampled = F.grid_sample(patches, torch.stack((gx, gy), dim=-1), mode=interpolation, align_corners=True).float()
            yield indices, sampled, valid[:, None]
