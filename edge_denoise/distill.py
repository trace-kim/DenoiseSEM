"""Per-scene distillation targets: the average of every replica's denoised image.

This implements the "answer sheet" of the two-phase proposal: run a frozen
phase-1 teacher over all N noisy replicas of a scene, average the outputs, and
store one float32 image per source for ``objective.gradient_target: file``.
Because the Sobel operator is linear, averaging the images IS averaging the
Sobel maps -- the target is stored in the image domain and the trainer applies
``sobel`` to its crops exactly as it does for every other target mode.

Statistics, stated up front so the arm is read honestly: averaging N teacher
outputs shrinks the teacher's noise-driven jitter by ~sqrt(N) but preserves its
systematic bias exactly, so this target is the teacher's mean behavior, not the
clean image.  The unbiased siblings (``gradient_target: clean`` /
``noisy_mean``) exist precisely to price that bias; see
docs/edge_denoise_method.md and the target-ladder report.

The backbone carries attention blocks, so full 512x512 frames cannot go
through in one pass (and the estimator is only trusted at its training
resolution anyway -- see infer.py).  Frames are therefore covered with
overlapping training-resolution tiles blended by a separable raised-cosine
window with a floor (the floor keeps frame borders, covered only by tile
edges where a plain Hann window is zero, at nonzero total weight).  With any
stride < tile the blend is seam-free in the sense that every pixel is a convex
combination of tile predictions; an identity denoiser round-trips exactly
(unit-tested).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from burst_diffusion.data import BurstCache
from burst_diffusion.provenance import file_sha256

from .config import Config

TARGETS_MANIFEST_NAME = "distill_targets.json"

#: frames [B, 1, S, S] in [-1, 1] -> denoised images [B, 1, S, S] in [-1, 1], CPU.
DenoiseFn = Callable[[torch.Tensor], torch.Tensor]


def build_teacher(
    path: str | Path, *, device: str = "auto", use_ema: bool = True
) -> tuple[DenoiseFn, str]:
    """A one-shot denoise callable from either pipeline's checkpoint.

    An edge_denoise checkpoint denoises through :class:`~edge_denoise.infer.Denoiser`;
    a burst_diffusion checkpoint through its sampler's one-shot prediction at
    the top level (``schedule=[num_steps]`` -- for the N2N arm, ``[1]``).
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"{path} is not a training checkpoint (no 'model' state)")
    if payload.get("kind") == "edge_denoise":
        from .infer import Denoiser

        denoiser = Denoiser.from_checkpoint(path, device=device, use_ema=use_ema)
        return denoiser.denoise, f"edge_denoise checkpoint (step {payload.get('step')})"
    from burst_diffusion.sample import Sampler

    sampler = Sampler.from_checkpoint(path, device=device, use_ema=use_ema)

    def denoise(frames: torch.Tensor) -> torch.Tensor:
        return sampler.run(frames, schedule=[sampler.num_steps]).prediction

    description = (
        f"burst_diffusion checkpoint (step {payload.get('step')}, one-shot "
        f"prediction at num_steps={sampler.num_steps})"
    )
    return denoise, description


def _window1d(tile: int) -> np.ndarray:
    """Raised cosine with a 0.1 floor (see module docstring for why a floor)."""
    return 0.1 + 0.9 * np.hanning(tile)


def _positions(extent: int, tile: int, stride: int) -> list[int]:
    stops = list(range(0, extent - tile + 1, stride))
    if stops[-1] != extent - tile:
        stops.append(extent - tile)
    return stops


def denoise_full_frame(
    denoise_fn: DenoiseFn,
    frame01: np.ndarray,
    *,
    tile: int,
    stride: int,
    tile_batch: int = 64,
) -> np.ndarray:
    """Denoise a full ``[H, W]`` float frame in [0, 1] by blended tiling."""
    if frame01.ndim != 2:
        raise ValueError(f"frame01 must be [H, W], got shape {frame01.shape}")
    height, width = frame01.shape
    if min(height, width) < tile:
        raise ValueError(f"frame {frame01.shape} is smaller than the tile size {tile}")
    if not 1 <= stride <= tile:
        raise ValueError(f"stride must be in [1, tile], got {stride} (tile {tile})")
    window = np.outer(_window1d(tile), _window1d(tile))
    accumulator = np.zeros((height, width), dtype=np.float64)
    weight_sum = np.zeros((height, width), dtype=np.float64)
    coordinates = [
        (top, left)
        for top in _positions(height, tile, stride)
        for left in _positions(width, tile, stride)
    ]
    for start in range(0, len(coordinates), tile_batch):
        chunk = coordinates[start : start + tile_batch]
        tiles = np.stack(
            [frame01[top : top + tile, left : left + tile] for top, left in chunk]
        ).astype(np.float32)
        prediction = denoise_fn(torch.from_numpy(tiles[:, None] * 2.0 - 1.0))
        tiles01 = ((prediction.numpy()[:, 0] + 1.0) / 2.0).astype(np.float64)
        for (top, left), tile01 in zip(chunk, tiles01):
            accumulator[top : top + tile, left : left + tile] += tile01 * window
            weight_sum[top : top + tile, left : left + tile] += window
    return accumulator / weight_sum


def write_distill_targets(
    config: Config,
    *,
    denoise_fn: DenoiseFn,
    out_dir: str | Path,
    splits: Sequence[str] = ("train", "val"),
    stride: int = 48,
    tile_batch: int = 64,
    teacher_path: str | Path | None = None,
    teacher_description: str | None = None,
    command: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Average-denoise every replica of every source in ``splits``; write one
    ``{source_index:05d}.npy`` (float32, [0, 1]) per source plus a manifest.

    The split machinery is the audited content-group split (``BurstCache``
    with the config's own data settings), so target coverage matches exactly
    the sources a training run with this config can touch.  The locked test
    split is refused: distillation targets exist for training, and the test
    sources must never enter any training input.
    """
    allowed = {"train", "val"}
    unknown = [split for split in splits if split not in allowed]
    if unknown:
        raise ValueError(f"splits must be among {sorted(allowed)}, got {unknown}")
    tile = config.data.image_size
    cache = BurstCache(
        config.data.dataset_dir,
        channels=config.data.channels,
        min_replicas=config.min_replicas,
        min_size=tile,
        val_fraction=config.data.val_fraction,
        test_fraction=config.data.test_fraction,
        split_seed=config.data.split_seed,
    )
    by_split = {"train": cache.train_sources, "val": cache.val_sources}
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)

    written: dict[str, list[int]] = {}
    for split in splits:
        written[split] = []
        for source in by_split[split]:
            accumulator: np.ndarray | None = None
            for frame in source.frames:
                denoised = denoise_full_frame(
                    denoise_fn,
                    frame.astype(np.float64) / 255.0,
                    tile=tile,
                    stride=stride,
                    tile_batch=tile_batch,
                )
                accumulator = denoised if accumulator is None else accumulator + denoised
            assert accumulator is not None
            mean = np.clip(accumulator / len(source.frames), 0.0, 1.0).astype(np.float32)
            np.save(destination / f"{source.source_index:05d}.npy", mean)
            written[split].append(source.source_index)
            if progress is not None:
                progress(f"{split} source {source.source_index} ({len(source.frames)} replicas)")

    manifest = {
        "teacher": str(teacher_path) if teacher_path is not None else None,
        "teacher_sha256": file_sha256(Path(teacher_path)) if teacher_path is not None else None,
        "teacher_description": teacher_description,
        "dataset_dir": str(config.data.dataset_dir),
        "tile": tile,
        "stride": stride,
        "splits": {split: indices for split, indices in written.items()},
        "created_unix": time.time(),
        "command": command,
    }
    (destination / TARGETS_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
