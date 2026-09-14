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

The backbone carries attention blocks, so full frames cannot go through in
one pass (and the estimator is only trusted at its training resolution anyway
-- see infer.py).  Frames are therefore covered with overlapping
training-resolution tiles blended by a separable raised-cosine window that
VANISHES at the tile edges.  A tile's outermost pixels are its least reliable
ones: every convolution in the backbone zero-pads, so they are predicted with
half their context missing, and on real data the loss never supervised the
outer ``LOSS_MARGIN`` pixels at all.  They must therefore fade in from nothing
wherever another tile overlaps.  The previous window had a 0.1 floor, which
let them enter abruptly at ~9% weight and drew visible lines along every tile
boundary of a 1024x1024 frame (2026-09-14).

``margin`` additionally excludes an outer ring outright, frame-aware: a tile
side flush with the frame border keeps the tile's own edge predictions, the
only estimate of those pixels.  The frame itself is never padded -- mirrored
or replicated extensions are out of distribution for a network trained on
zero-padded crops and measured WORSE in the frame's outer band than the
network's own border behavior.  With ``stride <= tile - 2 * margin`` every
pixel is a convex combination of tile predictions, an identity denoiser
round-trips exactly, and a denoiser that corrupts its tile borders leaves no
step behind (unit-tested); ``stride = tile // 2`` is the balanced
constant-overlap-add cross-fade.
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
    """Raised cosine that vanishes at the tile edges but stays positive.

    ``np.hanning(tile + 2)[1:-1]``: the first and last weight are
    ``~(pi / (tile + 1))^2`` -- 4e-5 for a 512 tile -- so a tile's outermost
    pixels contribute nothing visible wherever another tile overlaps, yet a
    frame-border pixel covered only by a tile edge still normalizes to that
    tile's own prediction instead of 0/0.  Two such windows offset by half a
    tile sum to ~1 (constant overlap-add), the balanced cross-fade.
    """
    return np.hanning(tile + 2)[1:-1]


def _positions(extent: int, tile: int, stride: int) -> list[int]:
    stops = list(range(0, extent - tile + 1, stride))
    if stops[-1] != extent - tile:
        stops.append(extent - tile)
    return stops


def _axis_profile(tile: int, margin: int, *, flush_low: bool, flush_high: bool) -> np.ndarray:
    """1-D weights along one axis of a tile.

    ``margin`` pixels at either end get weight 0 (excluded outright) and the
    remaining span a raised cosine that vanishes at the margin -- unless that
    end is flush with the frame border, where the tile's own edge predictions
    are the only estimate available and keep the full-tile profile.  Both
    profiles peak at ~1 mid-tile, so the halves join continuously.
    """
    full = _window1d(tile)
    if margin == 0:
        return full
    profile = np.zeros(tile, dtype=np.float64)
    profile[margin : tile - margin] = _window1d(tile - 2 * margin)
    half = tile // 2
    if flush_low:
        profile[:half] = full[:half]
    if flush_high:
        profile[half:] = full[half:]
    return profile


def denoise_full_frame(
    denoise_fn: DenoiseFn,
    frame01: np.ndarray,
    *,
    tile: int,
    stride: int,
    tile_batch: int = 64,
    margin: int = 0,
) -> np.ndarray:
    """Denoise a full ``[H, W]`` float frame in [0, 1] by blended tiling.

    ``stride`` is the tile spacing (``tile // 2`` recommended); ``tile_batch``
    only sets how many tiles share one forward pass and does not change the
    result; ``margin`` excludes that many outer pixels of every tile from the
    blend except on sides flush with the frame border (see module docstring).
    ``stride`` may not exceed ``tile - 2 * margin``, or some pixels would fall
    in no tile's valid region.
    """
    if frame01.ndim != 2:
        raise ValueError(f"frame01 must be [H, W], got shape {frame01.shape}")
    height, width = frame01.shape
    if min(height, width) < tile:
        raise ValueError(f"frame {frame01.shape} is smaller than the tile size {tile}")
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    if not 1 <= stride <= tile - 2 * margin:
        raise ValueError(
            f"stride must be in [1, tile - 2 * margin] = [1, {tile - 2 * margin}] so every "
            f"pixel lies in some tile's valid region, got {stride} (tile {tile}, margin {margin})"
        )
    windows: dict[tuple[bool, bool, bool, bool], np.ndarray] = {}

    def window_for(top: int, left: int) -> np.ndarray:
        flush = (top == 0, top + tile == height, left == 0, left + tile == width)
        if flush not in windows:
            windows[flush] = np.outer(
                _axis_profile(tile, margin, flush_low=flush[0], flush_high=flush[1]),
                _axis_profile(tile, margin, flush_low=flush[2], flush_high=flush[3]),
            )
        return windows[flush]

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
            window = window_for(top, left)
            accumulator[top : top + tile, left : left + tile] += tile01 * window
            weight_sum[top : top + tile, left : left + tile] += window
    if not (weight_sum > 0.0).all():
        raise RuntimeError("tile blend left pixels with zero weight; this is a bug")
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
