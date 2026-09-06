"""Drift-robust burst fusion: the level-conditioned estimator trained on
registered frame subsets against a single raw, unregistered frame.

Why this exists.  A single frame at Poisson peak 10 does not carry the
structured fine features (stains, blemishes) that a 16-frame average shows;
the information limit is the frame's photon count, not the estimator.  The
electrons of the other frames are available -- but real bursts drift, so
neither averaging nor pairing frames is admissible without registration, and
the objection on the table is that registration of noisy frames is not
available.  :mod:`edge_denoise.register` showed it is (bounded-search
least-squares, 0.01-0.04 px in every constrained direction).  This module
builds the estimator on top of it.

Training sample (all inside ONE burst, all raw):

    subset S of m frames  --register--> aligned crops --mean--> x_S
    f_theta(x_S, t = m)   --warp by the target's residual shift-->  compared to y_h, h not in S

- ``y_h`` is one raw frame, never averaged, never resampled: the loss keeps
  the Noise2Noise property (its noise is independent of every frame in S),
  so the optimum is ``E[y_h | x_S]``, the posterior mean given the burst's
  m frames.  Drift is handled by warping the *prediction* (smooth, so
  resampling it is harmless) into the target frame's own coordinates with
  the registration's shift -- the target is read where it was measured.
- ``t = m`` conditions the network on the dose, so one network is the
  single-frame estimator at m = 1 and the burst estimator at m = 15, with
  every intermediate operating point (:class:`EdgeDenoiser` feeds t into the
  backbone's timestep embedding; the N2N teacher trained at t = 1).
- At inference (:class:`FusionDenoiser`) a K-frame burst is registered,
  aligned to its first frame, averaged, and passed through the network at
  ``t = K`` in blended tiles; the output lives in frame 0's coordinates.

Ablations are one config line each: ``align: none`` trains and infers on the
plain drifting average (what happens if you do not register), ``align:
truth`` uses the generator's recorded drift (what registration error costs),
``condition_on_level: false`` feeds t = 1 at every dose (a dose-blind
network).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from burst_diffusion.data import BurstCache, BurstSource
from burst_diffusion.ema import EMAHelper
from burst_diffusion.repeatability import RealizationProvider

from .config import Config, FusionConfig
from .data import ClipDebiaser
from .distill import denoise_full_frame
from .drift import burst_truths, load_drift_truth
from .register import (
    PredenoiseFn,
    RegistrationTable,
    Trajectory,
    align_burst,
    fuse_mean,
    register_burst,
)

ALIGN_MODES = ("registered", "none", "truth")


# ---------------------------------------------------------------------------
# crops of drifted frames in frame-0 coordinates


_CROP_MARGIN = 3  # bicubic support + 1 px of slack around every aligned crop


def _padded_patch(frame: np.ndarray, r0: int, c0: int, extent: int) -> np.ndarray:
    """``frame[r0:r0+extent, c0:c0+extent]`` with reflect padding beyond the frame."""
    height, width = frame.shape[:2]
    pad_top, pad_left = max(0, -r0), max(0, -c0)
    pad_bottom, pad_right = max(0, r0 + extent - height), max(0, c0 + extent - width)
    patch = frame[max(r0, 0) : min(r0 + extent, height), max(c0, 0) : min(c0 + extent, width)]
    if pad_top or pad_left or pad_bottom or pad_right:
        patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect")
    return patch


def crop_aligned_batch(
    frames: Sequence[np.ndarray],
    top: int,
    left: int,
    size: int,
    shifts: Sequence[tuple[float, float]],
    *,
    mode: str = "bicubic",
) -> np.ndarray:
    """``[len(frames), size, size]`` float32 in [0, 1]: frame ``k`` sampled at
    ``(top + r + dy_k, left + c + dx_k)`` -- the window's content in frame-0
    coordinates -- in ONE batched resampling call."""
    if len(frames) != len(shifts):
        raise ValueError(f"{len(frames)} frames but {len(shifts)} shifts")
    if not frames:
        return np.zeros((0, size, size), dtype=np.float32)
    extent = size + 2 * _CROP_MARGIN + 1
    patches = []
    fractional = []
    for frame, (dy, dx) in zip(frames, shifts):
        iy, ix = int(math.floor(dy)), int(math.floor(dx))
        patches.append(_padded_patch(frame, top + iy - _CROP_MARGIN, left + ix - _CROP_MARGIN, extent))
        fractional.append((dy - iy, dx - ix))
    stack = torch.from_numpy(np.stack(patches).astype(np.float32) / 255.0)[:, None]
    frac = torch.tensor(fractional, dtype=torch.float32)  # [K, 2]
    if not bool(torch.any(frac != 0.0)):
        return stack[:, 0, _CROP_MARGIN : _CROP_MARGIN + size, _CROP_MARGIN : _CROP_MARGIN + size].numpy()
    base = torch.arange(size, dtype=torch.float32) + _CROP_MARGIN
    ys = base[None, :, None] + frac[:, 0][:, None, None]  # [K, S, 1]
    xs = base[None, None, :] + frac[:, 1][:, None, None]  # [K, 1, S]
    count = len(frames)
    grid = torch.stack(
        [
            (xs / (extent - 1) * 2.0 - 1.0).expand(count, size, size),
            (ys / (extent - 1) * 2.0 - 1.0).expand(count, size, size),
        ],
        dim=-1,
    )
    out = F.grid_sample(stack, grid, mode=mode, padding_mode="reflection", align_corners=True)
    # Bicubic overshoot on noise-like frames is an interpolation artefact, not
    # information; keep the crops in the stored frames' range (as inference does).
    return out[:, 0].clamp_(0.0, 1.0).numpy()


def crop_aligned(
    frame: np.ndarray, top: int, left: int, size: int, dy: float, dx: float, *, mode: str = "bicubic"
) -> np.ndarray:
    """``[size, size]`` float32 in [0, 1]: the raw frame sampled at
    ``(top + r + dy, left + c + dx)`` -- the window's content in frame-0
    coordinates -- with reflect padding beyond the frame."""
    return crop_aligned_batch([frame], top, left, size, [(dy, dx)], mode=mode)[0]


def warp_prediction(prediction: torch.Tensor, shifts: torch.Tensor, *, mode: str = "bicubic") -> torch.Tensor:
    """Resample ``[B, 1, S, S]`` predictions so sample ``b`` reads
    ``prediction[b](r - shifts[b, 0], c - shifts[b, 1])`` -- the target
    frame's coordinates (see the module docstring).  Differentiable."""
    if prediction.dim() != 4:
        raise ValueError(f"prediction must be [B, C, S, S], got {tuple(prediction.shape)}")
    batch, _, height, width = prediction.shape
    if shifts.shape != (batch, 2):
        raise ValueError(f"shifts must be [B, 2], got {tuple(shifts.shape)}")
    if not bool(torch.any(shifts != 0.0)):
        return prediction
    device = prediction.device
    ys = torch.arange(height, device=device, dtype=torch.float32)
    xs = torch.arange(width, device=device, dtype=torch.float32)
    sample_y = ys[None, :, None] - shifts[:, 0].to(torch.float32)[:, None, None]  # [B, H, 1]
    sample_x = xs[None, None, :] - shifts[:, 1].to(torch.float32)[:, None, None]  # [B, 1, W]
    grid = torch.stack(
        [
            (sample_x / max(width - 1, 1) * 2.0 - 1.0).expand(batch, height, width),
            (sample_y / max(height - 1, 1) * 2.0 - 1.0).expand(batch, height, width),
        ],
        dim=-1,
    )
    return F.grid_sample(prediction, grid, mode=mode, padding_mode="reflection", align_corners=True)


# ---------------------------------------------------------------------------
# alignment sources: registered table, none, or the generator's truth


class AlignmentSource:
    """Per-(source, burst) trajectories from one of the ``align`` modes."""

    def __init__(
        self,
        mode: str,
        *,
        frames_per_burst: int,
        table: RegistrationTable | None = None,
        truth: dict | None = None,
    ):
        if mode not in ALIGN_MODES:
            raise ValueError(f"align mode must be one of {ALIGN_MODES}, got {mode!r}")
        if mode == "registered" and table is None:
            raise ValueError("align 'registered' needs a registration table")
        if mode == "truth" and truth is None:
            raise ValueError("align 'truth' needs the dataset's drift.json truth")
        if mode == "registered" and table is not None and table.frames_per_burst != frames_per_burst:
            raise ValueError(
                f"registration table has {table.frames_per_burst} frames per burst but the "
                f"fusion config says {frames_per_burst}"
            )
        self.mode = mode
        self.frames_per_burst = frames_per_burst
        self.table = table
        self.truth = truth

    def trajectory(self, source_index: int, burst: int) -> Trajectory:
        if self.mode == "none":
            return Trajectory.identity(self.frames_per_burst)
        if self.mode == "truth":
            assert self.truth is not None
            item = burst_truths(self.truth, source_index)[burst]
            return Trajectory(
                position=item.position,
                velocity=item.velocity,
                raw_position=item.position,
                covariance=np.zeros((len(item.position), 2, 2)),
            )
        assert self.table is not None
        try:
            return self.table.bursts[int(source_index)][burst]
        except (KeyError, IndexError):
            raise KeyError(
                f"registration table has no burst {burst} for source {source_index}; "
                "run `python -m edge_denoise register` on this dataset"
            ) from None


def _window_shift(trajectory: Trajectory, index: int, center_row: float, height: int) -> tuple[float, float]:
    frac = center_row / max(height - 1, 1) - 0.5
    shift = trajectory.position[index] + trajectory.velocity[index] * frac
    return float(shift[0]), float(shift[1])


# ---------------------------------------------------------------------------
# training batches


@dataclass
class FusionBatch:
    inputs: torch.Tensor  # [B, 1, S, S] in [-1, 1]: mean of the aligned subset
    levels: torch.Tensor  # [B] float: m, the number of frames averaged
    targets: torch.Tensor  # [B, 1, S, S] in [-1, 1]: one raw frame, its own coordinates
    shifts: torch.Tensor  # [B, 2] residual (dy, dx) the prediction is warped by
    # ``target: multi_frame`` only: further raw targets [B, T, 1, S, S] with
    # their residual shifts [B, T, 2] and a validity mask [B, T] (a sample
    # whose complement is smaller than T pads with masked entries).
    extra_targets: torch.Tensor | None = None
    extra_shifts: torch.Tensor | None = None
    extra_mask: torch.Tensor | None = None


@dataclass
class FusionValBatch(FusionBatch):
    clean: torch.Tensor | None = None  # [B, 1, S, S] in [-1, 1], frame-0 coordinates


@dataclass(frozen=True)
class FusionSampleInfo:
    source_index: int
    burst: int
    crop_yx: tuple[int, int]
    level: int
    subset: tuple[int, ...]  # frame indices within the burst
    target: int  # frame index within the burst


class FusionFactory:
    """Seeded assembler of fusion training/validation batches."""

    def __init__(
        self,
        cache: BurstCache,
        *,
        image_size: int,
        batch_size: int,
        fusion: FusionConfig,
        alignment: AlignmentSource,
        seed: int = 0,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not cache.train_sources:
            raise ValueError("cache has no training sources")
        frames = fusion.frames_per_burst
        for source in cache.all_sources:
            if len(source.frames) < frames:
                raise ValueError(
                    f"source {source.source_index} has {len(source.frames)} frames but the "
                    f"fusion config needs bursts of {frames}"
                )
            if min(source.clean.shape[:2]) < image_size + 2:
                raise ValueError(
                    f"source {source.source_index} is {source.clean.shape[:2]} but crops need "
                    f"at least {image_size + 2} px"
                )
        self.cache = cache
        self.image_size = image_size
        self.batch_size = batch_size
        self.fusion = fusion
        self.alignment = alignment
        self.levels = list(fusion.levels)
        self._rng = np.random.default_rng(seed)

    # -- helpers ------------------------------------------------------------

    def _bursts_of(self, source: BurstSource) -> int:
        return len(source.frames) // self.fusion.frames_per_burst

    def _burst_frames(self, source: BurstSource, burst: int) -> list[np.ndarray]:
        frames = self.fusion.frames_per_burst
        return source.frames[burst * frames : (burst + 1) * frames]

    def _window_range(self, trajectory: Trajectory, indices: Sequence[int], height: int, width: int) -> tuple[int, int, int, int]:
        """Allowed (top_lo, top_hi, left_lo, left_hi) so every aligned crop of
        the listed frames samples inside the frame (1 px of bicubic slack)."""
        size = self.image_size
        shifts = np.stack([trajectory.position[i] for i in indices])
        vel = np.stack([trajectory.velocity[i] for i in indices])
        # Row-dependent shifts vary by at most |velocity| over the frame.
        lo_shift = (shifts - np.abs(vel) * 0.5).min(axis=0)
        hi_shift = (shifts + np.abs(vel) * 0.5).max(axis=0)
        top_lo = int(math.ceil(max(0.0, -lo_shift[0]))) + 1
        top_hi = int(math.floor(height - size - max(0.0, hi_shift[0]))) - 1
        left_lo = int(math.ceil(max(0.0, -lo_shift[1]))) + 1
        left_hi = int(math.floor(width - size - max(0.0, hi_shift[1]))) - 1
        if top_hi < top_lo:
            top_lo = top_hi = (height - size) // 2
        if left_hi < left_lo:
            left_lo = left_hi = (width - size) // 2
        return top_lo, top_hi, left_lo, left_hi

    def _assemble_frame_target(
        self,
        source: BurstSource,
        burst: int,
        trajectory: Trajectory,
        subset: Sequence[int],
        target: int,
        top: int,
        left: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """:meth:`_assemble` with the raw-frame target regardless of ``target``."""
        size = self.image_size
        frames = self._burst_frames(source, burst)
        height = frames[0].shape[0]
        center_row = top + (size - 1) / 2.0
        shifts = [_window_shift(trajectory, index, center_row, height) for index in subset]
        crops = crop_aligned_batch([frames[index] for index in subset], top, left, size, shifts)
        mean01 = crops.mean(axis=0)
        dy, dx = _window_shift(trajectory, target, center_row, height)
        ry, rx = int(round(dy)), int(round(dx))
        target01 = crop_aligned(frames[target], top, left, size, float(ry), float(rx))
        residual = np.array([dy - ry, dx - rx], dtype=np.float32)
        return (mean01 * 2.0 - 1.0)[None], (target01 * 2.0 - 1.0)[None], residual

    def _assemble(
        self,
        source: BurstSource,
        burst: int,
        trajectory: Trajectory,
        subset: Sequence[int],
        target: int,
        top: int,
        left: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(input crop [1,S,S], target crop [1,S,S], residual shift [2]) in model range."""
        size = self.image_size
        frames = self._burst_frames(source, burst)
        height = frames[0].shape[0]
        center_row = top + (size - 1) / 2.0
        shifts = [_window_shift(trajectory, index, center_row, height) for index in subset]
        crops = crop_aligned_batch([frames[index] for index in subset], top, left, size, shifts)
        mean01 = crops.mean(axis=0)
        if self.fusion.target == "complement_mean":
            complement = [index for index in range(len(frames)) if index not in subset]
            shifts = [_window_shift(trajectory, index, center_row, height) for index in complement]
            target01 = crop_aligned_batch([frames[index] for index in complement], top, left, size, shifts).mean(axis=0)
            residual = np.zeros(2, dtype=np.float32)
        elif self.fusion.target == "clean":
            # Oracle: the clean scene is frame 0's mid-frame content (the burst
            # starts at the origin), i.e. frame-0 coordinates exactly.
            target01 = source.clean[top : top + size, left : left + size].astype(np.float32) / 255.0
            residual = np.zeros(2, dtype=np.float32)
        else:
            dy, dx = _window_shift(trajectory, target, center_row, height)
            ry, rx = int(round(dy)), int(round(dx))
            target01 = crop_aligned(frames[target], top, left, size, float(ry), float(rx))
            residual = np.array([dy - ry, dx - rx], dtype=np.float32)
        return (mean01 * 2.0 - 1.0)[None], (target01 * 2.0 - 1.0)[None], residual

    # -- batches ------------------------------------------------------------

    def _extra_targets(
        self,
        source: BurstSource,
        burst: int,
        trajectory: Trajectory,
        candidates: Sequence[int],
        top: int,
        left: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``multi_frame``: up to ``targets_per_sample - 1`` further raw targets
        (the first is the batch's main target), zero-padded with a mask."""
        size = self.image_size
        frames = self._burst_frames(source, burst)
        height = frames[0].shape[0]
        center_row = top + (size - 1) / 2.0
        slots = self.fusion.targets_per_sample - 1
        targets = np.zeros((slots, 1, size, size), dtype=np.float32)
        shifts = np.zeros((slots, 2), dtype=np.float32)
        mask = np.zeros((slots,), dtype=np.float32)
        for slot, index in enumerate(list(candidates)[:slots]):
            dy, dx = _window_shift(trajectory, index, center_row, height)
            ry, rx = int(round(dy)), int(round(dx))
            crop = crop_aligned(frames[index], top, left, size, float(ry), float(rx))
            targets[slot, 0] = crop * 2.0 - 1.0
            shifts[slot] = (dy - ry, dx - rx)
            mask[slot] = 1.0
        return targets, shifts, mask

    def sample_batch(
        self, *, count: int | None = None, return_info: bool = False
    ) -> FusionBatch | tuple[FusionBatch, list[FusionSampleInfo]]:
        count = self.batch_size if count is None else count
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        inputs, targets, shifts, levels, info = [], [], [], [], []
        extra_t, extra_s, extra_m = [], [], []
        multi = self.fusion.target == "multi_frame"
        frames = self.fusion.frames_per_burst
        for _ in range(count):
            source = self.cache.train_sources[int(self._rng.integers(len(self.cache.train_sources)))]
            burst = int(self._rng.integers(self._bursts_of(source)))
            trajectory = self.alignment.trajectory(source.source_index, burst)
            level = int(self.levels[int(self._rng.integers(len(self.levels)))])
            permutation = self._rng.permutation(frames)
            target = int(permutation[0])
            subset = tuple(int(i) for i in permutation[1 : 1 + level])
            others = [int(i) for i in permutation[1 + level :]]
            height, width = source.clean.shape[:2]
            top_lo, top_hi, left_lo, left_hi = self._window_range(
                trajectory, list(subset) + [target] + (others if multi else []), height, width
            )
            top = int(self._rng.integers(top_lo, top_hi + 1))
            left = int(self._rng.integers(left_lo, left_hi + 1))
            x, y, shift = self._assemble(source, burst, trajectory, subset, target, top, left)
            inputs.append(x)
            targets.append(y)
            shifts.append(shift)
            levels.append(float(level))
            if multi:
                t, s, m = self._extra_targets(source, burst, trajectory, others, top, left)
                extra_t.append(t)
                extra_s.append(s)
                extra_m.append(m)
            if return_info:
                info.append(
                    FusionSampleInfo(
                        source_index=source.source_index,
                        burst=burst,
                        crop_yx=(top, left),
                        level=level,
                        subset=subset,
                        target=target,
                    )
                )
        batch = FusionBatch(
            inputs=torch.from_numpy(np.stack(inputs)),
            levels=torch.tensor(levels, dtype=torch.float32),
            targets=torch.from_numpy(np.stack(targets)),
            shifts=torch.from_numpy(np.stack(shifts)),
            extra_targets=torch.from_numpy(np.stack(extra_t)) if multi else None,
            extra_shifts=torch.from_numpy(np.stack(extra_s)) if multi else None,
            extra_mask=torch.from_numpy(np.stack(extra_m)) if multi else None,
        )
        if return_info:
            return batch, info
        return batch

    def val_batch(self, *, count: int, level: int | None = None) -> FusionValBatch:
        """Deterministic validation: burst 0 of each val source, center crop,
        the first ``level`` frames after frame 0 as the subset, frame 0 (the
        burst's origin, so its residual shift is zero) as the target."""
        if not self.cache.val_sources:
            raise ValueError("cache has no validation sources")
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        level = max(self.levels) if level is None else int(level)
        if not 1 <= level < self.fusion.frames_per_burst:
            raise ValueError(f"level must be in [1, {self.fusion.frames_per_burst - 1}], got {level}")
        size = self.image_size
        inputs, targets, shifts, levels, cleans = [], [], [], [], []
        for index in range(count):
            source = self.cache.val_sources[index % len(self.cache.val_sources)]
            trajectory = self.alignment.trajectory(source.source_index, 0)
            height, width = source.clean.shape[:2]
            top, left = (height - size) // 2, (width - size) // 2
            subset = tuple(range(1, 1 + level))
            # Validation always reads the raw frame 0 (the burst's origin) as
            # the loss target so the two target modes stay comparable.
            x, y, shift = self._assemble_frame_target(source, 0, trajectory, subset, 0, top, left)
            inputs.append(x)
            targets.append(y)
            shifts.append(shift)
            levels.append(float(level))
            clean = source.clean[top : top + size, left : left + size].astype(np.float32) / 255.0
            cleans.append((clean * 2.0 - 1.0)[None])
        return FusionValBatch(
            inputs=torch.from_numpy(np.stack(inputs)),
            levels=torch.tensor(levels, dtype=torch.float32),
            targets=torch.from_numpy(np.stack(targets)),
            shifts=torch.from_numpy(np.stack(shifts)),
            clean=torch.from_numpy(np.stack(cleans)),
        )

    def state_dict(self) -> dict:
        return {"rng_state": self._rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self._rng.bit_generator.state = state["rng_state"]


def build_alignment(config: Config, *, dataset_dir: str | Path | None = None) -> AlignmentSource:
    """The alignment source a fusion config asks for."""
    fusion = config.objective.fusion
    if fusion is None:
        raise ValueError("config has no objective.fusion block")
    root = Path(dataset_dir if dataset_dir is not None else config.data.dataset_dir)
    table = None
    truth = None
    if fusion.align == "registered":
        if fusion.registration is None:
            raise ValueError("objective.fusion.registration is required for align 'registered'")
        table = RegistrationTable.load(fusion.registration)
    elif fusion.align == "truth":
        truth = load_drift_truth(root)
        if truth is None:
            raise ValueError(f"align 'truth' needs {root / 'drift.json'}")
    return AlignmentSource(fusion.align, frames_per_burst=fusion.frames_per_burst, table=table, truth=truth)


# ---------------------------------------------------------------------------
# inference


class FusionDenoiser:
    """Register, align, average and denoise a burst at its own dose level."""

    def __init__(self, model, *, config: Config, device: torch.device):
        self.model = model.to(device).eval()
        self.config = config
        self.device = device
        fusion = config.objective.fusion
        self.fusion = fusion if fusion is not None else FusionConfig()
        self._debias = (
            ClipDebiaser(self.fusion.debias_peak) if self.fusion.debias_peak is not None else None
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, device: str = "auto", use_ema: bool = True) -> "FusionDenoiser":
        from .model import build_model
        from .train import load_checkpoint, resolve_device

        resolved = resolve_device(device)
        payload = load_checkpoint(path, map_location=resolved)
        config = Config.model_validate(payload["config"])
        model = build_model(config)
        model.load_state_dict(payload["model"])
        if use_ema and payload.get("ema") is not None:
            helper = EMAHelper()
            helper.load_state_dict(payload["ema"])
            named = dict(model.named_parameters())
            for name, value in helper.shadow.items():
                named[name].data.copy_(value)
        return cls(model, config=config, device=resolved)

    @property
    def image_size(self) -> int:
        return self.config.data.image_size

    def level_for(self, count: int) -> float:
        """The conditioning value for a ``count``-frame average."""
        if not self.fusion.condition_on_level:
            return 1.0
        if self.fusion.level_cap is not None:
            return float(min(count, self.fusion.level_cap))
        return float(count)

    def denoise_mean(self, mean01: np.ndarray, count: int, *, stride: int = 48, tile_batch: int = 64) -> np.ndarray:
        """Denoise a full-frame ``count``-frame mean (``[H, W]`` in [0, 1])."""
        level = self.level_for(count)

        def fn(tiles: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                batch = tiles.to(self.device)
                t = torch.full((batch.shape[0],), level, device=self.device)
                out = self.model.predict_image(batch, t=t)
            return out.clamp(-1.0, 1.0).cpu()

        out = denoise_full_frame(
            fn, mean01, tile=self.image_size, stride=min(stride, self.image_size), tile_batch=tile_batch
        )
        if self._debias is not None:
            out = self._debias(np.clip(out, 0.0, 1.0)).astype(np.float64)
        return out

    def register(
        self, frames01: Sequence[np.ndarray], *, predenoise: PredenoiseFn | None = None
    ) -> Trajectory:
        return register_burst(
            frames01,
            sigma=self.fusion.registration_sigma if predenoise is None else self.fusion.predenoised_sigma,
            radius=self.fusion.registration_radius,
            device=self.device,
            predenoise=predenoise,
        )

    def registered_mean(
        self, frames01: Sequence[np.ndarray], count: int, trajectory: Trajectory
    ) -> np.ndarray:
        aligned, valid = align_burst(frames01, trajectory, indices=range(count), device=self.device)
        return fuse_mean(aligned, valid, np.asarray(frames01[0], dtype=np.float64))

    def fuse(
        self,
        frames01: Sequence[np.ndarray],
        count: int,
        *,
        trajectory: Trajectory | None = None,
        predenoise: PredenoiseFn | None = None,
        stride: int = 48,
    ) -> np.ndarray:
        """Fused image of the first ``count`` frames of a burst, in frame-0
        coordinates, ``[H, W]`` in [0, 1]."""
        if count < 1 or count > len(frames01):
            raise ValueError(f"count must be in [1, {len(frames01)}], got {count}")
        if trajectory is None:
            trajectory = self.register(frames01[:count], predenoise=predenoise)
        mean01 = self.registered_mean(frames01, count, trajectory)
        return self.denoise_mean(mean01, count, stride=stride)


# ---------------------------------------------------------------------------
# evaluation providers (repeatability harness, fine-feature diagnostic)


BurstArm = Callable[[BurstSource, int], np.ndarray]  # (source, retake) -> full-frame [H, W] in [0, 1]


class BurstFusionArms:
    """Fusion arms over the retakes of a source: ``fuse{K}`` (+ ``regavg{K}``).

    A retake ``r`` is the burst of ``frames_per_retake`` consecutive replicas
    starting at ``r * frames_per_retake``; every method uses its first ``K``
    frames, registered once per retake and shared across K.
    """

    def __init__(
        self,
        denoiser: FusionDenoiser,
        *,
        frame_counts: Sequence[int],
        frames_per_retake: int,
        align: str = "registered",
        regavg: bool = True,
        predenoise: PredenoiseFn | None = None,
        truth: dict | None = None,
        stride: int = 48,
    ):
        if align not in ALIGN_MODES:
            raise ValueError(f"align must be one of {ALIGN_MODES}, got {align!r}")
        if align == "truth" and truth is None:
            raise ValueError("align 'truth' needs the dataset's drift truth")
        counts = sorted({int(c) for c in frame_counts})
        if not counts or counts[0] < 1 or counts[-1] > frames_per_retake:
            raise ValueError(f"frame counts must lie in [1, {frames_per_retake}], got {counts}")
        self.denoiser = denoiser
        self.counts = counts
        self.frames_per_retake = frames_per_retake
        self.align = align
        self.regavg = regavg
        self.predenoise = predenoise
        self.truth = truth
        self.stride = stride
        self._trajectories: dict[tuple[int, int], Trajectory] = {}

    @property
    def method_names(self) -> tuple[str, ...]:
        names = [f"fuse{count}" for count in self.counts]
        if self.regavg:
            names += [f"regavg{count}" for count in self.counts]
        return tuple(names)

    def retake_frames(self, source: BurstSource, retake: int) -> list[np.ndarray]:
        start = retake * self.frames_per_retake
        stop = start + self.frames_per_retake
        if stop > len(source.frames):
            raise ValueError(
                f"source {source.source_index} has {len(source.frames)} frames; retake {retake} "
                f"needs frames {start}..{stop - 1}"
            )
        return [frame.astype(np.float64) / 255.0 for frame in source.frames[start:stop]]

    def trajectory(self, source: BurstSource, retake: int, frames01: Sequence[np.ndarray]) -> Trajectory:
        key = (source.source_index, retake)
        if key not in self._trajectories:
            if self.align == "none":
                self._trajectories[key] = Trajectory.identity(len(frames01))
            elif self.align == "truth":
                assert self.truth is not None
                item = burst_truths(self.truth, source.source_index)[retake]
                self._trajectories[key] = Trajectory(
                    position=item.position,
                    velocity=item.velocity,
                    raw_position=item.position,
                    covariance=np.zeros((len(item.position), 2, 2)),
                )
            else:
                self._trajectories[key] = self.denoiser.register(frames01, predenoise=self.predenoise)
        return self._trajectories[key]

    def outputs(self, source: BurstSource, retake: int) -> dict[str, np.ndarray]:
        """Every method's full-frame output for one retake."""
        frames01 = self.retake_frames(source, retake)
        trajectory = self.trajectory(source, retake, frames01)
        result: dict[str, np.ndarray] = {}
        for count in self.counts:
            mean01 = self.denoiser.registered_mean(frames01, count, trajectory)
            result[f"fuse{count}"] = self.denoiser.denoise_mean(mean01, count, stride=self.stride)
            if self.regavg:
                result[f"regavg{count}"] = mean01
        return result

    def burst_arms(self) -> dict[str, BurstArm]:
        """Per-method callables for the fine-feature diagnostic."""
        cache: dict[tuple[int, int], dict[str, np.ndarray]] = {}

        def make(name: str) -> BurstArm:
            def arm(source: BurstSource, retake: int) -> np.ndarray:
                key = (source.source_index, retake)
                if key not in cache:
                    cache[key] = self.outputs(source, retake)
                return cache[key][name]

            return arm

        return {name: make(name) for name in self.method_names}

    def provider(self) -> RealizationProvider:
        """A source-aware realization provider for the repeatability harness."""

        def generate(seeds01: list[np.ndarray]) -> dict[str, list[np.ndarray]]:
            raise RuntimeError("burst fusion arms need source access; the harness must call generate_source")

        def generate_source(
            source: BurstSource, window: tuple[slice, slice], num_seeds: int
        ) -> dict[str, list[np.ndarray]]:
            produced: dict[str, list[np.ndarray]] = {name: [] for name in self.method_names}
            for retake in range(num_seeds):
                outputs = self.outputs(source, retake)
                for name, image in outputs.items():
                    produced[name].append(np.ascontiguousarray(image[window])[:, :, None])
            return produced

        return RealizationProvider(
            method_names=self.method_names, generate=generate, generate_source=generate_source
        )


def frames_per_retake_of(dataset_dir: str | Path, default: int = 1) -> int:
    """``frames_per_burst`` of a drifting dataset, else ``default``."""
    truth = load_drift_truth(dataset_dir)
    return int(truth["frames_per_burst"]) if truth is not None else default
