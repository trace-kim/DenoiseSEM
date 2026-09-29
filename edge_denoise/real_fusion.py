"""Burst fusion on prepared real SEM repeats (``objective.fusion.align: matched``).

The burst-diffusion training objective without its iterative sampler: the
input is the registered mean of ``m`` consecutive acquisitions, the network is
told ``t = m``, and the target is ONE other acquisition of the same site.  By
the Noise2Noise argument the optimum is ``E[clean | m-frame mean]``, so one
network serves every dose from a single frame (``m = 1``, exactly the matched
real N2N pair) to ``max(levels)`` frames.

Geometry and brightness are the inline matching measurements of
:class:`~edge_denoise.real_matching.MatchedRealPairFactory`, unchanged:

- the mean is formed in the native coordinates of the subset's first frame
  with usable geometry (the anchor, never resampled); every other member is
  sampled once into that crop and percentile-mapped to the anchor's
  brightness,
- the target is sampled and brightness-matched into the same crop exactly as
  the real N2N target, and
- a pixel where a member has no cubic support averages the members that do;
  the loss (and, at inference, contour measurement) uses only pixels every
  member supports.

The same :func:`registered_mean` builds training crops and full inference
frames, so a deployed m-frame input is the training input.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch

from burst_diffusion.data import BurstCache, BurstSource
from burst_diffusion.real_data import normalize_native

from .config import Config
from .data import PairBatch, ValPairBatch
from .device_sampling import sample_native_crops
from .real_data import LOSS_MARGIN, fixed_windows, registration_contrast
from .real_matching import (PERCENTILES, MatchedRealPairFactory, brightness_to_anchor, crop_matrix)
from .timing import TrainingTimings

#: Default full-frame tile for building inference inputs (memory bound only).
MEAN_TILE = 512


@dataclass(frozen=True)
class FusionInputSpec:
    """One training/validation input: members of a site, crop, anchor mapping."""

    source_index: int
    members: tuple[int, ...]  # consecutive acquisition indices within the site
    anchor: int  # position of the anchor within ``members``
    origin: tuple[int, int]  # crop top-left in the anchor's native coordinates
    size: int
    matrices: np.ndarray  # [m, 2, 3] anchor full-frame coords -> member coords (anchor: identity)
    brightness: np.ndarray  # [m, 2] member unit intensity -> anchor: gain, offset


@dataclass(frozen=True)
class RealFusionInfo:
    """Provenance of one sample, exposed for tests and debugging."""

    source_index: int
    crop_yx: tuple[int, int]
    members: tuple[int, ...]
    anchor_replica: int
    target_replica: int
    level: int


@torch.no_grad()
def registered_mean(frames: Sequence[np.ndarray], anchor: int, matrices: np.ndarray, brightness: np.ndarray,
                    origin: tuple[int, int], size: int, black: float, white: float,
                    device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean of ``frames`` in a ``size`` crop of the anchor's native coordinates.

    Returns the unit-intensity mean ``[size, size]`` (float32) and the mask of
    pixels every member supports. The anchor is read exactly; other members
    use the training sampler (cubic, OpenCV affine coordinates) and their
    brightness mapping. Where a member lacks support the pixel averages the
    members that have it, so the value stays a real frame average.
    """
    device = torch.device(device)
    matrices, brightness = np.asarray(matrices, dtype=np.float64), np.asarray(brightness, dtype=np.float64)
    if not 0 <= anchor < len(frames) or matrices.shape != (len(frames), 2, 3) or brightness.shape != (len(frames), 2):
        raise ValueError("registered mean needs one 2x3 matrix and one gain/offset per member")
    top, left = origin
    crop = np.asarray(frames[anchor][top:top + size, left:left + size])
    if crop.shape != (size, size):
        raise ValueError(f"anchor crop at {origin} does not fit a {size}px window")
    total = torch.from_numpy(normalize_native(crop, black, white)).to(device=device, dtype=torch.float64)
    count = torch.ones_like(total)
    others = [j for j in range(len(frames)) if j != anchor]
    if others:
        local = np.stack([crop_matrix(matrices[j], np.asarray(origin)) for j in others])
        for indices, crops, support in sample_native_crops([frames[j] for j in others], local, size, black, white,
                                                            device, require_overlap=False):
            coefficients = torch.as_tensor(brightness[[others[i] for i in indices]], device=device, dtype=torch.float64)
            corrected = crops[:, 0].double() * coefficients[:, 0, None, None] + coefficients[:, 1, None, None]
            mask = support[:, 0]
            total += torch.where(mask, corrected, 0).sum(dim=0)
            count += mask.sum(dim=0)
    return (total / count).float(), count == len(frames)


def build_fusion_inputs(frames_by_site: dict[int, np.ndarray], specs: Sequence[FusionInputSpec],
                        black: float, white: float, device: str | torch.device
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """``[B, 1, S, S]`` unit means and full-support masks for a batch of specs."""
    means, masks = [], []
    for spec in specs:
        frames = frames_by_site[spec.source_index]
        mean, valid = registered_mean([frames[j] for j in spec.members], spec.anchor, spec.matrices,
                                      spec.brightness, spec.origin, spec.size, black, white, device)
        means.append(mean[None])
        masks.append(valid[None])
    return torch.stack(means), torch.stack(masks)


def registered_mean_frame(frames: Sequence[np.ndarray], anchor: int, matrices: np.ndarray, brightness: np.ndarray,
                          black: float, white: float, device: str | torch.device, *, tile: int = MEAN_TILE
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Full-frame :func:`registered_mean` in bounded square tiles."""
    height, width = np.asarray(frames[anchor]).shape
    size = min(tile, height, width)
    mean = np.empty((height, width), dtype=np.float64)
    valid = np.empty((height, width), dtype=bool)
    rows = sorted({min(y, height - size) for y in range(0, height, size)})
    columns = sorted({min(x, width - size) for x in range(0, width, size)})
    for top in rows:
        for left in columns:
            values, support = registered_mean(frames, anchor, matrices, brightness, (top, left), size,
                                              black, white, device)
            mean[top:top + size, left:left + size] = values.cpu().numpy()
            valid[top:top + size, left:left + size] = support.cpu().numpy()
    return mean, valid


def block_geometry(frames: np.ndarray, black: float = 0.0, white: float = 255.0,
                   min_contrast: float = 0.005) -> tuple[np.ndarray, list[dict], int | None]:
    """Translation-seeded affine ECC over one block, anchored on its first usable frame.

    Returns 2x3 matrices mapping the anchor's coordinates to each frame (the
    anchor and every unmeasured frame keep the identity), per-frame
    diagnostics and the anchor index (``None`` if no frame is usable). This is
    the estimator of the registered raw-average baseline, so a fused block and
    the registered average of the same block share their geometry.
    """
    from sem_noise.pair_matching import GeometryEstimationError, check_geometry_reference, estimate_geometry
    from sem_noise.registration import clip_mask

    matrices = np.repeat(np.eye(2, 3)[None], len(frames), axis=0)
    diagnostics, reference = [], None
    seed = np.eye(2, 3)
    for index, frame in enumerate(frames):
        row = {"frame": index + 1, "status": "registered", "reason": ""}
        diagnostics.append(row)
        contrast = registration_contrast(torch.from_numpy(normalize_native(frame, black, white))[None, None])
        row["contrast"] = contrast
        if contrast < min_contrast:
            row.update(status="skipped_low_contrast", reason="insufficient structured contrast; retained in native coordinates")
            continue
        invalid = clip_mask(frame, (black, white))
        try:
            check_geometry_reference(frame, sigma=1, invalid=invalid)
            if reference is None:
                reference = index
                row["status"] = "reference"
                continue
            translation, _ = estimate_geometry(frames[reference], frame, motion="translation", initial=seed,
                sigma=1, input_invalid=clip_mask(frames[reference], (black, white)), target_invalid=invalid)
            matrix, score = estimate_geometry(frames[reference], frame, motion="affine", initial=translation,
                sigma=1, input_invalid=clip_mask(frames[reference], (black, white)), target_invalid=invalid)
        except GeometryEstimationError as error:
            row.update(status="skipped_failed_registration", reason=str(error))
            continue
        matrices[index], seed = matrix, translation
        row["score"] = score
    for row, matrix in zip(diagnostics, matrices):
        row["matrix"] = matrix.tolist()
    return matrices, diagnostics, reference


def fused_input(frames: np.ndarray, *, black: float, white: float, registration: str, brightness: str,
                device: str | torch.device) -> tuple[np.ndarray, np.ndarray, dict]:
    """The deployed m-frame input: unit mean in the anchor's coordinates, support, record.

    ``registration``/``brightness`` are the checkpoint's training treatment
    (``data.real_matching``). Brightness matching applies to the INPUT
    members only; the denoised output is never corrected.
    """
    frames = np.asarray(frames)
    if frames.ndim != 3 or not len(frames):
        raise ValueError("fused input needs a nonempty [m, H, W] stack of native frames")
    count = len(frames)
    if registration == "affine" and count > 1:
        matrices, diagnostics, reference = block_geometry(frames, black, white)
    elif registration == "none" or count == 1:
        matrices, diagnostics, reference = np.repeat(np.eye(2, 3)[None], count, axis=0), [], None
    else:
        raise ValueError(f"fused inference supports affine or none registration, not {registration!r}")
    anchor = 0 if reference is None else reference
    coefficients = np.tile([1.0, 0.0], (count, 1))
    if brightness == "percentile":
        quantiles = [np.percentile(frame, PERCENTILES) for frame in frames]
        for j in range(count):
            if j != anchor:
                coefficients[j] = brightness_to_anchor(quantiles[anchor], quantiles[j], black, white)
    elif brightness != "none":
        raise ValueError(f"unknown brightness treatment {brightness!r}")
    mean, valid = registered_mean_frame(list(frames), anchor, matrices, coefficients, black, white, device)
    record = {"frames": count, "anchor_frame": anchor + 1, "registration": registration,
              "brightness": brightness, "brightness_to_anchor": coefficients.tolist(),
              "geometry": diagnostics, "common_valid_fraction": float(valid.mean()),
              "outside_support": "mean of the members that support the pixel; excluded from contour measurements"}
    return mean, valid, record


class MatchedRealFusionFactory(MatchedRealPairFactory):
    """m consecutive acquisitions in, one other matched acquisition as target."""

    supports_fusion = True

    def __init__(self, cache: BurstCache, config: Config, *, seed: int,
                 measurements: dict | None = None, device: str | torch.device = "cpu",
                 timings: TrainingTimings | None = None) -> None:
        fusion = config.objective.fusion
        if fusion is None or fusion.align != "matched":
            raise ValueError("real SEM burst fusion requires objective.fusion.align 'matched'")
        if config.objective.lambda_consistency > 0:
            raise ValueError("real SEM burst fusion does not support consistency")
        super().__init__(cache, config, seed=seed, measurements=measurements, device=device, timings=timings)
        self.levels = [int(level) for level in fusion.levels]
        self.condition_on_level = fusion.condition_on_level

    def _anchor(self, index: int, members: tuple[int, ...]) -> int:
        """First member with usable geometry (the inference block's anchor rule)."""
        available = self.geometry_available[index]
        return next((j for j in members if available[j]), members[0])

    def _fusion_sample(self, source: BurstSource, window: tuple[int, int], members: tuple[int, ...],
                       target: int) -> tuple:
        index = source.source_index
        anchor = self._anchor(index, members)
        # The target is exactly the matched real N2N target of the anchor crop.
        _, targets, _, target_valid, _, _, _, info = self._pair(source, window, anchor, target, None)
        matrices = np.stack([np.eye(2, 3) if j == anchor else self._pair_matrix(index, anchor, j) for j in members])
        brightness = np.array([(1.0, 0.0) if j == anchor else self._brightness(index, anchor, j) for j in members])
        spec = FusionInputSpec(index, members, members.index(anchor), tuple(int(v) for v in info.crop_yx),
                               self.image_size, matrices, brightness)
        level = float(len(members)) if self.condition_on_level else 1.0
        return spec, targets, target_valid, level, RealFusionInfo(index, info.crop_yx, members, anchor, target,
                                                                  len(members))

    def _batch(self, samples: list[tuple], *, validation: bool = False):
        specs, targets, valid, levels, infos = zip(*samples)
        with self.timings.measure("fusion_input") if self.timings is not None else nullcontext():
            means, supported = build_fusion_inputs(self.frames_by_site, specs, self.black, self.white, self.device)
        target_valid = torch.from_numpy(np.stack(valid))[:, None].to(supported.device) & supported
        values = dict(inputs=means * 2 - 1, targets=torch.from_numpy(np.stack(targets).astype(np.float32)),
                      second=None, loss_margin=LOSS_MARGIN, target_valid=target_valid,
                      levels=torch.tensor(levels, dtype=torch.float32))
        return (ValPairBatch(**values, clean=None) if validation else PairBatch(**values)), list(infos)

    def sample_batch(self, *, count: int | None = None, return_info: bool = False):
        count = self.batch_size if count is None else count
        if count < 1:
            raise ValueError("count must be positive")
        samples = []
        for _ in range(count):
            source = self.cache.train_sources[int(self._rng.integers(len(self.cache.train_sources)))]
            y0, x0, y1, x1 = self.sites[source.source_index]["bounds"]
            window = (int(self._rng.integers(y0, y1 - self.image_size + 1)),
                      int(self._rng.integers(x0, x1 - self.image_size + 1)))
            level = self.levels[int(self._rng.integers(len(self.levels)))]
            frames = len(source.frames)
            start = int(self._rng.integers(frames - level + 1))
            pick = int(self._rng.integers(frames - level))
            target = pick if pick < start else pick + level
            samples.append(self._fusion_sample(source, window, tuple(range(start, start + level)), target))
        batch, info = self._batch(samples)
        return (batch, info) if return_info else batch

    def val_batch(self, *, count: int, level: int | None = None) -> ValPairBatch:
        """Fixed windows; the first ``level`` acquisitions in, the next one as target."""
        level = max(self.levels) if level is None else int(level)
        if not self.cache.val_sources or count < 1:
            raise ValueError("positive count and validation sites are required")
        samples = []
        for index in range(count):
            source = self.cache.val_sources[index % len(self.cache.val_sources)]
            windows = fixed_windows(self.sites[source.source_index]["bounds"], self.image_size)
            window = windows[(index // len(self.cache.val_sources)) % len(windows)]
            samples.append(self._fusion_sample(source, window, tuple(range(level)), level))
        return self._batch(samples, validation=True)[0]
