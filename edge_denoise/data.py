"""Paired-frame batch assembly on top of burst_diffusion's BurstCache.

The cache -- and with it the content-group split machinery that the leakage
audit mandated -- is imported from ``burst_diffusion.data`` rather than
reduplicated: with identical (dataset_dir, channels, fractions, split_seed)
the two packages see byte-identical train/val/test splits, so every
edge_denoise arm is measured on the same held-out sources as the burst and
N2N arms it is compared against.

No DataLoader, for the same documented reasons as burst_diffusion: the
worker-restart trap on small datasets, and the need for random access to all
frames of a source.  :class:`PairFactory` assembles batches from one seeded
RNG whose state rides in the checkpoint, so runs resume exactly.

A training sample is (input frame, target, optional second frame, optional
gradient target), all crops of the SAME window of the same source:

- ``input``:  one noisy frame, drawn uniformly.
- ``target``: the clean image (``target: clean``), or a FRESH noisy frame from
  a different replica (``target: noisy`` -- Noise2Noise).  Sharing the input
  replica would make the MSE-optimal network the identity.
- ``second``: an independent third frame for the consistency penalty,
  distinct from both input and target so the penalty correlates with neither
  the input noise nor the target noise.
- ``gradient_targets``: present only when ``objective.gradient_target``
  overrides the gradient term's reference (the target-ladder experiment):
  the clean crop (``clean``), the leave-one-out mean of every replica EXCEPT
  the input (``noisy_mean`` -- excluding the input keeps the target
  independent of the input noise, the load-bearing Noise2Noise condition),
  or a precomputed per-source image loaded from disk (``file`` -- the
  distillation arm's frozen "answer sheet").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from burst_diffusion.data import BurstCache, BurstSource

GRADIENT_TARGET_MODES = ("target", "clean", "noisy_mean", "file")


@dataclass(frozen=True)
class PairInfo:
    """Provenance of one training sample, exposed for tests and debugging."""

    source_index: int
    crop_yx: tuple[int, int]
    input_replica: int
    target_replica: int | None  # None when the target is the clean image
    second_replica: int | None


@dataclass
class PairBatch:
    inputs: torch.Tensor  # [B, 1, S, S] float32 in [-1, 1]
    targets: torch.Tensor  # [B, 1, S, S] float32 in [-1, 1]
    second: torch.Tensor | None  # [B, 1, S, S] float32 in [-1, 1]
    gradient_targets: torch.Tensor | None = None  # [B, 1, S, S]; None for mode "target"


@dataclass
class ValPairBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    second: torch.Tensor
    clean: torch.Tensor
    gradient_targets: torch.Tensor | None = None


def _to_model_chw(crop: np.ndarray) -> np.ndarray:
    scaled = crop.astype(np.float32) / 255.0 * 2.0 - 1.0
    if scaled.ndim == 2:
        return scaled[None, :, :]
    return np.ascontiguousarray(scaled.transpose(2, 0, 1))


class PairFactory:
    """Seeded assembler of paired training/validation batches."""

    def __init__(
        self,
        cache: BurstCache,
        *,
        image_size: int,
        batch_size: int,
        target: str = "noisy",
        need_second: bool = False,
        gradient_target: str = "target",
        gradient_target_dir: str | Path | None = None,
        seed: int = 0,
    ):
        if target not in ("clean", "noisy"):
            raise ValueError(f"target must be 'clean' or 'noisy', got {target!r}")
        if gradient_target not in GRADIENT_TARGET_MODES:
            raise ValueError(
                f"gradient_target must be one of {GRADIENT_TARGET_MODES}, got {gradient_target!r}"
            )
        if gradient_target == "file" and gradient_target_dir is None:
            raise ValueError("gradient_target 'file' requires gradient_target_dir")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not cache.train_sources:
            raise ValueError("cache has no training sources")
        required = 1 + int(target == "noisy") + int(need_second)
        if gradient_target == "noisy_mean":
            required = max(required, 2)
        for source in cache.all_sources:
            if len(source.frames) < required:
                raise ValueError(
                    f"source {source.source_index} has {len(source.frames)} frames but "
                    f"target={target!r}"
                    + (" with a consistency pair" if need_second else "")
                    + f" needs at least {required} distinct replicas"
                )
            if min(source.clean.shape[0], source.clean.shape[1]) < image_size:
                raise ValueError(
                    f"source {source.source_index} is {source.clean.shape[:2]} but crops "
                    f"need at least {image_size}x{image_size}"
                )
        self.cache = cache
        self.image_size = image_size
        self.batch_size = batch_size
        self.target = target
        self.need_second = need_second
        self.gradient_target = gradient_target
        self._rng = np.random.default_rng(seed)

        # Per-source auxiliaries for the gradient-target modes, covering every
        # source training or validation can touch (never the test split).
        active_sources = list(cache.train_sources) + list(cache.val_sources)
        self._frame_sums: dict[int, np.ndarray] = {}
        self._file_targets: dict[int, np.ndarray] = {}
        if gradient_target == "noisy_mean":
            for source in active_sources:
                if len(source.frames) > 257:
                    raise ValueError(
                        f"source {source.source_index} has {len(source.frames)} replicas; "
                        "the uint16 frame-sum accumulator supports at most 257"
                    )
                total = np.zeros(source.frames[0].shape, dtype=np.uint16)
                for frame in source.frames:
                    total += frame
                self._frame_sums[source.source_index] = total
        elif gradient_target == "file":
            directory = Path(gradient_target_dir)  # type: ignore[arg-type]
            for source in active_sources:
                path = directory / f"{source.source_index:05d}.npy"
                if not path.is_file():
                    raise ValueError(
                        f"gradient_target 'file' needs {path} for source "
                        f"{source.source_index}; generate targets with "
                        "`python -m edge_denoise distill-targets`"
                    )
                array = np.load(path)
                if array.ndim != 2 or array.shape != source.clean.shape[:2]:
                    raise ValueError(
                        f"{path} has shape {array.shape} but source "
                        f"{source.source_index} is {source.clean.shape[:2]}"
                    )
                if not np.isfinite(array).all():
                    raise ValueError(f"{path} contains non-finite values")
                self._file_targets[source.source_index] = array.astype(np.float32)

    def _gradient_target_crop(
        self, source: BurstSource, window: tuple[slice, slice], input_replica: int
    ) -> np.ndarray:
        """The gradient term's reference crop in model range, [1, S, S]."""
        if self.gradient_target == "clean":
            return _to_model_chw(source.clean[window])
        if self.gradient_target == "noisy_mean":
            total = self._frame_sums[source.source_index][window].astype(np.float32)
            others = total - source.frames[input_replica][window].astype(np.float32)
            mean01 = others / (255.0 * (len(source.frames) - 1))
            return (mean01 * 2.0 - 1.0)[None, :, :]
        # "file": stored as float32 in [0, 1]
        return (self._file_targets[source.source_index][window] * 2.0 - 1.0)[None, :, :]

    def sample_batch(
        self, *, count: int | None = None, return_info: bool = False
    ) -> PairBatch | tuple[PairBatch, list[PairInfo]]:
        count = self.batch_size if count is None else count
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        size = self.image_size
        input_list: list[np.ndarray] = []
        target_list: list[np.ndarray] = []
        second_list: list[np.ndarray] = []
        gradient_list: list[np.ndarray] = []
        info: list[PairInfo] = []
        for _ in range(count):
            source = self.cache.train_sources[
                int(self._rng.integers(len(self.cache.train_sources)))
            ]
            height, width = source.clean.shape[:2]
            top = int(self._rng.integers(0, height - size + 1))
            left = int(self._rng.integers(0, width - size + 1))
            window = np.s_[top : top + size, left : left + size]

            permutation = self._rng.permutation(len(source.frames))
            input_replica = int(permutation[0])
            cursor = 1
            input_list.append(_to_model_chw(source.frames[input_replica][window]))

            target_replica: int | None = None
            if self.target == "noisy":
                target_replica = int(permutation[cursor])
                cursor += 1
                target_list.append(_to_model_chw(source.frames[target_replica][window]))
            else:
                target_list.append(_to_model_chw(source.clean[window]))

            second_replica: int | None = None
            if self.need_second:
                second_replica = int(permutation[cursor])
                second_list.append(_to_model_chw(source.frames[second_replica][window]))

            if self.gradient_target != "target":
                gradient_list.append(
                    self._gradient_target_crop(source, window, input_replica)
                )

            if return_info:
                info.append(
                    PairInfo(
                        source_index=source.source_index,
                        crop_yx=(top, left),
                        input_replica=input_replica,
                        target_replica=target_replica,
                        second_replica=second_replica,
                    )
                )
        batch = PairBatch(
            inputs=torch.from_numpy(np.stack(input_list)),
            targets=torch.from_numpy(np.stack(target_list)),
            second=torch.from_numpy(np.stack(second_list)) if self.need_second else None,
            gradient_targets=(
                torch.from_numpy(np.stack(gradient_list))
                if self.gradient_target != "target"
                else None
            ),
        )
        if return_info:
            return batch, info
        return batch

    def val_batch(self, *, count: int) -> ValPairBatch:
        """Deterministic validation batch: center crops, fixed replicas.

        Input is replica 0 and ``second`` replica 1 (their disagreement after
        denoising is the live repeatability readout); the loss target uses the
        highest fixed replica available so it is distinct from both whenever
        the dataset allows.
        """
        if not self.cache.val_sources:
            raise ValueError("cache has no validation sources")
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        size = self.image_size
        input_list: list[np.ndarray] = []
        target_list: list[np.ndarray] = []
        second_list: list[np.ndarray] = []
        clean_list: list[np.ndarray] = []
        gradient_list: list[np.ndarray] = []
        for index in range(count):
            source = self.cache.val_sources[index % len(self.cache.val_sources)]
            height, width = source.clean.shape[:2]
            top = (height - size) // 2
            left = (width - size) // 2
            window = np.s_[top : top + size, left : left + size]
            input_list.append(_to_model_chw(source.frames[0][window]))
            second_list.append(
                _to_model_chw(source.frames[min(1, len(source.frames) - 1)][window])
            )
            if self.target == "noisy":
                target_replica = min(2, len(source.frames) - 1)
                target_list.append(_to_model_chw(source.frames[target_replica][window]))
            else:
                target_list.append(_to_model_chw(source.clean[window]))
            clean_list.append(_to_model_chw(source.clean[window]))
            if self.gradient_target != "target":
                gradient_list.append(self._gradient_target_crop(source, window, 0))
        return ValPairBatch(
            inputs=torch.from_numpy(np.stack(input_list)),
            targets=torch.from_numpy(np.stack(target_list)),
            second=torch.from_numpy(np.stack(second_list)),
            clean=torch.from_numpy(np.stack(clean_list)),
            gradient_targets=(
                torch.from_numpy(np.stack(gradient_list))
                if self.gradient_target != "target"
                else None
            ),
        )

    def state_dict(self) -> dict:
        return {"rng_state": self._rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self._rng.bit_generator.state = state["rng_state"]
