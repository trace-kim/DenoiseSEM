"""Bridge into burst_diffusion's repeatability evaluation.

``burst_diffusion.repeatability`` owns the CD/registration metrology harness
(site selection, subpixel threshold crossings, pooled c4-debiased sigmas);
this module supplies edge_denoise checkpoints to it as extra realization
providers, so a single run produces ONE table holding classical baselines,
burst/N2N arms, and edge_denoise arms -- measured on identical sources, seeds,
crops, and CD sites.

The single method name is ``one_shot``: like the burst pipeline's one_shot it
is a single deterministic forward pass on the raw frame, so rows read
``one_shot@<arm>`` across both pipelines and mean the same estimator shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from burst_diffusion.provenance import file_sha256
from burst_diffusion.repeatability import RealizationProvider

from .infer import Denoiser
from .train import load_checkpoint

METHOD_NAME = "one_shot"


def checkpoint_record(path: str | Path) -> dict:
    """What exactly was evaluated: path, content hash, training step, kind.

    The harness records provider arms by *name* only; this record, stored next
    to it in ``repeatability.json``, binds each edge arm to one checkpoint so
    a table can be audited (and a re-run compared) without trusting file
    names or directory contents that a later run may have overwritten.
    """
    resolved = Path(path)
    payload = load_checkpoint(resolved, map_location="cpu")
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "step": int(payload.get("step", -1)),
        "bytes": resolved.stat().st_size,
        "kind": str(payload.get("kind", "")),
    }


def checkpoint_records(checkpoints: dict[str, str | Path]) -> dict[str, dict]:
    return {arm: checkpoint_record(path) for arm, path in checkpoints.items()}


def realization_provider(denoiser: Denoiser, *, max_batch: int = 10) -> RealizationProvider:
    """Wrap a :class:`Denoiser` as a repeatability realization provider."""

    def generate(seeds01: list[np.ndarray]) -> dict[str, list[np.ndarray]]:
        return {METHOD_NAME: denoiser.denoise01(seeds01, max_batch=max_batch)}

    return RealizationProvider(method_names=(METHOD_NAME,), generate=generate)


def callable_provider(
    denoise01: "Callable[[list[np.ndarray]], list[np.ndarray]]", *, max_batch: int = 10
) -> RealizationProvider:
    """Wrap any ``denoise01(list of [H, W, C] in [0, 1]) -> same`` callable
    (e.g. a posterior sampler) as a ``one_shot`` provider."""

    def generate(seeds01: list[np.ndarray]) -> dict[str, list[np.ndarray]]:
        return {METHOD_NAME: denoise01(seeds01, max_batch=max_batch)}

    return RealizationProvider(method_names=(METHOD_NAME,), generate=generate)


def providers_from_checkpoints(
    checkpoints: dict[str, str | Path], *, device: str = "auto", max_batch: int = 10
) -> dict[str, RealizationProvider]:
    """Load each checkpoint arm and wrap it as a provider."""
    return {
        arm: realization_provider(
            Denoiser.from_checkpoint(path, device=device), max_batch=max_batch
        )
        for arm, path in checkpoints.items()
    }
