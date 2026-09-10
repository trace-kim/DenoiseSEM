"""Accuracy evaluation: model vs the classical baselines, per held-out source.

Three methods, PSNR + SSIM against clean on deterministic center crops:

- ``single_frame``: burst frame 0 as-is -- the measurement everything starts from.
- ``avg_of_n``: plain average of ALL burst frames (reference baseline).
- ``one_shot``: the model's single deterministic forward pass on frame 0.

Both the mean and the MEDIAN are reported -- the burst report's audit found
PSNR means skewed by a few near-flat crops.  Precision (the axis this package
exists for) is measured by the ``repeatability`` command instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from burst_diffusion.data import BurstCache
from burst_diffusion.metrics import psnr, ssim

from .config import Config
from .infer import Denoiser

METHOD_NAMES = ("single_frame", "avg_of_n", "one_shot")


def _to_hwc01(array: np.ndarray) -> np.ndarray:
    image = array.astype(np.float64) / 255.0
    if image.ndim == 2:
        image = image[:, :, None]
    return image


def evaluate(
    config: Config,
    checkpoint: str | Path,
    *,
    split: str = "val",
    limit: int | None = None,
    out_dir: str | Path,
    device: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Run the evaluation; writes results.json and returns the results dict."""
    if split not in ("val", "train", "test"):
        raise ValueError(f"split must be 'val', 'train', or 'test', got {split!r}")
    denoiser = Denoiser.from_checkpoint(
        checkpoint, device=device if device is not None else config.training.device
    )
    if denoiser.image_size != config.data.image_size:
        raise ValueError(
            f"checkpoint was trained at image_size={denoiser.image_size} but the "
            f"config says {config.data.image_size}"
        )
    cache = BurstCache(
        config.data.dataset_dir,
        channels=config.data.channels,
        min_replicas=config.min_replicas,
        min_size=config.data.image_size,
        val_fraction=config.data.val_fraction,
        test_fraction=config.data.test_fraction,
        split_seed=config.data.split_seed,
    )
    sources = cache.sources_for_split(split)
    if cache.real_metadata is not None:
        raise ValueError("Real SEM has no clean ground truth; use edge_denoise evaluate-real")
    if not sources:
        raise ValueError(f"no sources in the {split!r} split")
    if limit is not None:
        sources = sources[:limit]

    size = config.data.image_size
    per_image: dict[str, dict[str, list[float]]] = {
        name: {"psnr": [], "ssim": []} for name in METHOD_NAMES
    }
    for index, source in enumerate(sources):
        height, width = source.clean.shape[:2]
        top = (height - size) // 2
        left = (width - size) // 2
        window = np.s_[top : top + size, left : left + size]
        clean01 = _to_hwc01(source.clean[window])
        frames01 = [_to_hwc01(frame[window]) for frame in source.frames]
        outputs = {
            "single_frame": frames01[0],
            "avg_of_n": np.mean(frames01, axis=0),
            "one_shot": denoiser.denoise01([frames01[0]])[0],
        }
        for name in METHOD_NAMES:
            per_image[name]["psnr"].append(psnr(clean01, outputs[name]))
            per_image[name]["ssim"].append(ssim(clean01, outputs[name]))
        if progress_callback is not None:
            progress_callback(index + 1, len(sources))

    results = {
        "checkpoint": str(checkpoint),
        "dataset_dir": str(config.data.dataset_dir),
        "split": split,
        "count": len(sources),
        "source_indices": [source.source_index for source in sources],
        "representation": config.objective.representation,
        "methods": {
            name: {
                "psnr_mean": float(np.mean(values["psnr"])),
                "psnr_median": float(np.median(values["psnr"])),
                "ssim_mean": float(np.mean(values["ssim"])),
                "psnr_per_image": values["psnr"],
                "ssim_per_image": values["ssim"],
            }
            for name, values in per_image.items()
        },
    }
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return results
