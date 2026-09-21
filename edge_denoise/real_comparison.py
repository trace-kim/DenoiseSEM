"""Fixed real-SEM training panels, independent of the pair-sampling RNG."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from .real_data import fixed_windows
from .uint8_output import RANGE_WARNING, average_uint8, prediction_uint8

if TYPE_CHECKING:
    from burst_diffusion.data import BurstCache
    from torch.utils.tensorboard import SummaryWriter
    from .model import EdgeDenoiser


def prepare_examples(cache: BurstCache, image_size: int) -> list[dict]:
    """Precompute raw average crops from train/val only; never inspect test pixels."""
    if cache.real_metadata is None:
        raise ValueError("training.real_comparison_images requires prepared real SEM data")
    sites = {site["source_index"]: site for site in cache.real_metadata["sites"]}
    examples = []
    for split in ("train", "val"):
        sources = sorted(getattr(cache, f"{split}_sources"), key=lambda s: s.source_index)
        candidates = []
        for source in sources:
            if source.frames.dtype != np.uint8 or len(source.frames) < 8:
                raise ValueError("real comparison panels require uint8 sites with at least eight frames")
            windows = fixed_windows(sites[source.source_index]["bounds"], image_size)
            candidates.append((source, windows))
        count = 0
        for window_index in range(5):
            for source, windows in candidates:
                if count >= 4 or window_index >= len(windows):
                    continue
                y, x = windows[window_index]
                crops = source.frames[:, y:y + image_size, x:x + image_size]
                examples.append({"split": split, "source_index": source.source_index,
                                 "origin_yx": (y, x), "input": crops[0].copy(),
                                 "average8": average_uint8(crops[:8]), "full_average": average_uint8(crops)})
                count += 1
    return examples


def log_examples(model: EdgeDenoiser, examples: list[dict], writer: SummaryWriter, step: int, *,
                 device: str | torch.device, black: float, white: float) -> None:
    """Audit before display clipping; preserve model mode and all torch RNG state."""
    was_training = model.training
    resolved = torch.device(device)
    devices = [resolved.index if resolved.index is not None else torch.cuda.current_device()] if resolved.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            for split in ("train", "val"):
                panels = []
                for index, example in enumerate(e for e in examples if e["split"] == split):
                    normalized = np.clip((example["input"].astype(np.float64) - black) / (white - black), 0, 1)
                    tensor = torch.as_tensor(normalized * 2 - 1, dtype=torch.float32, device=device)[None, None]
                    prediction = (model.predict_image(tensor).cpu().numpy()[0, 0].astype(np.float64) + 1) / 2
                    prefix = f"real_comparison/{split}/{index}"
                    try:
                        image, stats = prediction_uint8(prediction, black, white)
                    except ValueError as error:
                        writer.add_text(f"{prefix}/prediction_failed", str(error), step)
                        writer.add_scalar(f"{prefix}/prediction_failed", 1, step)
                        continue  # Explicit failure, never fabricate a display image.
                    writer.add_scalar(f"{prefix}/prediction_failed", 0, step)
                    for key in ("minimum_dn", "maximum_dn", "below_zero", "above_255", "out_of_range_fraction"):
                        writer.add_scalar(f"{prefix}/{key}", stats[key], step)
                    if stats["clipped"]:
                        writer.add_text(f"{prefix}/range_warning", RANGE_WARNING + f"\n\n{stats}", step)
                    panels.append(np.concatenate([example["input"], image, example["average8"], example["full_average"]], axis=1))
                    writer.add_text(f"{prefix}/example", f"source={example['source_index']}; origin_yx={example['origin_yx']}; "
                                    "input acquisition=1; average acquisitions=1–8; raw native crops", step)
                if panels:
                    writer.add_image(f"real_comparison/{split}/input_prediction_average8_full", np.concatenate(panels, axis=0)[None], step)
                    writer.add_text(f"real_comparison/{split}/columns", "Input | uint8 prediction | raw eight-frame average | "
                                    "full average (visual reference only). Fixed crops; no test sites.", step)
    finally:
        model.train(was_training)
