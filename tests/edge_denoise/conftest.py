from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from edge_denoise.config import Config


def write_burst(
    root: Path,
    *,
    num_sources: int = 4,
    replicas: int = 4,
    height: int = 20,
    width: int = 24,
    bar: bool = False,
) -> Path:
    """Tiny burst dataset. ``bar=True`` plants a measurable vertical bar whose
    width differs per source (distinct content, so the group split keeps every
    source its own group)."""
    burst = root / "burst"
    (burst / "clean").mkdir(parents=True)
    (burst / "noisy").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for source_index in range(num_sources):
        if bar:
            cols = np.arange(width, dtype=np.float64)
            left, right = 9.5, 14.5 + source_index / 4.0
            coverage = np.clip(np.minimum(cols + 1.0, right) - np.maximum(cols, left), 0.0, 1.0)
            profile = 0.2 + 0.6 * coverage
            clean = np.rint(np.tile(profile, (height, 1)) * 255.0).astype(np.uint8)
        else:
            base = np.arange(height)[:, None] * 7 + np.arange(width)[None, :] * 3
            clean = np.rint((base + source_index * 11) % 200 + 28).astype(np.uint8)
        Image.fromarray(clean).save(burst / "clean" / f"{source_index:05d}.png")
        for replica_index in range(replicas):
            noisy = np.clip(
                clean.astype(np.int16) + rng.integers(-30, 31, clean.shape), 0, 255
            ).astype(np.uint8)
            name = f"noisy/{source_index:05d}_{replica_index:05d}.png"
            Image.fromarray(noisy).save(burst / name)
            rows.append(
                json.dumps(
                    {
                        "source_index": source_index,
                        "replica_index": replica_index,
                        "clean_path": f"clean/{source_index:05d}.png",
                        "noisy_path": name,
                    }
                )
            )
    (burst / "manifest.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


def make_config(
    dataset_dir: Path,
    run_dir: Path,
    *,
    representation: str = "hybrid",
    target: str = "noisy",
    lambda_image: float = 1.0,
    lambda_gradient: float = 4.0,
    lambda_consistency: float = 0.0,
    max_steps: int = 2,
) -> Config:
    if representation == "gradient":
        lambda_image = 0.0
        lambda_gradient = max(lambda_gradient, 1.0)
    return Config.model_validate(
        {
            "data": {
                "dataset_dir": str(dataset_dir),
                "image_size": 16,
                "channels": 1,
                "val_fraction": 0.34,
                "split_seed": 7,
            },
            "objective": {
                "representation": representation,
                "target": target,
                "lambda_image": lambda_image,
                "lambda_gradient": lambda_gradient,
                "lambda_consistency": lambda_consistency,
            },
            "model": {"ch": 8, "ch_mult": [1, 2], "num_res_blocks": 1, "attn_resolutions": []},
            "training": {
                "run_dir": str(run_dir),
                "batch_size": 2,
                "max_steps": max_steps,
                "log_every": 1,
                "val_every": max(1, max_steps),
                "val_images": 1,
                "checkpoint_every": max(1, max_steps),
                "device": "cpu",
                "seed": 3,
            },
        }
    )


@pytest.fixture()
def burst_dataset(tmp_path: Path) -> Path:
    return write_burst(tmp_path / "data")


@pytest.fixture()
def bar_dataset(tmp_path: Path) -> Path:
    return write_burst(tmp_path / "data", bar=True)
