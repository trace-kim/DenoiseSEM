from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import make_config, write_burst

from burst_diffusion.repeatability import repeatability

from edge_denoise.infer import Denoiser
from edge_denoise.provider import METHOD_NAME, providers_from_checkpoints
from edge_denoise.train import LATEST_CHECKPOINT_NAME, Trainer


def _train_checkpoint(tmp_path: Path, representation: str = "hybrid") -> Path:
    dataset = write_burst(tmp_path / "data", bar=True)
    config = make_config(dataset, tmp_path / "run", representation=representation)
    Trainer(config).run()
    return tmp_path / "run" / LATEST_CHECKPOINT_NAME


@pytest.mark.parametrize("representation", ["image", "gradient", "hybrid"])
def test_denoiser_round_trips_every_representation(
    tmp_path: Path, representation: str
) -> None:
    checkpoint = _train_checkpoint(tmp_path, representation)
    denoiser = Denoiser.from_checkpoint(checkpoint, device="cpu")
    frames = torch.rand(3, 1, 16, 16) * 2.0 - 1.0
    denoised = denoiser.denoise(frames)
    assert denoised.shape == (3, 1, 16, 16)
    assert denoised.min().item() >= -1.0 and denoised.max().item() <= 1.0


def test_denoiser_is_deterministic(tmp_path: Path) -> None:
    checkpoint = _train_checkpoint(tmp_path)
    denoiser = Denoiser.from_checkpoint(checkpoint, device="cpu")
    frames = torch.rand(2, 1, 16, 16) * 2.0 - 1.0
    assert torch.equal(denoiser.denoise(frames), denoiser.denoise(frames))


def test_denoise01_preserves_format_and_order(tmp_path: Path) -> None:
    checkpoint = _train_checkpoint(tmp_path)
    denoiser = Denoiser.from_checkpoint(checkpoint, device="cpu")
    frames = [np.random.default_rng(i).random((16, 16, 1)) for i in range(5)]
    outputs = denoiser.denoise01(frames, max_batch=2)
    assert len(outputs) == 5
    assert all(o.shape == (16, 16, 1) and o.dtype == np.float64 for o in outputs)
    # Batching must not change results.
    assert np.allclose(outputs[0], denoiser.denoise01(frames[:1])[0])


def test_provider_joins_the_repeatability_table(tmp_path: Path) -> None:
    """End-to-end: an edge_denoise arm runs through burst_diffusion's harness
    alongside the classical rows, sharing sources, seeds, and CD sites."""
    from burst_diffusion.config import Config as BurstConfig

    checkpoint = _train_checkpoint(tmp_path)
    providers = providers_from_checkpoints({"edge": checkpoint}, device="cpu")
    bridge = BurstConfig.model_validate(
        {
            "data": {
                "dataset_dir": str(tmp_path / "data"),
                "image_size": 16,
                "channels": 1,
                "val_fraction": 0.34,
                "split_seed": 7,
            },
            "schedule": {"num_steps": 1},
            "model": {"ch": 8, "ch_mult": [1], "num_res_blocks": 1, "attn_resolutions": []},
            "training": {"run_dir": str(tmp_path / "rep"), "device": "cpu"},
            "sampling": {},
        }
    )
    results = repeatability(
        bridge,
        {},
        out_dir=tmp_path / "rep",
        num_seeds=2,
        avg_counts=(2,),
        edge_tolerance=3.0,
        extra_providers=providers,
    )
    row = f"{METHOD_NAME}@edge"
    assert row in results["methods"]
    assert results["provider_arms"] == ["edge"]
    method = results["methods"][row]
    assert method["pixel_repeatability"] is not None
    assert np.isfinite(method["accuracy"]["psnr_mean"])
    assert (tmp_path / "rep" / "summary.md").read_text(encoding="utf-8").count(row) >= 1
