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


def test_denoise_full_preserves_frame_size(tmp_path: Path) -> None:
    checkpoint = _train_checkpoint(tmp_path)
    denoiser = Denoiser.from_checkpoint(checkpoint, device="cpu")
    frame01 = np.random.default_rng(0).random((40, 33))
    out = denoiser.denoise_full(frame01, stride=8)
    assert out.shape == (40, 33)
    assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0


def test_denoise_full_at_training_resolution_matches_the_direct_pass(tmp_path: Path) -> None:
    """A frame exactly at the training resolution is one tile, so the blended
    path must reproduce the plain forward pass (window weights cancel)."""
    checkpoint = _train_checkpoint(tmp_path)
    denoiser = Denoiser.from_checkpoint(checkpoint, device="cpu")
    frame01 = np.random.default_rng(1).random((16, 16))
    direct = denoiser.denoise(
        torch.from_numpy((frame01 * 2.0 - 1.0)[None, None]).to(torch.float32)
    )
    direct01 = (direct.numpy()[0, 0] + 1.0) / 2.0
    assert np.allclose(denoiser.denoise_full(frame01), direct01, atol=1e-6)


def test_load_measurement01_reads_8_and_16_bit_whole_frames(tmp_path: Path) -> None:
    from PIL import Image

    from edge_denoise.infer import load_measurement01

    eight = tmp_path / "m8.png"
    Image.fromarray(np.array([[0, 128], [255, 64]], dtype=np.uint8)).save(eight)
    arr8 = load_measurement01(eight)
    assert arr8.shape == (2, 2)  # whole frame, no crop
    assert arr8[1, 0] == 1.0 and abs(arr8[0, 1] - 128.0 / 255.0) < 1e-12

    sixteen = tmp_path / "m16.png"
    Image.fromarray(np.array([[0, 32768], [65535, 257]], dtype=np.uint16)).save(sixteen)
    arr16 = load_measurement01(sixteen)
    assert arr16[1, 0] == 1.0 and abs(arr16[0, 1] - 32768.0 / 65535.0) < 1e-12


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
