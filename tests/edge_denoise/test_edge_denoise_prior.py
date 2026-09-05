from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import write_burst

from edge_denoise.prior import (
    CHECKPOINT_KIND,
    LATEST_CHECKPOINT_NAME,
    DDIMSampler,
    PosteriorSampler,
    PriorConfig,
    PriorTrainer,
    Schedule,
    clipped_poisson_nll,
    counts_from_frames,
    ddim_timesteps,
    load_prior_checkpoint,
    parse_posterior_spec,
)


def _prior_config(dataset: Path, run_dir: Path, **training) -> PriorConfig:
    payload = {
        "data": {"dataset_dir": str(dataset), "image_size": 16, "val_fraction": 0.34, "split_seed": 7},
        "diffusion": {"num_timesteps": 50},
        "model": {"ch": 8, "ch_mult": [1, 2], "num_res_blocks": 1, "attn_resolutions": []},
        "training": {
            "run_dir": str(run_dir),
            "batch_size": 2,
            "max_steps": 2,
            "log_every": 1,
            "val_every": 2,
            "checkpoint_every": 2,
            "device": "cpu",
            "seed": 3,
            **training,
        },
    }
    return PriorConfig.model_validate(payload)


def test_schedule_and_timesteps() -> None:
    schedule = Schedule.from_config(PriorConfig.model_validate(
        {"data": {"dataset_dir": "x", "image_size": 16}, "model": {"ch": 8, "ch_mult": [1, 2], "num_res_blocks": 1, "attn_resolutions": []}, "training": {"run_dir": "r"}}
    ).diffusion)
    assert schedule.num_timesteps == 1000
    assert schedule.alpha_bar(-1) == 1.0
    assert 0.0 < schedule.alpha_bar(999) < 1e-3
    # The level matching variance 0.16 is early in the chain, and monotone in the variance.
    t_low = schedule.level_for_variance(0.16)
    t_high = schedule.level_for_variance(1.0)
    assert 50 < t_low < 200 < t_high
    ratio = (1 - schedule.alpha_bar(t_low)) / schedule.alpha_bar(t_low)
    assert ratio <= 0.16 < (1 - schedule.alpha_bar(t_low + 1)) / schedule.alpha_bar(t_low + 1)
    steps = ddim_timesteps(1000, 10)
    assert steps[0] == 999 and steps[-1] == 0 and steps == sorted(steps, reverse=True)
    assert ddim_timesteps(1000, 5, start=100)[0] == 100
    assert ddim_timesteps(1000, 500, start=3) == [3, 2, 1, 0]


def test_clipped_poisson_nll_matches_the_explicit_formula() -> None:
    peak = 10.0
    x01 = torch.tensor([[[[0.3, 0.85]]]], dtype=torch.float32)
    counts = torch.tensor([[[[2.0, 10.0]]]], dtype=torch.float32)
    nll = clipped_poisson_nll(counts, x01, peak)
    lam0, lam1 = 3.0, 8.5
    ordinary = lam0 - 2.0 * math.log(lam0)
    cdf9 = sum(math.exp(-lam1) * lam1**j / math.factorial(j) for j in range(10))
    clipped = -math.log(1.0 - cdf9)
    assert abs(float(nll[0]) - (ordinary + clipped)) < 1e-5
    # Counts recover the stored 8-bit frames exactly (k/peak quantized to uint8).
    stored = torch.tensor([np.rint(k * 255.0 / peak) / 255.0 for k in range(11)], dtype=torch.float32)
    assert torch.equal(counts_from_frames(stored, peak), torch.arange(0, 11, dtype=torch.float32))
    # The NLL is minimized near the observed rate for an unclipped count.
    grid = torch.linspace(0.05, 0.95, 91).view(-1, 1, 1, 1)
    values = clipped_poisson_nll(torch.full_like(grid, 4.0), grid, peak)
    assert abs(float(grid[values.argmin()]) - 0.4) < 0.02


def test_prior_trains_checkpoints_and_samples(tmp_path: Path) -> None:
    dataset = write_burst(tmp_path / "data")
    config = _prior_config(dataset, tmp_path / "run")
    trainer = PriorTrainer(config)
    checkpoint = trainer.run()
    assert trainer.step == 2
    assert checkpoint == tmp_path / "run" / LATEST_CHECKPOINT_NAME
    payload = load_prior_checkpoint(checkpoint)
    assert payload["kind"] == CHECKPOINT_KIND and payload["step"] == 2
    resumed = PriorTrainer(_prior_config(dataset, tmp_path / "run", max_steps=3), resume_from=checkpoint)
    assert resumed.step == 2
    resumed.run()
    assert resumed.step == 3

    sampler = PosteriorSampler.from_checkpoint(checkpoint, device="cpu", peak=10.0, mode="sdedit", num_steps=4)
    frames = torch.rand(2, 1, 16, 16) * 2.0 - 1.0
    out = sampler.denoise(frames)
    assert out.shape == frames.shape and out.abs().max() <= 1.0
    for mode, samples in (("dps", 1), ("dps_mean", 2)):
        sampler = PosteriorSampler.from_checkpoint(
            checkpoint, device="cpu", peak=10.0, mode=mode, num_steps=3, num_samples=samples, seed=5
        )
        first = sampler.denoise(frames)
        second = sampler.denoise(frames)
        assert first.shape == frames.shape
        assert torch.equal(first, second)  # deterministic given (frame, seed)
    other = PosteriorSampler.from_checkpoint(
        checkpoint, device="cpu", peak=10.0, mode="dps", num_steps=3, seed=6
    ).denoise(frames)
    assert not torch.equal(other, first)
    # Stochastic chains (eta > 0) stay deterministic given the seed and differ across seeds.
    stochastic = PosteriorSampler.from_checkpoint(
        checkpoint, device="cpu", peak=10.0, mode="sdedit", num_steps=4, num_samples=2, eta=1.0, seed=1
    )
    assert torch.equal(stochastic.denoise(frames), stochastic.denoise(frames))
    assert not torch.equal(
        stochastic.denoise(frames),
        PosteriorSampler.from_checkpoint(
            checkpoint, device="cpu", peak=10.0, mode="sdedit", num_steps=4, num_samples=2, eta=1.0, seed=2
        ).denoise(frames),
    )
    listed = sampler.denoise01([np.random.default_rng(0).random((16, 16, 1)) for _ in range(3)], max_batch=2)
    assert len(listed) == 3 and listed[0].shape == (16, 16, 1)


def test_proximal_step_moves_the_prior_estimate_toward_the_counts() -> None:
    """With a wide prior variance the proximal solution is the maximum-
    likelihood intensity (k / peak); with a tiny one it stays at the prior."""
    from edge_denoise.prior import PosteriorSampler, Schedule

    class Zero(torch.nn.Module):
        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

    schedule = Schedule.from_config(_prior_config(Path("x"), Path("r")).diffusion)
    sampler = PosteriorSampler(Zero(), schedule, device=torch.device("cpu"), peak=10.0, newton_steps=8)
    counts = torch.tensor([[[[2.0, 6.0, 10.0]]]])
    prior = torch.zeros(1, 1, 1, 3)  # model 0 = intensity 0.5
    loose = (sampler._proximal(counts, prior, r2=100.0) + 1.0) / 2.0
    assert abs(float(loose[0, 0, 0, 0]) - 0.2) < 0.01 and abs(float(loose[0, 0, 0, 1]) - 0.6) < 0.01
    assert float(loose[0, 0, 0, 2]) > 0.9  # clipped bin: as bright as allowed
    tight = (sampler._proximal(counts, prior, r2=1e-6) + 1.0) / 2.0
    assert torch.allclose(tight, torch.full_like(tight, 0.5), atol=1e-3)


def test_ddim_sampler_is_the_identity_when_the_model_predicts_zero_noise() -> None:
    class Zero(torch.nn.Module):
        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

    schedule = Schedule.from_config(_prior_config(Path("x"), Path("r")).diffusion)
    sampler = DDIMSampler(Zero(), schedule, device=torch.device("cpu"))
    x = torch.rand(1, 1, 8, 8) * 0.5
    start = 20
    # With eps = 0 the x0 estimate is x / sqrt(alpha_bar) and the chain converges to it.
    out = sampler.sample(x, ddim_timesteps(schedule.num_timesteps, 10, start=start))
    expected = (x / math.sqrt(schedule.alpha_bar(start))).clamp(-1.0, 1.0)
    assert torch.allclose(out, expected, atol=1e-5)


def test_parse_posterior_spec() -> None:
    name, kwargs = parse_posterior_spec("dps8=dps_mean,steps=50,samples=8,guidance=0.5,prior=0.2,newton=3,eta=1,seed=3")
    assert name == "dps8"
    assert kwargs == {
        "mode": "dps_mean", "num_steps": 50, "num_samples": 8, "guidance": 0.5,
        "prior_variance": 0.2, "newton_steps": 3, "eta": 1.0, "seed": 3,
    }
    with pytest.raises(ValueError, match="unknown posterior option"):
        parse_posterior_spec("a=dps,bogus=1")
    with pytest.raises(ValueError, match="NAME=mode"):
        parse_posterior_spec("a")
