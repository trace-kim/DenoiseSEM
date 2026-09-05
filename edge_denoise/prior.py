"""A diffusion prior over clean SEM crops, and posterior sampling from it given
one noisy frame -- the generative arm of the fine-feature study.

Why this exists.  Every regression arm in this package outputs (an approximation
of) the conditional mean E[x | y].  For structure whose single-frame evidence is
weak -- grain far below the noise, or a rare blemish -- the conditional mean
shrinks that structure toward the prior mean, which is what "too smooth" means.
A *sample* from the posterior p(x | y) does not shrink: it commits to a
plausible realization, so texture statistics and committed features look like
the clean image.  The price is well known (the perception-distortion trade-off):
a posterior sample has twice the expected squared error of the posterior mean,
and its committed fine details are drawn, not measured -- a retake commits to
different ones.  The point of this module is to *measure* that trade on the
metrology harness rather than argue it.

Components:

- :class:`PriorConfig` / :class:`PriorTrainer`: a standard DDPM (linear beta
  schedule, epsilon prediction, ``burst_diffusion.unet.UNet`` backbone with the
  real timestep conditioning) trained on random crops of the TRAIN split's
  clean images.  In synthetic-land the clean images are the long-dwell captures
  the noisy bursts were generated from; on an instrument the analogous data
  are long-dwell reference captures.  No noisy frame is ever a training target.
- :class:`PosteriorSampler`: DDIM (eta = 0) with the *exact* clipped-Poisson
  likelihood of the stored frames (counts k = 0..peak, the top bin being
  "k >= peak") applied as a per-pixel proximal step on the chain's x0
  estimate (DiffPIR-style), plus the cheap "start the chain at the
  measurement's noise level" variant (SDEdit-style; deterministic, no
  likelihood at all).  ``dps`` with a fixed start noise per call is a deterministic
  function of the frame; ``dps_mean`` averages K such samples and walks back
  toward the posterior mean.

Nothing here touches the locked test split.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.utils.tensorboard import SummaryWriter

from burst_diffusion.data import BurstCache
from burst_diffusion.ema import EMAHelper, ema_parameters
from burst_diffusion.unet import UNet

from .config import DataConfig, ModelConfig, TrainingConfig

logger = logging.getLogger("edge_denoise.prior")

CHECKPOINT_FORMAT = 1
CHECKPOINT_KIND = "edge_denoise_prior"
LATEST_CHECKPOINT_NAME = "ckpt_latest.pt"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DiffusionConfig(_StrictModel):
    """DDPM forward process: ``num_timesteps`` linear betas from ``beta_start``
    to ``beta_end`` (Ho et al. 2020 defaults)."""

    num_timesteps: int = Field(default=1000, ge=10)
    beta_start: float = Field(default=1.0e-4, gt=0.0)
    beta_end: float = Field(default=0.02, gt=0.0)
    flip_augment: bool = True

    @model_validator(mode="after")
    def _check(self) -> "DiffusionConfig":
        if self.beta_end <= self.beta_start:
            raise ValueError("diffusion.beta_end must exceed diffusion.beta_start")
        if self.beta_end >= 1.0:
            raise ValueError("diffusion.beta_end must be < 1")
        return self


class PriorConfig(_StrictModel):
    data: DataConfig
    diffusion: DiffusionConfig = DiffusionConfig()
    model: ModelConfig
    training: TrainingConfig

    @model_validator(mode="after")
    def _check_structure(self) -> "PriorConfig":
        image_size = self.data.image_size
        divisor = 2 ** (len(self.model.ch_mult) - 1)
        if image_size % divisor != 0:
            raise ValueError(
                f"data.image_size ({image_size}) must be divisible by {divisor} "
                "so every U-Net level lines up"
            )
        if self.training.init_checkpoint is not None:
            raise ValueError("training.init_checkpoint is not supported for the prior")
        return self


def load_prior_config(path: str | Path) -> PriorConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return PriorConfig.model_validate(raw)


# ---------------------------------------------------------------------------
# schedule


@dataclass(frozen=True)
class Schedule:
    betas: torch.Tensor  # [T]
    alphas_cumprod: torch.Tensor  # [T]

    @classmethod
    def from_config(cls, config: DiffusionConfig) -> "Schedule":
        betas = torch.linspace(
            config.beta_start, config.beta_end, config.num_timesteps, dtype=torch.float64
        )
        return cls(betas=betas, alphas_cumprod=torch.cumprod(1.0 - betas, dim=0))

    @property
    def num_timesteps(self) -> int:
        return int(self.betas.shape[0])

    def alpha_bar(self, t: int) -> float:
        """``alpha_bar_t`` for integer t in [-1, T-1]; t = -1 is the clean end (1.0)."""
        if t < 0:
            return 1.0
        return float(self.alphas_cumprod[t])

    def level_for_variance(self, noise_variance: float) -> int:
        """The timestep whose signal-to-noise matches a measurement with
        ``noise_variance`` (model units) on unit-scaled signal: the largest t
        with ``(1 - alpha_bar_t) / alpha_bar_t <= noise_variance``."""
        ratio = (1.0 - self.alphas_cumprod) / self.alphas_cumprod
        below = torch.nonzero(ratio <= noise_variance).flatten()
        return int(below[-1]) if below.numel() else 0


def ddim_timesteps(num_timesteps: int, num_steps: int, start: int | None = None) -> list[int]:
    """Strictly decreasing integer timesteps from ``start`` (default T-1) to 0."""
    top = num_timesteps - 1 if start is None else start
    if top < 0:
        return []
    count = max(1, min(num_steps, top + 1))
    raw = np.linspace(top, 0, count)
    steps = sorted({int(round(v)) for v in raw}, reverse=True)
    return steps


# ---------------------------------------------------------------------------
# data: random clean crops of the train split


class CleanCropFactory:
    def __init__(
        self,
        cache: BurstCache,
        *,
        image_size: int,
        batch_size: int,
        flip_augment: bool,
        seed: int,
    ):
        if not cache.train_sources:
            raise ValueError("cache has no training sources")
        self.sources = list(cache.train_sources)
        self.image_size = image_size
        self.batch_size = batch_size
        self.flip_augment = flip_augment
        self._rng = np.random.default_rng(seed)

    def sample_batch(self) -> torch.Tensor:
        size = self.image_size
        crops = []
        for _ in range(self.batch_size):
            source = self.sources[int(self._rng.integers(len(self.sources)))]
            height, width = source.clean.shape[:2]
            top = int(self._rng.integers(0, height - size + 1))
            left = int(self._rng.integers(0, width - size + 1))
            crop = source.clean[top : top + size, left : left + size].astype(np.float32) / 255.0
            if self.flip_augment:
                if self._rng.random() < 0.5:
                    crop = crop[:, ::-1]
                if self._rng.random() < 0.5:
                    crop = crop[::-1, :]
            crops.append(np.ascontiguousarray(crop) * 2.0 - 1.0)
        return torch.from_numpy(np.stack(crops))[:, None]

    def state_dict(self) -> dict:
        return {"rng_state": self._rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self._rng.bit_generator.state = state["rng_state"]


# ---------------------------------------------------------------------------
# model + checkpoints


def build_prior_unet(config: PriorConfig) -> UNet:
    return UNet(
        in_channels=1,
        out_ch=1,
        ch=config.model.ch,
        ch_mult=config.model.ch_mult,
        num_res_blocks=config.model.num_res_blocks,
        attn_resolutions=config.model.attn_resolutions,
        dropout=config.model.dropout,
        resamp_with_conv=config.model.resamp_with_conv,
        resolution=config.data.image_size,
        num_groups=config.model.effective_num_groups,
    )


def save_prior_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    ema: EMAHelper | None,
    optimizer: torch.optim.Optimizer,
    factory: CleanCropFactory,
    step: int,
    config: PriorConfig,
) -> None:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "kind": CHECKPOINT_KIND,
        "step": step,
        "config": config.model_dump(mode="json"),
        "model": model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "factory": factory.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_prior_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("format") != CHECKPOINT_FORMAT
        or payload.get("kind") != CHECKPOINT_KIND
    ):
        raise ValueError(f"{path} is not an edge_denoise prior checkpoint")
    return payload


class PriorTrainer:
    """DDPM training on clean train-split crops (epsilon prediction, MSE)."""

    def __init__(self, config: PriorConfig, *, resume_from: str | Path | None = None):
        self.config = config
        self.device = torch.device(
            "cuda" if config.training.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
        )
        if config.training.device == "cuda" and self.device.type != "cuda":
            raise RuntimeError("training.device is 'cuda' but CUDA is not available")
        self.run_dir = Path(config.training.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stop_file = self.run_dir / "stop"
        if self.stop_file.exists():
            self.stop_file.unlink()
        torch.manual_seed(config.training.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.training.seed)
        self.cache = BurstCache(
            config.data.dataset_dir,
            channels=config.data.channels,
            min_replicas=1,
            min_size=config.data.image_size,
            val_fraction=config.data.val_fraction,
            test_fraction=config.data.test_fraction,
            split_seed=config.data.split_seed,
        )
        self.factory = CleanCropFactory(
            self.cache,
            image_size=config.data.image_size,
            batch_size=config.training.batch_size,
            flip_augment=config.diffusion.flip_augment,
            seed=config.training.seed,
        )
        self.schedule = Schedule.from_config(config.diffusion)
        self.model = build_prior_unet(config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.training.lr,
            betas=(config.training.beta1, 0.999),
            eps=config.training.adam_eps,
            weight_decay=config.training.weight_decay,
        )
        self.ema: EMAHelper | None = None
        if config.training.ema:
            self.ema = EMAHelper(mu=config.training.ema_rate)
            self.ema.register(self.model)
        self.step = 0
        if resume_from is not None:
            payload = load_prior_checkpoint(resume_from, map_location=self.device)
            self.model.load_state_dict(payload["model"])
            self.optimizer.load_state_dict(payload["optimizer"])
            if self.ema is not None and payload["ema"] is not None:
                self.ema.load_state_dict(
                    {name: value.to(self.device) for name, value in payload["ema"].items()}
                )
            torch.set_rng_state(payload["torch_rng"].cpu())
            self.factory.load_state_dict(payload["factory"])
            self.step = int(payload["step"])
        (self.run_dir / "config.yml").write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
        )

    @property
    def latest_checkpoint_path(self) -> Path:
        return self.run_dir / LATEST_CHECKPOINT_NAME

    def _save(self, *, milestone: bool) -> Path:
        kwargs = dict(
            model=self.model, ema=self.ema, optimizer=self.optimizer,
            factory=self.factory, step=self.step, config=self.config,
        )
        save_prior_checkpoint(self.latest_checkpoint_path, **kwargs)
        if milestone:
            save_prior_checkpoint(self.run_dir / f"ckpt_{self.step:07d}.pt", **kwargs)
        return self.latest_checkpoint_path

    def _sample_grid(self, writer: SummaryWriter, count: int = 8, steps: int = 50) -> None:
        """A quick unconditional DDIM sample sheet for TensorBoard (visual check)."""
        self.model.eval()
        with ema_parameters(self.model, self.ema), torch.no_grad():
            generator = torch.Generator(device="cpu").manual_seed(1234)
            size = self.config.data.image_size
            x = torch.randn(count, 1, size, size, generator=generator).to(self.device)
            sampler = DDIMSampler(self.model, self.schedule, device=self.device)
            samples = sampler.sample(x, ddim_timesteps(self.schedule.num_timesteps, steps))
        grid = ((samples.clamp(-1, 1) + 1) / 2).cpu().numpy()
        writer.add_image("prior/samples", np.concatenate(list(grid), axis=-1), self.step)
        self.model.train()

    def run(self) -> Path:
        training = self.config.training
        writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
        alphas = self.schedule.alphas_cumprod.to(self.device, dtype=torch.float32)
        num_t = self.schedule.num_timesteps
        window: list[float] = []
        window_started = time.time()
        stop_reason: str | None = None
        self.model.train()
        try:
            while self.step < training.max_steps:
                if self.stop_file.exists():
                    stop_reason = f"stop file present: {self.stop_file}"
                    break
                x0 = self.factory.sample_batch().to(self.device)
                batch = x0.shape[0]
                # Antithetic timesteps (the legacy DDIM trick): t and T-1-t in one batch.
                half = torch.randint(0, num_t, (batch // 2 + 1,), device=self.device)
                t = torch.cat([half, num_t - 1 - half])[:batch]
                noise = torch.randn_like(x0)
                ab = alphas[t].view(-1, 1, 1, 1)
                x_t = ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise
                prediction = self.model(x_t, t.float())
                loss = F.mse_loss(prediction, noise)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), training.grad_clip)
                self.optimizer.step()
                if self.ema is not None:
                    self.ema.update(self.model)
                self.step += 1
                window.append(float(loss.item()))
                if self.step == 1 or self.step % training.log_every == 0:
                    elapsed = max(time.time() - window_started, 1e-9)
                    writer.add_scalar("train/loss", float(np.mean(window)), self.step)
                    writer.add_scalar("train/steps_per_sec", len(window) / elapsed, self.step)
                    logger.info(
                        "step %d | loss %.5f | %.1f steps/s",
                        self.step, float(np.mean(window)), len(window) / elapsed,
                    )
                    window.clear()
                    window_started = time.time()
                if self.step % training.val_every == 0 or self.step == training.max_steps:
                    self._sample_grid(writer)
                if self.step % training.checkpoint_every == 0 or self.step == training.max_steps:
                    self._save(milestone=True)
        except KeyboardInterrupt:
            stop_reason = "keyboard interrupt"
        finally:
            checkpoint = self._save(milestone=False)
            writer.close()
        if stop_reason is not None:
            logger.info("stopped early at step %d (%s)", self.step, stop_reason)
        else:
            logger.info("finished %d steps", self.step)
        return checkpoint


# ---------------------------------------------------------------------------
# likelihood of the stored frames


def counts_from_frames(frames01: torch.Tensor, peak: float) -> torch.Tensor:
    """Recover the clipped Poisson counts ``k`` from stored frames in [0, 1]
    (``noising_pipeline`` stores ``min(Pois(x*peak), peak)/peak``, 8-bit)."""
    return torch.round(frames01 * peak).clamp(0.0, peak)


def clipped_poisson_nll_map(counts: torch.Tensor, x01: torch.Tensor, peak: float) -> torch.Tensor:
    """Per-pixel negative log-likelihood of counts ``k`` given a clean estimate
    ``x01`` in [0, 1]: Poisson(lambda = x01 * peak) for k < peak, and the
    clipped top bin ``P(K >= peak)`` for k == peak.  Constants dropped."""
    lam = (x01.clamp(1e-4, 1.0) * peak).to(torch.float64)
    k = counts.to(torch.float64)
    kmax = float(peak)
    ordinary = lam - k * torch.log(lam)
    # log P(K <= kmax-1) via logsumexp over j = 0..kmax-1 of (j log lam - lam - log j!)
    js = torch.arange(0, int(kmax), dtype=torch.float64, device=lam.device)
    log_fact = torch.lgamma(js + 1.0)
    terms = js.view(*([1] * lam.dim()), -1) * torch.log(lam).unsqueeze(-1) - lam.unsqueeze(-1) - log_fact.view(
        *([1] * lam.dim()), -1
    )
    log_cdf = torch.logsumexp(terms, dim=-1)
    # log P(K >= kmax) = log(1 - exp(log_cdf)), stable via log1p(-exp(.))
    log_tail = torch.log1p(-torch.exp(log_cdf).clamp(max=1.0 - 1e-12))
    clipped = -log_tail
    return torch.where(k >= kmax - 0.5, clipped, ordinary)


def clipped_poisson_nll(counts: torch.Tensor, x01: torch.Tensor, peak: float) -> torch.Tensor:
    """:func:`clipped_poisson_nll_map` summed over pixels; returns ``[B]``."""
    return clipped_poisson_nll_map(counts, x01, peak).flatten(1).sum(dim=1).to(torch.float32)


# ---------------------------------------------------------------------------
# samplers


class DDIMSampler:
    def __init__(self, model: torch.nn.Module, schedule: Schedule, *, device: torch.device):
        self.model = model
        self.schedule = schedule
        self.device = device

    def predict(self, x: torch.Tensor, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        ab = self.schedule.alpha_bar(t)
        t_tensor = torch.full((x.shape[0],), float(t), device=x.device)
        eps = self.model(x, t_tensor)
        x0 = (x - math.sqrt(1.0 - ab) * eps) / math.sqrt(ab)
        return eps, x0

    @staticmethod
    def step(x0: torch.Tensor, eps: torch.Tensor, ab_prev: float) -> torch.Tensor:
        return math.sqrt(ab_prev) * x0 + math.sqrt(1.0 - ab_prev) * eps

    def sample(self, x: torch.Tensor, timesteps: list[int]) -> torch.Tensor:
        """Deterministic DDIM (eta = 0) from state ``x`` at ``timesteps[0]`` to clean."""
        with torch.no_grad():
            for index, t in enumerate(timesteps):
                t_prev = timesteps[index + 1] if index + 1 < len(timesteps) else -1
                eps, x0 = self.predict(x, t)
                x = self.step(x0.clamp(-1.0, 1.0), eps, self.schedule.alpha_bar(t_prev))
        return x


PosteriorMode = Literal["dps", "dps_mean", "sdedit"]


class PosteriorSampler:
    """Posterior samples p(x | one noisy frame) from a trained prior.

    - ``dps``: DDIM (eta = 0) from a fixed Gaussian start; at every step the
      prior's x0 estimate is replaced by the per-pixel proximal solution
      ``argmin_x NLL(k | x) + (x - x0)^2 / (2 r_t^2)`` (Newton on the exact
      clipped-Poisson likelihood), with ``r_t^2`` = the prior's posterior
      variance at that step (``prior_variance * s_t^2 / (prior_variance +
      s_t^2)``, ``s_t^2 = (1 - a_t)/a_t``) times ``guidance``.  This is the
      DiffPIR / proximal form of posterior sampling: the measurement pulls
      hard while the chain is noisy and fades out as ``r_t^2 -> 0``, so the
      finest structure is committed by the prior.  A plain gradient step on
      this likelihood is numerically stiff (its gradient grows like 1/x at
      low intensity) and was found to collapse; the proximal step is exact
      per pixel and stable.  Deterministic given (frame, seed).
    - ``dps_mean``: the average of ``num_samples`` ``dps`` samples with
      different start noise -- a Monte-Carlo posterior mean.
    - ``sdedit``: place the frame at the timestep whose noise matches the
      single-frame variance and run plain DDIM down (no gradients).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        schedule: Schedule,
        *,
        device: torch.device,
        peak: float = 10.0,
        mode: PosteriorMode = "dps",
        num_steps: int = 100,
        num_samples: int = 4,
        guidance: float = 1.0,
        prior_variance: float = 0.1,
        newton_steps: int = 5,
        eta: float = 0.0,
        seed: int = 0,
        noise_variance: float | None = None,
    ):
        if mode not in ("dps", "dps_mean", "sdedit"):
            raise ValueError(f"unknown posterior mode {mode!r}")
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        self.model = model.eval()
        self.schedule = schedule
        self.device = device
        self.peak = peak
        self.mode = mode
        self.num_steps = num_steps
        self.num_samples = num_samples if mode in ("dps_mean", "sdedit") else 1
        self.guidance = guidance
        # Variance of the clean crops in model units (MIIC: intensities in
        # [0.15, 0.85] -> [-0.7, 0.7], std ~0.3): the Gaussian-prior stand-in
        # that sets how much the measurement may move the x0 estimate.
        self.prior_variance = prior_variance
        self.newton_steps = newton_steps
        # DDIM stochasticity: 0 = deterministic probability-flow steps, 1 =
        # DDPM-style fresh noise at every step.  Fresh noise matters for the
        # likelihood-guided chain: the proximal pull copies part of the frame's
        # noise into the x0 estimate, and only a re-noised state lets the prior
        # treat that copy as noise again at the next step.  With ``seed`` fixed
        # the chain stays a deterministic function of the frame.
        self.eta = eta
        self.seed = seed
        # Single-frame Poisson variance in model units at the corpus mean
        # intensity (~0.4 at peak 10 -> 4 * 0.04 = 0.16); an SDEdit input.
        self.noise_variance = 0.16 if noise_variance is None else noise_variance
        self.ddim = DDIMSampler(model, schedule, device=device)

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, device: str = "auto", use_ema: bool = True, **kwargs) -> "PosteriorSampler":
        resolved = torch.device("cuda" if device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
        payload = load_prior_checkpoint(path, map_location=resolved)
        config = PriorConfig.model_validate(payload["config"])
        model = build_prior_unet(config)
        model.load_state_dict(payload["model"])
        if use_ema and payload.get("ema"):
            named = dict(model.named_parameters())
            for name, value in payload["ema"].items():
                named[name].data.copy_(value)
        model = model.to(resolved)
        return cls(model, Schedule.from_config(config.diffusion), device=resolved, **kwargs)

    # -- the three estimators ----------------------------------------------

    def _step(
        self,
        x0: torch.Tensor,
        eps: torch.Tensor,
        ab: float,
        ab_prev: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """DDIM update with stochasticity ``eta`` (Song et al. 2021, eq. 16)."""
        if ab_prev >= 1.0:
            return x0
        sigma = self.eta * math.sqrt((1.0 - ab_prev) / (1.0 - ab)) * math.sqrt(1.0 - ab / ab_prev)
        sigma = min(sigma, math.sqrt(1.0 - ab_prev))
        direction = math.sqrt(max(1.0 - ab_prev - sigma**2, 0.0)) * eps
        noise = torch.randn(x0.shape, generator=generator).to(x0.device) if sigma > 0.0 else 0.0
        return math.sqrt(ab_prev) * x0 + direction + sigma * noise

    def _proximal(self, counts: torch.Tensor, x0_prior: torch.Tensor, r2: float) -> torch.Tensor:
        """Per-pixel ``argmin_x NLL(k | x) + (x - x0_prior)^2 / (2 r2)`` by damped
        Newton iterations (the objective is separable and convex per pixel)."""
        x = x0_prior.detach().clone()
        for _ in range(self.newton_steps):
            x = x.detach().requires_grad_(True)
            x01 = (x + 1.0) / 2.0
            objective = clipped_poisson_nll_map(counts, x01, self.peak).to(torch.float32) + (
                (x - x0_prior) ** 2 / (2.0 * r2)
            )
            (gradient,) = torch.autograd.grad(objective.sum(), x, create_graph=True)
            (curvature,) = torch.autograd.grad(gradient.sum(), x)
            with torch.no_grad():
                step = gradient / curvature.clamp_min(1e-3)
                x = (x - step.clamp(-0.5, 0.5)).clamp(-1.0, 1.0)
        return x.detach()

    def _dps(self, counts: torch.Tensor, start: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        timesteps = ddim_timesteps(self.schedule.num_timesteps, self.num_steps)
        x = start
        for index, t in enumerate(timesteps):
            t_prev = timesteps[index + 1] if index + 1 < len(timesteps) else -1
            ab = self.schedule.alpha_bar(t)
            with torch.no_grad():
                _, x0 = self.ddim.predict(x, t)
                x0 = x0.clamp(-1.0, 1.0)
            sigma2 = (1.0 - ab) / ab
            r2 = self.guidance * self.prior_variance * sigma2 / (self.prior_variance + sigma2)
            x0_post = self._proximal(counts, x0, max(r2, 1e-6))
            with torch.no_grad():
                eps_post = (x - math.sqrt(ab) * x0_post) / math.sqrt(max(1.0 - ab, 1e-12))
                x = self._step(x0_post, eps_post, ab, self.schedule.alpha_bar(t_prev), generator)
        return x.detach()

    def _sdedit(self, frames: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        start_t = self.schedule.level_for_variance(self.noise_variance)
        timesteps = ddim_timesteps(self.schedule.num_timesteps, self.num_steps, start=start_t)
        x = math.sqrt(self.schedule.alpha_bar(start_t)) * frames
        with torch.no_grad():
            for index, t in enumerate(timesteps):
                t_prev = timesteps[index + 1] if index + 1 < len(timesteps) else -1
                eps, x0 = self.ddim.predict(x, t)
                x = self._step(x0.clamp(-1.0, 1.0), eps, self.schedule.alpha_bar(t), self.schedule.alpha_bar(t_prev), generator)
        return x

    def denoise(self, frames: torch.Tensor) -> torch.Tensor:
        """``[B, 1, S, S]`` frames in [-1, 1] -> estimates in [-1, 1] on CPU."""
        if frames.dim() != 4:
            raise ValueError(f"frames must be [B, C, H, W], got {tuple(frames.shape)}")
        frames = frames.to(device=self.device, dtype=torch.float32)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        accumulator = torch.zeros_like(frames)
        if self.mode == "sdedit":
            for _ in range(self.num_samples):
                accumulator += self._sdedit(frames, generator)
        else:
            counts = counts_from_frames((frames + 1.0) / 2.0, self.peak)
            for _ in range(self.num_samples):
                start = torch.randn(frames.shape, generator=generator).to(self.device)
                accumulator += self._dps(counts, start, generator)
        return (accumulator / self.num_samples).clamp(-1.0, 1.0).cpu()

    def denoise01(self, frames01: list[np.ndarray], *, max_batch: int = 10) -> list[np.ndarray]:
        outputs: list[np.ndarray] = []
        for start in range(0, len(frames01), max_batch):
            chunk = frames01[start : start + max_batch]
            stacked = np.stack([np.ascontiguousarray(f.transpose(2, 0, 1)) for f in chunk]).astype(np.float32)
            estimate = self.denoise(torch.from_numpy(stacked * 2.0 - 1.0))
            array01 = ((estimate + 1.0) / 2.0).numpy()
            outputs.extend(
                np.ascontiguousarray(array01[i].transpose(1, 2, 0)).astype(np.float64)
                for i in range(array01.shape[0])
            )
        return outputs


def parse_posterior_spec(spec: str) -> tuple[str, dict]:
    """``NAME=mode[,key=value...]`` -> (name, kwargs for PosteriorSampler).

    Keys: steps, samples, guidance, prior, newton, eta, seed, variance.  Example:
    ``dps8=dps_mean,steps=100,samples=8,guidance=1.0``.
    """
    name, _, rest = spec.partition("=")
    if not rest:
        raise ValueError(f"posterior arm {spec!r} needs NAME=mode[,key=value...]")
    parts = [p.strip() for p in rest.split(",") if p.strip()]
    kwargs: dict = {"mode": parts[0]}
    casts = {
        "steps": ("num_steps", int),
        "samples": ("num_samples", int),
        "guidance": ("guidance", float),
        "prior": ("prior_variance", float),
        "newton": ("newton_steps", int),
        "eta": ("eta", float),
        "seed": ("seed", int),
        "variance": ("noise_variance", float),
    }
    for part in parts[1:]:
        key, _, value = part.partition("=")
        if key not in casts or not value:
            raise ValueError(f"unknown posterior option {part!r} (keys: {sorted(casts)})")
        field, cast = casts[key]
        kwargs[field] = cast(value)
    return name.strip(), kwargs
