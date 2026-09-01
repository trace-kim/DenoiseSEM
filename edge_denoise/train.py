"""Training loop for the edge_denoise regression family.

Mirrors burst_diffusion's trainer where the concerns are identical (Adam with
beta2 = 0.999, gradient clipping, per-step EMA, keyed atomic checkpoints that
restore all RNG state for exact resume, no DataLoader) and differs only in the
objective:

    L = lambda_image    * d(f(y1), target)                (image / hybrid)
      + lambda_gradient * d(S f(y1), S target)            (S = sobel; for the
                                                           'gradient' representation
                                                           the output already lives
                                                           there: d(f, S target))
      + lambda_consistency * |f(y1) - f(y2)|^2            (two independent frames)

with d(.) mean-reduced L2 or L1 and target either the clean crop or a fresh
noisy frame (Noise2Noise).

Expectation setting for ``target: noisy`` -- the loss cannot fall below the
target-noise floor, so a plateau is correct behavior, not divergence:

- image term floor  ~ 4 * sigma01^2                        (model range is 2x [0, 1])
- gradient term floor ~ 4 * sigma01^2 * 12/64              (Sobel white-noise gain;
                                                           mean over the 2 channels)

For the MIIC peak-10 data (sigma01^2 ~ 0.0385) that predicts ~0.154 for a pure
N2N arm and ~0.154 * (lambda_image + 0.1875 * lambda_gradient) in general.
Progress is measured by ``val/psnr`` (denoised image vs clean) and by
``val/consistency_sigma`` -- the direct repeatability readout: the RMS
disagreement between the denoised versions of two independent frames of the
same scene, sqrt(E|f(y1) - f(y2)|^2 / 2), in [0, 1] intensity units.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from burst_diffusion.data import BurstCache
from burst_diffusion.ema import EMAHelper, ema_parameters
from burst_diffusion.metrics import psnr

from .config import Config
from .data import PairFactory
from .gradient import sobel
from .model import EdgeDenoiser, build_model

logger = logging.getLogger("edge_denoise.train")

CHECKPOINT_FORMAT = 1
CHECKPOINT_KIND = "edge_denoise"
LATEST_CHECKPOINT_NAME = "ckpt_latest.pt"


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device is 'cuda' but CUDA is not available")
    return torch.device(requested)


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    ema: EMAHelper | None,
    optimizer: torch.optim.Optimizer,
    factory: PairFactory,
    step: int,
    config: Config,
) -> None:
    """Atomically write a keyed checkpoint (tensors + primitives only)."""
    payload = {
        "format": CHECKPOINT_FORMAT,
        "kind": CHECKPOINT_KIND,
        "step": step,
        "config": config.model_dump(mode="json"),
        "model": model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "factory": factory.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("format") != CHECKPOINT_FORMAT
        or payload.get("kind") != CHECKPOINT_KIND
    ):
        raise ValueError(
            f"unrecognized checkpoint in {path}; expected an edge_denoise "
            f"format-{CHECKPOINT_FORMAT} checkpoint (a burst_diffusion checkpoint "
            "is a different pipeline -- use burst_diffusion to load it)"
        )
    return payload


class Trainer:
    def __init__(self, config: Config, *, resume_from: str | Path | None = None):
        self.config = config
        self.device = resolve_device(config.training.device)
        self.run_dir = Path(config.training.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stop_file = self.run_dir / "stop"
        if self.stop_file.exists():
            logger.info("removing leftover stop file %s", self.stop_file)
            self.stop_file.unlink()

        torch.manual_seed(config.training.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.training.seed)

        self.cache = BurstCache(
            config.data.dataset_dir,
            channels=config.data.channels,
            min_replicas=config.min_replicas,
            min_size=config.data.image_size,
            val_fraction=config.data.val_fraction,
            test_fraction=config.data.test_fraction,
            split_seed=config.data.split_seed,
        )
        summary = self.cache.summary()
        logger.info(
            "dataset: %d train / %d val sources, min %d frames, %.0f MB cached",
            summary["train_sources"],
            summary["val_sources"],
            summary["min_frames"],
            summary["ram_bytes"] / 1e6,
        )
        objective = config.objective
        self.factory = PairFactory(
            self.cache,
            image_size=config.data.image_size,
            batch_size=config.training.batch_size,
            target=objective.target,
            need_second=objective.lambda_consistency > 0.0,
            seed=config.training.seed,
        )
        self.model: EdgeDenoiser = build_model(config).to(self.device)
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
            self._restore(Path(resume_from))

        (self.run_dir / "config.yml").write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True),
            encoding="utf-8",
        )

    def _restore(self, checkpoint_path: Path) -> None:
        payload = load_checkpoint(checkpoint_path, map_location=self.device)
        stored_config = Config.model_validate(payload["config"])
        if stored_config != self.config:
            import warnings

            warnings.warn(
                f"checkpoint {checkpoint_path} was written with a different config; "
                "resuming with the CURRENT config",
                stacklevel=2,
            )
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if self.ema is not None:
            if payload["ema"] is None:
                raise ValueError("config enables EMA but the checkpoint has no EMA state")
            self.ema.load_state_dict(
                {name: value.to(self.device) for name, value in payload["ema"].items()}
            )
        torch.set_rng_state(payload["torch_rng"].cpu())
        if payload.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])
        self.factory.load_state_dict(payload["factory"])
        self.step = int(payload["step"])
        logger.info("resumed from %s at step %d", checkpoint_path, self.step)

    @property
    def latest_checkpoint_path(self) -> Path:
        return self.run_dir / LATEST_CHECKPOINT_NAME

    def _save(self, *, milestone: bool) -> Path:
        save_checkpoint(
            self.latest_checkpoint_path,
            model=self.model,
            ema=self.ema,
            optimizer=self.optimizer,
            factory=self.factory,
            step=self.step,
            config=self.config,
        )
        if milestone:
            save_checkpoint(
                self.run_dir / f"ckpt_{self.step:07d}.pt",
                model=self.model,
                ema=self.ema,
                optimizer=self.optimizer,
                factory=self.factory,
                step=self.step,
                config=self.config,
            )
        return self.latest_checkpoint_path

    def _loss_terms(
        self,
        prediction: torch.Tensor,
        targets: torch.Tensor,
        second: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Per-term mean losses; keys are absent when their weight is zero."""
        objective = self.config.objective

        def distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            if objective.loss == "l1":
                return (a - b).abs().mean()
            return ((a - b) ** 2).mean()

        terms: dict[str, torch.Tensor] = {}
        if objective.representation == "gradient":
            terms["gradient"] = distance(prediction, sobel(targets))
        else:
            if objective.lambda_image > 0.0:
                terms["image"] = distance(prediction, targets)
            if objective.lambda_gradient > 0.0:
                terms["gradient"] = distance(sobel(prediction), sobel(targets))
        if objective.lambda_consistency > 0.0:
            if second is None:
                raise RuntimeError("consistency loss requires a second realization")
            prediction_second = self.model(second)
            terms["consistency"] = ((prediction - prediction_second) ** 2).mean()
        return terms

    def _combine(self, terms: dict[str, torch.Tensor]) -> torch.Tensor:
        objective = self.config.objective
        weights = {
            "image": objective.lambda_image,
            "gradient": objective.lambda_gradient,
            "consistency": objective.lambda_consistency,
        }
        total = None
        for name, value in terms.items():
            weighted = weights[name] * value
            total = weighted if total is None else total + weighted
        assert total is not None
        return total

    def _validate(self, writer: SummaryWriter) -> None:
        if not self.cache.val_sources:
            return
        count = self.config.training.val_images
        self.model.eval()
        with ema_parameters(self.model, self.ema), torch.no_grad():
            batch = self.factory.val_batch(count=count)
            inputs = batch.inputs.to(self.device)
            targets = batch.targets.to(self.device)
            second = batch.second.to(self.device)

            prediction = self.model(inputs)
            terms = self._loss_terms(
                prediction,
                targets,
                second if self.config.objective.lambda_consistency > 0.0 else None,
            )
            writer.add_scalar("val/loss", float(self._combine(terms).item()), self.step)

            denoised = self.model.predict_image(inputs).clamp(-1.0, 1.0)
            denoised_second = self.model.predict_image(second).clamp(-1.0, 1.0)
            clean01 = ((batch.clean + 1.0) / 2.0).numpy()
            pred01 = ((denoised + 1.0) / 2.0).cpu().numpy()
            psnr_values = [
                psnr(np.moveaxis(clean01[i], 0, -1), np.moveaxis(pred01[i], 0, -1))
                for i in range(pred01.shape[0])
            ]
            writer.add_scalar("val/psnr", float(np.mean(psnr_values)), self.step)

            # Edge fidelity: MSE of the denoised image's Sobel field vs clean's.
            grad_mse = ((sobel(denoised) - sobel(batch.clean.to(self.device))) ** 2).mean()
            writer.add_scalar("val/gradient_mse_vs_clean", float(grad_mse.item()), self.step)

            # Live repeatability readout: RMS disagreement of two independent
            # frames' denoised images, in [0, 1] units (model range is 2x).
            consistency = ((denoised - denoised_second) ** 2).mean()
            writer.add_scalar(
                "val/consistency_sigma",
                float(torch.sqrt(consistency / 2.0).item()) / 2.0,
                self.step,
            )

            shown = min(4, pred01.shape[0])
            input01 = ((batch.inputs.clamp(-1.0, 1.0) + 1.0) / 2.0).numpy()
            rows = [
                np.concatenate([input01[i], pred01[i], clean01[i]], axis=-1)
                for i in range(shown)
            ]
            writer.add_image("val/input_pred_clean", np.concatenate(rows, axis=-2), self.step)
        self.model.train()

    def run(self) -> Path:
        training = self.config.training
        writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
        window_losses: dict[str, list[float]] = {}
        window_started = time.time()
        window_steps = 0
        stop_reason: str | None = None
        self.model.train()
        try:
            while self.step < training.max_steps:
                if self.stop_file.exists():
                    stop_reason = f"stop file present: {self.stop_file}"
                    break
                batch = self.factory.sample_batch()
                inputs = batch.inputs.to(self.device)
                targets = batch.targets.to(self.device)
                second = batch.second.to(self.device) if batch.second is not None else None

                prediction = self.model(inputs)
                terms = self._loss_terms(prediction, targets, second)
                loss = self._combine(terms)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), training.grad_clip)
                self.optimizer.step()
                if self.ema is not None:
                    self.ema.update(self.model)
                self.step += 1
                window_steps += 1

                window_losses.setdefault("total", []).append(float(loss.item()))
                for name, value in terms.items():
                    window_losses.setdefault(name, []).append(float(value.item()))

                if self.step == 1 or self.step % training.log_every == 0:
                    elapsed = max(time.time() - window_started, 1e-9)
                    writer.add_scalar(
                        "train/loss", float(np.mean(window_losses["total"])), self.step
                    )
                    for name in ("image", "gradient", "consistency"):
                        if name in window_losses:
                            writer.add_scalar(
                                f"train/loss_{name}",
                                float(np.mean(window_losses[name])),
                                self.step,
                            )
                    writer.add_scalar("train/steps_per_sec", window_steps / elapsed, self.step)
                    logger.info(
                        "step %d | loss %.5f | %.1f steps/s",
                        self.step,
                        float(np.mean(window_losses["total"])),
                        window_steps / elapsed,
                    )
                    window_losses.clear()
                    window_steps = 0
                    window_started = time.time()

                if self.step % training.val_every == 0 or self.step == training.max_steps:
                    self._validate(writer)
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
