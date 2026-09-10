"""Training loop for the edge_denoise regression family.

Mirrors burst_diffusion's trainer where the concerns are identical (Adam with
beta2 = 0.999, gradient clipping, per-step EMA, keyed atomic checkpoints that
restore all RNG state for exact resume, no DataLoader) and differs only in the
objective:

    L = lambda_image    * d(f(y1), target)                (image / hybrid)
      + lambda_gradient * d(S f(y1), S gtarget)           (S = sobel; for the
                                                           'gradient' representation
                                                           the output already lives
                                                           there: d(f, S gtarget))
      + lambda_consistency * |f(y1) - f(y2)|^2            (two independent frames)

with d(.) mean-reduced L2 or L1 and target either the clean crop or a fresh
noisy frame (Noise2Noise).  ``gtarget`` defaults to ``target`` and is
overridden by ``objective.gradient_target`` (clean oracle, leave-one-out
noisy mean, or a precomputed distillation image) -- the target-ladder
experiment; see the data module for the statistics of each choice.

Expectation setting for ``target: noisy`` -- the loss cannot fall below the
target-noise floor, so a plateau is correct behavior, not divergence:

- image term floor  ~ 4 * sigma01^2                        (model range is 2x [0, 1])
- gradient term floor ~ 4 * sigma01^2 * 12/64              (Sobel white-noise gain;
                                                           mean over the 2 channels)

For the MIIC peak-10 data (sigma01^2 ~ 0.0385) that predicts ~0.154 for a pure
N2N arm and ~0.154 * (lambda_image + 0.1875 * lambda_gradient) in general.
With ``target: noisy_mean`` (leave-one-out mean of the other N-1 replicas) the
same floors divide by N-1 (~0.0103 for the image term at N = 16).
Progress is measured by ``val/psnr`` (denoised image vs clean) and by
``val/consistency_sigma`` -- the direct repeatability readout: the RMS
disagreement between the denoised versions of two independent frames of the
same scene, sqrt(E|f(y1) - f(y2)|^2 / 2), in [0, 1] intensity units.
"""

from __future__ import annotations

import logging
import os
import signal
import shutil
import time
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

from burst_diffusion.data import BurstCache
from burst_diffusion.ema import EMAHelper, ema_parameters
from burst_diffusion.metrics import psnr
from runctl.control import StopController, configure_reproducibility

from .config import Config
from .data import PairFactory
from .distributed import DistributedRuntime
from .fusion import FusionBatch, FusionFactory, build_alignment, warp_prediction
from .gradient import sobel
from .model import EdgeDenoiser, build_model

logger = logging.getLogger("edge_denoise.train")

CHECKPOINT_FORMAT = 1
CHECKPOINT_KIND = "edge_denoise"
LATEST_CHECKPOINT_NAME = "ckpt_latest.pt"
TENSORBOARD_DIR_NAME = "tb"
# Everything a run writes into its run_dir.  A fresh run refuses to start on
# top of these (a second run would overwrite the checkpoints and provenance
# while TensorBoard merged both histories -- exactly what happened to the
# 2026-09-02 ft_consist run); ``overwrite=True`` deletes them first.
RUN_ARTIFACT_GLOBS = ("ckpt_*.pt", "provenance.json", "config.yml", TENSORBOARD_DIR_NAME)


def existing_run_artifacts(run_dir: str | Path) -> list[Path]:
    """Run artifacts already present in ``run_dir`` (empty for a fresh dir)."""
    root = Path(run_dir)
    if not root.is_dir():
        return []
    found: list[Path] = []
    for pattern in RUN_ARTIFACT_GLOBS:
        found.extend(sorted(root.glob(pattern)))
    return found


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
    distributed_state: dict | None = None,
    dataset_fingerprint: str | None = None,
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
        "cuda_rng": (None if distributed_state is not None else
                     torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        "factory": factory.state_dict(),
    }
    if distributed_state is not None:
        payload["distributed"] = distributed_state
    if dataset_fingerprint is not None:
        payload["dataset_fingerprint"] = dataset_fingerprint
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


def load_init_weights(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, torch.Tensor]:
    """Weights for ``training.init_checkpoint``: a warm start, not a resume.

    Takes the EMA weights when present (they are what every evaluation runs)
    overlaid on the live state dict (which also carries any buffers), from
    either pipeline's checkpoint: an edge_denoise payload maps 1:1, while a
    burst_diffusion payload stores the bare U-Net so its keys gain the
    ``unet.`` prefix of :class:`~edge_denoise.model.EdgeDenoiser`.  A backbone
    mismatch surfaces as a strict ``load_state_dict`` error at the caller.
    """
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"{path} is not a training checkpoint (no 'model' state)")
    state = dict(payload["model"])
    if payload.get("ema"):
        state.update(payload["ema"])
    if payload.get("kind") != CHECKPOINT_KIND:
        state = {f"unet.{name}": value for name, value in state.items()}
    return state


class Trainer:
    def __init__(
        self,
        config: Config,
        *,
        resume_from: str | Path | None = None,
        overwrite: bool = False,
    ):
        self.config = config
        self.runtime = DistributedRuntime(config.training.device)
        self.device = self.runtime.device
        if config.training.precision == "bf16" and self.device.type != "cuda":
            raise ValueError("training.precision=bf16 requires CUDA; use fp32 for CPU runs")
        if config.training.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("this CUDA device does not support bfloat16")
        self.run_dir = Path(config.training.run_dir)
        if resume_from is None:
            self.runtime.on_primary(lambda: self._claim_run_dir(overwrite=overwrite))
        self.runtime.on_primary(lambda: self.run_dir.mkdir(parents=True, exist_ok=True))
        self.stop_file = self.run_dir / "stop"
        self.runtime.on_primary(lambda: self.stop_file.unlink(missing_ok=True))
        self.stop_controller = StopController(self.stop_file)

        configure_reproducibility(config.training.seed, "repeatable")

        logger.info("loading dataset %s (prepared real arrays are content-verified)", config.data.dataset_dir)
        self.cache = BurstCache(
            config.data.dataset_dir,
            channels=config.data.channels,
            min_replicas=config.min_replicas,
            min_size=config.data.image_size,
            val_fraction=config.data.val_fraction,
            test_fraction=config.data.test_fraction,
            split_seed=config.data.split_seed,
        )
        if self.cache.real_metadata is not None:
            levels = self.cache.real_metadata["normalization"]
            if config.data.white_level is not None and (
                    config.data.white_level != levels["white"] or config.data.black_level != levels["black"]):
                raise ValueError("config normalization differs from the prepared real dataset")
            config = config.model_copy(update={"data": config.data.model_copy(update={
                "black_level": levels["black"], "white_level": levels["white"],
            })})
            self.config = config
        elif config.data.white_level is not None:
            raise ValueError("explicit intensity levels require a prepared real SEM dataset")
        summary = self.cache.summary()
        logger.info(
            "dataset: %d train / %d val sources, min %d frames, %.0f MB cached",
            summary["train_sources"],
            summary["val_sources"],
            summary["min_frames"],
            summary["ram_bytes"] / 1e6,
        )
        objective = config.objective
        self.factory: PairFactory | FusionFactory
        factory_seed = config.training.seed + self.runtime.rank
        if self.cache.real_metadata is not None:
            from .real_data import RealPairFactory

            self.factory = RealPairFactory(self.cache, config, seed=factory_seed)
        elif objective.fusion is not None:
            self.factory = FusionFactory(
                self.cache,
                image_size=config.data.image_size,
                batch_size=config.training.batch_size,
                fusion=objective.fusion,
                alignment=build_alignment(config),
                seed=factory_seed,
            )
            logger.info(
                "burst fusion: %d frames per burst, levels %s, align %s",
                objective.fusion.frames_per_burst,
                objective.fusion.levels,
                objective.fusion.align,
            )
        else:
            self.factory = PairFactory(
                self.cache,
                image_size=config.data.image_size,
                batch_size=config.training.batch_size,
                target=objective.target,
                need_second=objective.lambda_consistency > 0.0,
                gradient_target=objective.gradient_target,
                gradient_target_dir=objective.gradient_target_dir,
                seed=factory_seed,
                defect_augment=config.training.defect_augment,
                target_debias_peak=objective.target_debias_peak,
            )
        self.model: EdgeDenoiser = build_model(config).to(self.device)
        if config.training.init_checkpoint is not None and resume_from is None:
            state = load_init_weights(config.training.init_checkpoint, map_location=self.device)
            try:
                self.model.load_state_dict(state)
            except RuntimeError as error:
                raise ValueError(
                    f"training.init_checkpoint {config.training.init_checkpoint} does not "
                    f"match this config's backbone: {error}"
                ) from error
            logger.info("warm start: weights from %s", config.training.init_checkpoint)
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

        self.train_model = self.model
        if self.runtime.enabled:
            self.train_model = DistributedDataParallel(
                self.model, device_ids=[self.runtime.local_rank] if self.device.type == "cuda" else None,
                broadcast_buffers=False,
            )
            # Model initialization is identical; stochastic training and patch
            # sampling use independent streams on each rank.
            torch.manual_seed(config.training.seed + self.runtime.rank)
        if resume_from is not None:
            self._restore(Path(resume_from))

        self.runtime.on_primary(lambda: (self.run_dir / "config.yml").write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True),
            encoding="utf-8",
        ))

    def _claim_run_dir(self, *, overwrite: bool) -> None:
        """Refuse to start a fresh run on top of an existing one.

        Without ``overwrite`` an occupied run_dir raises ``FileExistsError``
        (resume instead, or choose another directory).  With it, the previous
        run's own artifacts -- and only those -- are deleted so the new run's
        checkpoints, provenance and TensorBoard history are unambiguous.
        """
        existing = existing_run_artifacts(self.run_dir)
        if not existing:
            return
        if not overwrite:
            names = ", ".join(path.name for path in existing)
            raise FileExistsError(
                f"run_dir {self.run_dir} already holds a run ({names}); resume it, "
                "pick another run_dir, or pass overwrite=True / --overwrite to start over"
            )
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        logger.warning(
            "overwrite: removed %d artifact(s) of the previous run in %s",
            len(existing),
            self.run_dir,
        )

    def _restore(self, checkpoint_path: Path) -> None:
        payload = load_checkpoint(checkpoint_path, map_location=self.device)
        if payload.get("dataset_fingerprint") != getattr(self.cache, "real_fingerprint", None):
            raise ValueError("resume dataset differs from the checkpoint; use init_checkpoint for a new dataset")
        stored_config = Config.model_validate(payload["config"])
        if stored_config != self.config:
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
        self.step = int(payload["step"])
        distributed = payload.get("distributed")
        old_world = distributed["world_size"] if distributed is not None else 1
        if old_world != self.runtime.world_size:
            warnings.warn("GPU count changed: restoring model/optimizer/EMA with new RNG streams; resume is not bit-for-bit", stacklevel=2)
            seed = self.config.training.seed + self.runtime.rank + self.step * 1_000_003
            torch.manual_seed(seed)
            self.factory.load_state_dict({"rng_state": np.random.default_rng(seed).bit_generator.state})
        elif distributed is not None:
            state = distributed["ranks"][self.runtime.rank]
            self.factory.load_state_dict(state["factory"])
            torch.set_rng_state(state["torch_rng"].cpu())
            if self.device.type == "cuda" and state["cuda_rng"] is not None:
                torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.device)
        else:
            torch.set_rng_state(payload["torch_rng"].cpu())
            if payload.get("cuda_rng") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])
            self.factory.load_state_dict(payload["factory"])
        logger.info("resumed from %s at step %d", checkpoint_path, self.step)

    @property
    def latest_checkpoint_path(self) -> Path:
        return self.run_dir / LATEST_CHECKPOINT_NAME

    def _save(self, *, milestone: bool) -> Path:
        state = self.runtime.gather_state(self.factory, self.effective_batch)

        def write() -> None:
            paths = [self.latest_checkpoint_path]
            if milestone:
                paths.append(self.run_dir / f"ckpt_{self.step:07d}.pt")
            for path in paths:
                save_checkpoint(path, model=self.model, ema=self.ema, optimizer=self.optimizer,
                                factory=self.factory, step=self.step, config=self.config,
                                distributed_state=state,
                                dataset_fingerprint=getattr(self.cache, "real_fingerprint", None))
        self.runtime.on_primary(write)
        return self.latest_checkpoint_path

    @property
    def effective_batch(self) -> int:
        training = self.config.training
        return training.batch_size * training.accumulation_steps * self.runtime.world_size

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                              enabled=self.config.training.precision == "bf16")

    def _pair_step(self, batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        inputs, targets = batch.inputs.to(self.device), batch.targets.to(self.device)
        second = batch.second.to(self.device) if batch.second is not None else None
        with self._autocast():
            if second is not None:
                # One DDP forward per backward, also for the two-view loss.
                prediction, prediction_second = self.train_model(torch.cat([inputs, second])).chunk(2)
            else:
                prediction, prediction_second = self.train_model(inputs), None
        terms = self._loss_terms(
            prediction.float(), targets, second,
            batch.gradient_targets.to(self.device) if batch.gradient_targets is not None else None,
            prediction_second=None if prediction_second is None else prediction_second.float(),
            second_shifts=None if batch.second_shifts is None else batch.second_shifts.to(self.device),
            margin=batch.loss_margin,
        )
        return prediction, terms

    def _loss_terms(
        self,
        prediction: torch.Tensor,
        targets: torch.Tensor,
        second: torch.Tensor | None,
        gradient_targets: torch.Tensor | None = None,
        *,
        prediction_second: torch.Tensor | None = None,
        second_shifts: torch.Tensor | None = None,
        margin: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Per-term mean losses; keys are absent when their weight is zero."""
        objective = self.config.objective
        reference = targets if gradient_targets is None else gradient_targets

        def distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            if margin:
                a, b = a[..., margin:-margin, margin:-margin], b[..., margin:-margin, margin:-margin]
            if objective.loss == "l1":
                return (a - b).abs().mean()
            return ((a - b) ** 2).mean()

        terms: dict[str, torch.Tensor] = {}
        if objective.representation == "gradient":
            terms["gradient"] = distance(prediction, sobel(reference))
        else:
            if objective.lambda_image > 0.0:
                terms["image"] = distance(prediction, targets)
            if objective.lambda_gradient > 0.0:
                terms["gradient"] = distance(sobel(prediction), sobel(reference))
        if objective.lambda_consistency > 0.0:
            if second is None:
                raise RuntimeError("consistency loss requires a second realization")
            if prediction_second is None:
                prediction_second = self.model(second).float()
            if second_shifts is not None:
                prediction_second = warp_prediction(prediction_second, second_shifts)
            difference = prediction - prediction_second
            if margin:
                difference = difference[..., margin:-margin, margin:-margin]
            terms["consistency"] = (difference ** 2).mean()
        return terms

    @property
    def is_fusion(self) -> bool:
        return self.config.objective.fusion is not None

    def _fusion_loss_terms(
        self, prediction: torch.Tensor, targets: torch.Tensor, shifts: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Fusion objective: the prediction (frame-0 coordinates) warped into
        the raw target frame's coordinates, compared on the interior."""
        objective = self.config.objective
        assert objective.fusion is not None
        margin = objective.fusion.warp_margin
        size = prediction.shape[-1]
        interior = np.s_[..., margin : size - margin, margin : size - margin]
        warped = warp_prediction(prediction, shifts)

        def distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            if objective.loss == "l1":
                return (a - b).abs().mean()
            return ((a - b) ** 2).mean()

        terms: dict[str, torch.Tensor] = {}
        if objective.lambda_image > 0.0:
            terms["image"] = distance(warped[interior], targets[interior])
        if objective.lambda_gradient > 0.0:
            terms["gradient"] = distance(sobel(warped)[interior], sobel(targets)[interior])
        return terms

    def _fusion_multi_terms(
        self, prediction: torch.Tensor, batch: FusionBatch
    ) -> dict[str, torch.Tensor]:
        """``target: multi_frame``: the main target plus every extra raw target,
        each through its own warp of the prediction, averaged per sample over
        the valid targets (masked mean, so a sample with fewer targets is not
        under-weighted)."""
        objective = self.config.objective
        assert objective.fusion is not None and batch.extra_targets is not None
        margin = objective.fusion.warp_margin
        size = prediction.shape[-1]
        interior = np.s_[..., margin : size - margin, margin : size - margin]
        targets = torch.cat([batch.targets.to(self.device)[:, None], batch.extra_targets.to(self.device)], dim=1)
        shifts = torch.cat([batch.shifts.to(self.device)[:, None], batch.extra_shifts.to(self.device)], dim=1)
        mask = torch.cat(
            [torch.ones_like(batch.extra_mask[:, :1]), batch.extra_mask], dim=1
        ).to(self.device)  # [B, T]
        count = targets.shape[1]
        flat_pred = prediction[:, None].expand(-1, count, -1, -1, -1).reshape(-1, *prediction.shape[1:])
        warped = warp_prediction(flat_pred, shifts.reshape(-1, 2))
        flat_targets = targets.reshape(-1, *prediction.shape[1:])
        weights = mask.reshape(-1)
        weight_sum = weights.sum().clamp_min(1.0)

        def distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            per = ((a - b).abs() if objective.loss == "l1" else (a - b) ** 2).mean(dim=(1, 2, 3))
            return (per * weights).sum() / weight_sum

        terms: dict[str, torch.Tensor] = {}
        if objective.lambda_image > 0.0:
            terms["image"] = distance(warped[interior], flat_targets[interior])
        if objective.lambda_gradient > 0.0:
            terms["gradient"] = distance(sobel(warped)[interior], sobel(flat_targets)[interior])
        return terms

    def _fusion_step(self, batch: FusionBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        inputs = batch.inputs.to(self.device)
        levels = batch.levels.to(self.device)
        with self._autocast():
            forward = self.train_model if self.model.training else self.model
            prediction = forward(inputs, t=levels)
        prediction = prediction.float()
        if batch.extra_targets is not None:
            return prediction, self._fusion_multi_terms(prediction, batch)
        targets = batch.targets.to(self.device)
        shifts = batch.shifts.to(self.device)
        return prediction, self._fusion_loss_terms(prediction, targets, shifts)

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

    def _validate_fusion(self, writer: SummaryWriter) -> None:
        """Validation at the lowest and highest configured dose levels: the
        loss against frame 0 of each val burst, PSNR of the fused output vs
        clean (frame 0 sits at the burst's origin, so no warp is needed), and
        a picture per level."""
        assert isinstance(self.factory, FusionFactory)
        count = self.config.training.val_images
        levels = sorted({min(self.factory.levels), max(self.factory.levels)})
        self.model.eval()
        with ema_parameters(self.model, self.ema), torch.no_grad():
            for level in levels:
                batch = self.factory.val_batch(count=count, level=level)
                prediction, terms = self._fusion_step(batch)
                writer.add_scalar(f"val/loss_m{level}", float(self._combine(terms).item()), self.step)
                denoised = self.model.predict_image(
                    batch.inputs.to(self.device), t=batch.levels.to(self.device)
                ).clamp(-1.0, 1.0)
                clean01 = ((batch.clean + 1.0) / 2.0).numpy()
                pred01 = ((denoised + 1.0) / 2.0).cpu().numpy()
                psnr_values = [
                    psnr(np.moveaxis(clean01[i], 0, -1), np.moveaxis(pred01[i], 0, -1))
                    for i in range(pred01.shape[0])
                ]
                writer.add_scalar(f"val/psnr_m{level}", float(np.mean(psnr_values)), self.step)
                grad_mse = ((sobel(denoised) - sobel(batch.clean.to(self.device))) ** 2).mean()
                writer.add_scalar(f"val/gradient_mse_vs_clean_m{level}", float(grad_mse.item()), self.step)
                shown = min(4, pred01.shape[0])
                input01 = ((batch.inputs.clamp(-1.0, 1.0) + 1.0) / 2.0).numpy()
                rows = [
                    np.concatenate([input01[i], pred01[i], clean01[i]], axis=-1)
                    for i in range(shown)
                ]
                writer.add_image(
                    f"val/input_pred_clean_m{level}", np.concatenate(rows, axis=-2), self.step
                )
            # The headline scalar keeps its historical name at the top level.
            writer.add_scalar("val/psnr", float(np.mean(psnr_values)), self.step)
        self.model.train()

    def _validate(self, writer: SummaryWriter) -> None:
        if not self.cache.val_sources:
            return
        if self.is_fusion:
            self._validate_fusion(writer)
            return
        count = self.config.training.val_images
        self.model.eval()
        with ema_parameters(self.model, self.ema), torch.no_grad():
            batch = self.factory.val_batch(count=count)
            inputs = batch.inputs.to(self.device)
            targets = batch.targets.to(self.device)
            second = batch.second.to(self.device)
            gradient_targets = (
                batch.gradient_targets.to(self.device)
                if batch.gradient_targets is not None
                else None
            )

            prediction = self.model(inputs)
            terms = self._loss_terms(
                prediction,
                targets,
                second if self.config.objective.lambda_consistency > 0.0 else None,
                gradient_targets,
                second_shifts=None if batch.second_shifts is None else batch.second_shifts.to(self.device),
                margin=batch.loss_margin,
            )
            writer.add_scalar("val/loss", float(self._combine(terms).item()), self.step)

            denoised = self.model.predict_image(inputs).clamp(-1.0, 1.0)
            denoised_second = self.model.predict_image(second).clamp(-1.0, 1.0)
            reference01 = (((batch.clean if batch.clean is not None else batch.targets) + 1.0) / 2.0).numpy()
            pred01 = ((denoised + 1.0) / 2.0).cpu().numpy()
            if batch.clean is not None:
                psnr_values = [psnr(np.moveaxis(reference01[i], 0, -1), np.moveaxis(pred01[i], 0, -1))
                               for i in range(pred01.shape[0])]
                writer.add_scalar("val/psnr", float(np.mean(psnr_values)), self.step)
                grad_mse = ((sobel(denoised) - sobel(batch.clean.to(self.device))) ** 2).mean()
                writer.add_scalar("val/gradient_mse_vs_clean", float(grad_mse.item()), self.step)

            # Live repeatability readout: RMS disagreement of two independent
            # frames' denoised images, in [0, 1] units (model range is 2x).
            if batch.second_shifts is not None:
                denoised_second = warp_prediction(denoised_second, batch.second_shifts.to(self.device))
            difference = denoised - denoised_second
            if batch.loss_margin:
                m = batch.loss_margin
                difference = difference[..., m:-m, m:-m]
            consistency = (difference ** 2).mean()
            writer.add_scalar(
                "val/consistency_sigma",
                float(torch.sqrt(consistency / 2.0).item()) / 2.0,
                self.step,
            )

            shown = min(4, pred01.shape[0])
            input01 = ((batch.inputs.clamp(-1.0, 1.0) + 1.0) / 2.0).numpy()
            rows = [
                np.concatenate([input01[i], pred01[i], reference01[i]], axis=-1)
                for i in range(shown)
            ]
            tag = "val/input_pred_clean" if batch.clean is not None else "val/input_pred_target"
            writer.add_image(tag, np.concatenate(rows, axis=-2), self.step)
        self.model.train()

    def run(self) -> Path:
        training = self.config.training
        writer = self.runtime.on_primary(lambda: SummaryWriter(log_dir=str(self.run_dir / TENSORBOARD_DIR_NAME)))
        window_losses: dict[str, list[float]] = {}
        window_started = time.time()
        window_steps = 0
        stop_reason: str | None = None
        failed = False
        handlers = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                previous = signal.signal(signum, lambda number, frame: self.stop_controller.request(f"signal {number}"))
                handlers[signum] = previous
            except ValueError:  # Library callers may train outside the main thread.
                pass
        if self.runtime.primary:
            logger.info("%d process(es), per-GPU batch %d, accumulation %d, effective batch %d, %s",
                        self.runtime.world_size, training.batch_size, training.accumulation_steps,
                        self.effective_batch, training.precision)
        self.model.train()
        try:
            while self.step < training.max_steps:
                if self.runtime.should_stop(self.stop_controller.is_requested()):
                    stop_reason = "coordinated stop request"
                    break
                self.optimizer.zero_grad(set_to_none=True)
                values: dict[str, float] = {}
                for microstep in range(training.accumulation_steps):
                    sync = (self.train_model.no_sync() if self.runtime.enabled and
                            microstep + 1 < training.accumulation_steps else nullcontext())
                    with sync:
                        batch = self.factory.sample_batch()
                        _, terms = self._fusion_step(batch) if isinstance(batch, FusionBatch) else self._pair_step(batch)
                        loss = self._combine(terms)
                        (loss / training.accumulation_steps).backward()
                    for name, value in {"total": loss, **terms}.items():
                        values[name] = values.get(name, 0.0) + float(value.detach().item()) / training.accumulation_steps
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), training.grad_clip)
                self.optimizer.step()
                if self.ema is not None:
                    self.ema.update(self.model)
                self.step += 1
                window_steps += 1

                for name, value in values.items():
                    window_losses.setdefault(name, []).append(value)

                if self.step == 1 or self.step % training.log_every == 0:
                    elapsed = max(time.time() - window_started, 1e-9)
                    means = self.runtime.mean_values({name: float(np.mean(items)) for name, items in window_losses.items()})
                    if writer is not None:
                        for name, value in means.items():
                            writer.add_scalar("train/loss" if name == "total" else f"train/loss_{name}", value, self.step)
                        writer.add_scalar("train/steps_per_sec", window_steps / elapsed, self.step)
                        writer.add_scalar("train/patches_per_sec", window_steps * self.effective_batch / elapsed, self.step)
                        writer.add_scalar("train/effective_batch", self.effective_batch, self.step)
                        logger.info("step %d | loss %.5f | %.2f steps/s", self.step, means["total"], window_steps / elapsed)
                    window_losses.clear()
                    window_steps = 0
                    window_started = time.time()

                if self.step % training.val_every == 0 or self.step == training.max_steps:
                    self.runtime.on_primary(lambda: self._validate(writer))
                if self.step % training.checkpoint_every == 0 or self.step == training.max_steps:
                    self._save(milestone=True)
        except KeyboardInterrupt:
            if self.runtime.enabled:
                failed = True
                raise
            stop_reason = "keyboard interrupt"
        except BaseException:
            failed = True
            raise
        finally:
            # A failed rank must exit promptly so torchrun can terminate its
            # peers, rather than entering a checkpoint collective they cannot join.
            try:
                if not failed:
                    self._save(milestone=False)
            finally:
                for signum, previous in handlers.items():
                    signal.signal(signum, previous)
                if writer is not None:
                    writer.close()
        if stop_reason is not None:
            logger.info("stopped early at step %d (%s)", self.step, stop_reason)
        else:
            logger.info("finished %d steps", self.step)
        return self.latest_checkpoint_path
