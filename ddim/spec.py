"""The DDIM flow's training contract.

``runctl`` owns the flow-agnostic half of these settings via
:class:`runctl.schemas.BaseTrainingSpec`.  Everything below is specific to the
Song/Meng/Ermon DDIM pipeline: the U-Net geometry, the beta schedule, and the
data transforms the legacy ``datasets``/``models`` code expects.

The strictness of the original design is deliberate and preserved: unknown
fields are rejected, values are not implicitly coerced, and cross-field
relationships (attention resolutions must exist in the model, GroupNorm needs
channel counts divisible by 32) are validated up front rather than crashing
several minutes into a run.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from runctl.schemas import BaseTrainingSpec


class DdimTrainingSpec(BaseTrainingSpec):
    """Fully resolved SEM DDIM training settings."""

    label: str = Field(default="sem-ddim", pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
    dataset_alias: str = Field(default="sem", pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")

    image_size: int = Field(default=32, ge=8, le=8192)
    channels: Literal[1] = 1
    logit_transform: bool = False
    uniform_dequantization: bool = False
    gaussian_dequantization: bool = False
    random_flip: bool = False
    rescaled: bool = True

    model_type: Literal["simple"] = "simple"
    in_channels: Literal[1] = 1
    out_channels: Literal[1] = 1
    model_ch: int = Field(default=64, ge=1)
    ch_mult: tuple[int, ...] = (1, 2, 2, 2)
    num_res_blocks: int = Field(default=2, ge=1)
    attn_resolutions: tuple[int, ...] = (16,)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    var_type: Literal["fixedlarge", "fixedsmall"] = "fixedlarge"
    ema_rate: float = Field(default=0.999, gt=0.0, lt=1.0)
    ema: bool = True
    resamp_with_conv: bool = True

    beta_schedule: Literal["quad", "linear", "const", "jsd", "sigmoid"] = "linear"
    beta_start: float = Field(default=0.001, gt=0.0, lt=1.0)
    beta_end: float = Field(default=0.2, gt=0.0, lt=1.0)
    diffusion_steps: int = Field(default=100, ge=2)

    batch_size: int = Field(default=7, ge=1)
    sampling_batch_size: int = Field(default=8, ge=1)
    sampling_last_only: bool = True

    weight_decay: float = Field(default=0.0, ge=0.0)
    optimizer: Literal["Adam"] = "Adam"
    beta1: float = Field(default=0.9, ge=0.0, lt=1.0)
    amsgrad: bool = False
    eps: float = Field(default=1e-8, gt=0.0)
    grad_clip: float = Field(default=1.0, gt=0.0)

    @field_validator("ch_mult")
    @classmethod
    def validate_ch_mult(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(item < 1 for item in value):
            raise ValueError("ch_mult must contain positive integers")
        return value

    @field_validator("attn_resolutions")
    @classmethod
    def validate_attention_values(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(value)) != len(value) or any(item < 1 for item in value):
            raise ValueError("attn_resolutions must contain unique positive integers")
        return value

    @model_validator(mode="after")
    def validate_training_relationships(self) -> "DdimTrainingSpec":
        if self.beta_start >= self.beta_end:
            raise ValueError("beta_start must be less than beta_end")
        if self.logit_transform and self.rescaled:
            raise ValueError("logit_transform and rescaled are mutually exclusive")
        if self.uniform_dequantization and self.gaussian_dequantization:
            raise ValueError("choose at most one dequantization method")
        if self.cache_in_memory and self.random_flip:
            raise ValueError("cache_in_memory is incompatible with random_flip")

        downsample_factor = 2 ** (len(self.ch_mult) - 1)
        if self.image_size % downsample_factor:
            raise ValueError(
                f"image_size must be divisible by the model downsampling factor {downsample_factor}"
            )
        possible_resolutions = {
            self.image_size // (2**level) for level in range(len(self.ch_mult))
        }
        invalid_attention = set(self.attn_resolutions) - possible_resolutions
        if invalid_attention:
            expected = ", ".join(str(item) for item in sorted(possible_resolutions))
            invalid = ", ".join(str(item) for item in sorted(invalid_attention))
            raise ValueError(
                f"attention resolutions {invalid} are not model resolutions; choose from {expected}"
            )
        invalid_group_norm_channels = [
            self.model_ch * multiplier
            for multiplier in self.ch_mult
            if self.model_ch * multiplier < 32 or (self.model_ch * multiplier) % 32
        ]
        if self.model_ch < 32 or self.model_ch % 32 or invalid_group_norm_channels:
            raise ValueError(
                "model_ch and every model_ch * ch_mult value must be at least 32 "
                "and divisible by GroupNorm's 32 groups"
            )
        return self

    def estimate_checkpoint_bytes(self, *, bytes_per_parameter: int = 16) -> int:
        """Conservative UNet checkpoint estimate for launch previews.

        It is intentionally approximate: model + gradient-free EMA + Adam
        moments commonly consume about sixteen bytes per FP32 parameter on
        disk/in memory.
        """

        channel_units = sum(multiplier * self.model_ch for multiplier in self.ch_mult)
        rough_parameters = max(1, self.num_res_blocks) * 18 * channel_units**2
        return math.ceil(rough_parameters * bytes_per_parameter)

    def to_legacy_dict(self, dataset_path: Path | str) -> dict[str, Any]:
        """Render the nested shape expected by the original model/dataset code."""

        return {
            "data": {
                "dataset": "SEM",
                "data_path": str(dataset_path),
                "data_dir": str(dataset_path),
                "image_size": self.image_size,
                "channels": self.channels,
                "logit_transform": self.logit_transform,
                "uniform_dequantization": self.uniform_dequantization,
                "gaussian_dequantization": self.gaussian_dequantization,
                "random_flip": self.random_flip,
                "rescaled": self.rescaled,
                "num_workers": self.num_workers,
                "cache_in_memory": self.cache_in_memory,
                "recursive": self.recursive,
                "validation_split": self.validation_split,
                "split_seed": self.split_seed,
                "extensions": list(self.extensions),
            },
            "model": {
                "type": self.model_type,
                "in_channels": self.in_channels,
                "out_ch": self.out_channels,
                "ch": self.model_ch,
                "ch_mult": list(self.ch_mult),
                "num_res_blocks": self.num_res_blocks,
                "attn_resolutions": list(self.attn_resolutions),
                "dropout": self.dropout,
                "var_type": self.var_type,
                "ema_rate": self.ema_rate,
                "ema": self.ema,
                "resamp_with_conv": self.resamp_with_conv,
            },
            "diffusion": {
                "beta_schedule": self.beta_schedule,
                "beta_start": self.beta_start,
                "beta_end": self.beta_end,
                "num_diffusion_timesteps": self.diffusion_steps,
            },
            "training": {
                "batch_size": self.batch_size,
                "max_steps": self.max_steps,
                "n_iters": self.max_steps,
                "snapshot_freq": self.checkpoint_every,
                "validation_freq": self.validation_every,
                "sample_freq": self.sample_every,
                "checkpoint_minutes": self.checkpoint_minutes,
            },
            "sampling": {
                "batch_size": self.sampling_batch_size,
                "last_only": self.sampling_last_only,
            },
            "optim": {
                "weight_decay": self.weight_decay,
                "optimizer": self.optimizer,
                "lr": self.lr,
                "beta1": self.beta1,
                "amsgrad": self.amsgrad,
                "eps": self.eps,
                "grad_clip": self.grad_clip,
            },
        }


def estimate_checkpoint_bytes(
    spec: DdimTrainingSpec, *, bytes_per_parameter: int = 16
) -> int:
    """Module-level shim kept for callers of the pre-split API."""

    return spec.estimate_checkpoint_bytes(bytes_per_parameter=bytes_per_parameter)
