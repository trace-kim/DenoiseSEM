"""Typed configuration for edge_denoise: YAML -> validated pydantic models.

Follows burst_diffusion's config conventions exactly (``extra="forbid"``
everywhere, structural U-Net constraints checked up front) so a config typo
fails at load, not deep inside torch.  The new block is ``objective``: it
selects the input/output *representation* (image, gradient, or hybrid), the
training *target* (a fresh noisy frame -- Noise2Noise -- or the clean image),
and the loss weights.  ``representation: image`` with ``target: noisy`` and
``lambda_gradient: 0`` is textbook Noise2Noise, giving N2N a first-class
pipeline; every other combination is the gradient-domain experiment this
package exists for (see docs/edge_denoise_method.md for the math).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_StrictModel):
    """Burst-dataset access; fields mirror burst_diffusion so the identical
    content-group split (and therefore the identical held-out sources) falls
    out of the shared :class:`burst_diffusion.data.BurstCache`."""

    dataset_dir: Path
    image_size: int = Field(default=64, ge=8)
    # Gradient channels are defined for grayscale; SEM data is grayscale.
    channels: Literal[1] = 1
    val_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    test_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)
    split_seed: int = Field(default=2019, ge=0)

    @model_validator(mode="after")
    def _check_holdout_fractions(self) -> "DataConfig":
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError(
                f"data.val_fraction + data.test_fraction must be < 1, got "
                f"{self.val_fraction} + {self.test_fraction}"
            )
        return self


class ObjectiveConfig(_StrictModel):
    """What the network sees, what it predicts, and what the loss weighs.

    - ``image``:    input [y],            output x_hat (1 channel)
    - ``gradient``: input sobel(y),       output sobel(x)_hat (2 channels);
                    the image is recovered at inference by the exact FFT
                    least-squares inverse, with the DC offset taken from the
                    noisy input's crop mean.
    - ``hybrid``:   input [y, sobel(y)],  output x_hat (1 channel)

    ``lambda_image`` weighs the image-domain residual, ``lambda_gradient`` the
    Sobel-domain residual (for ``gradient`` the output already lives there, so
    ``lambda_image`` must be 0).  ``lambda_consistency`` adds the two-
    realization repeatability penalty ``|f(y_a) - f(y_b)|^2`` -- the direct
    precision lever; it needs a second independent frame per sample and trades
    bias for variance, so it defaults to off.
    """

    representation: Literal["image", "gradient", "hybrid"] = "hybrid"
    target: Literal["clean", "noisy"] = "noisy"
    lambda_image: float = Field(default=1.0, ge=0.0)
    lambda_gradient: float = Field(default=4.0, ge=0.0)
    lambda_consistency: float = Field(default=0.0, ge=0.0)
    loss: Literal["l2", "l1"] = "l2"

    @model_validator(mode="after")
    def _check_weights(self) -> "ObjectiveConfig":
        if self.representation == "gradient":
            if self.lambda_image != 0.0:
                raise ValueError(
                    "objective.lambda_image must be 0 for representation 'gradient': "
                    "the network output is a gradient field, so there is no "
                    "image-domain residual to weigh"
                )
            if self.lambda_gradient <= 0.0:
                raise ValueError(
                    "objective.lambda_gradient must be > 0 for representation 'gradient'"
                )
        if self.lambda_image <= 0.0 and self.lambda_gradient <= 0.0:
            raise ValueError(
                "at least one of objective.lambda_image / lambda_gradient must be > 0 "
                "(lambda_consistency alone is minimized by any constant output)"
            )
        return self


class ModelConfig(_StrictModel):
    """U-Net hyperparameters; identical fields and defaults to burst_diffusion
    so an equal-capacity comparison is a config diff, not a code diff."""

    ch: int = Field(default=64, ge=4)
    ch_mult: list[int] = Field(default=[1, 2, 2, 2], min_length=1)
    num_res_blocks: int = Field(default=2, ge=1)
    attn_resolutions: list[int] = Field(default=[16])
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    resamp_with_conv: bool = True
    num_groups: int | None = Field(default=None, ge=1)

    @property
    def effective_num_groups(self) -> int:
        return self.num_groups if self.num_groups is not None else min(32, self.ch)


class TrainingConfig(_StrictModel):
    run_dir: Path
    batch_size: int = Field(default=8, ge=1)
    max_steps: int = Field(default=30000, ge=1)
    lr: float = Field(default=2.0e-4, gt=0.0)
    beta1: float = Field(default=0.9, ge=0.0, lt=1.0)
    adam_eps: float = Field(default=1.0e-8, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    grad_clip: float = Field(default=1.0, gt=0.0)
    ema: bool = True
    ema_rate: float = Field(default=0.999, ge=0.0, lt=1.0)
    seed: int = Field(default=0, ge=0)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    log_every: int = Field(default=50, ge=1)
    val_every: int = Field(default=1000, ge=1)
    val_images: int = Field(default=8, ge=1)
    checkpoint_every: int = Field(default=2000, ge=1)


class Config(_StrictModel):
    data: DataConfig
    objective: ObjectiveConfig
    model: ModelConfig
    training: TrainingConfig

    @property
    def in_channels(self) -> int:
        return {"image": 1, "gradient": 2, "hybrid": 3}[self.objective.representation]

    @property
    def out_channels(self) -> int:
        return 2 if self.objective.representation == "gradient" else 1

    @property
    def min_replicas(self) -> int:
        """Independent frames every source must supply for this objective.

        A noisy target must come from a frame other than the input, and the
        consistency pair from a third frame distinct from both (a shared frame
        would correlate the penalty with the input or target noise).
        """
        required = 1
        if self.objective.target == "noisy":
            required += 1
        if self.objective.lambda_consistency > 0.0:
            required += 1
        return required

    @model_validator(mode="after")
    def _check_structure(self) -> "Config":
        image_size = self.data.image_size
        num_levels = len(self.model.ch_mult)
        divisor = 2 ** (num_levels - 1)
        if image_size % divisor != 0:
            raise ValueError(
                f"data.image_size ({image_size}) must be divisible by "
                f"2**(len(model.ch_mult)-1) = {divisor} so every "
                "downsample/upsample level lines up"
            )
        for mult in self.model.ch_mult:
            if mult < 1:
                raise ValueError(f"model.ch_mult entries must be >= 1, got {mult}")
        groups = self.model.effective_num_groups
        if self.model.ch % groups != 0:
            raise ValueError(
                f"model.num_groups ({groups}) must divide model.ch ({self.model.ch})"
            )
        level_resolutions = {image_size >> level for level in range(num_levels)}
        for resolution in self.model.attn_resolutions:
            if resolution not in level_resolutions:
                raise ValueError(
                    f"model.attn_resolutions entry {resolution} is not one of the "
                    f"reachable level resolutions {sorted(level_resolutions, reverse=True)}"
                )
            if resolution > 16:
                warnings.warn(
                    f"attention at resolution {resolution} costs O((H*W)^2) memory; "
                    "resolutions above 16 are rarely affordable",
                    stacklevel=2,
                )
        return self


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping, got {type(raw).__name__}: {config_path}")
    return Config.model_validate(raw)
