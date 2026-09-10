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
    # Filled from a prepared real dataset and persisted in its checkpoint.
    black_level: float | None = Field(default=None, allow_inf_nan=False)
    white_level: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def _check_holdout_fractions(self) -> "DataConfig":
        if (self.black_level is None) != (self.white_level is None):
            raise ValueError("provide both data.black_level and data.white_level, or neither")
        if self.white_level is not None and self.white_level <= self.black_level:
            raise ValueError("data.white_level must exceed data.black_level")
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError(
                f"data.val_fraction + data.test_fraction must be < 1, got "
                f"{self.val_fraction} + {self.test_fraction}"
            )
        return self


class FusionConfig(_StrictModel):
    """Drift-robust burst fusion (see ``edge_denoise/fusion.py``).

    The network is trained on the mean of a registered subset of ``m`` raw
    frames of one burst against ONE other raw frame of the same burst, read
    in its own (drifted) coordinates: the prediction is warped by the
    registration's residual shift before the loss, the noisy target is never
    resampled or averaged.  ``t = m`` conditions the backbone on the dose, so
    the same network serves every frame count from 1 to ``frames_per_burst``.

    - ``frames_per_burst``: replicas per burst in the dataset (the generator's
      ``frames_per_burst``); training subsets never cross a burst.
    - ``levels``: the subset sizes ``m`` drawn uniformly at training time
      (each in ``1 .. frames_per_burst - 1``, one frame is the target).
    - ``align``: ``registered`` uses the table written by
      ``python -m edge_denoise register`` (``registration``); ``none`` averages
      the drifting frames as they are (the un-registered ablation); ``truth``
      uses the generator's recorded drift (the oracle-registration ablation).
    - ``condition_on_level``: feed ``t = m`` (``False`` = t = 1 everywhere, a
      dose-blind ablation).
    - ``warp_margin``: border (px) excluded from the loss after the warp.
    - ``registration_sigma`` / ``predenoised_sigma`` / ``registration_radius``:
      the registration's smoothing (raw / pre-denoised frames) and search
      radius, used at inference when a burst is registered on the fly.
    - ``debias_peak``: inference-time inverse of the clipped-Poisson response
      applied to the OUTPUT (the network estimates the clipped mean g(x);
      g is monotone, so g^-1 of the estimate is an estimate of x).
    - ``target``: ``frame`` = one raw frame outside the subset, read in its own
      coordinates through the warped prediction (the default; the target is
      never resampled).  ``complement_mean`` = the registered mean of EVERY
      frame outside the subset, in frame-0 coordinates like the input: still
      independent of the input's noise, so still an unbiased Noise2Noise
      target, but with ``(frames_per_burst - m)`` times less noise -- the
      gradient signal of the high-dose levels, where the estimation error is
      a hundred times below the single-frame target noise, is otherwise
      buried.  Legitimate exactly because the registration is measured.
    - ``level_cap``: inference conditioning uses ``min(K, level_cap)`` so a
      network trained up to a lower level still serves larger bursts.
    - ``target: clean`` is the synthetic-land oracle (the clean image in
      frame-0 coordinates): the ceiling any unbiased target can reach.
    - ``target: multi_frame``: ``targets_per_sample`` raw frames outside the
      subset, each compared with the prediction warped into ITS coordinates
      and averaged -- the frame target's unbiasedness with a fraction of its
      gradient noise (the complement mean resamples its frames and inherits
      the interpolation blur; this does not).
    """

    frames_per_burst: int = Field(default=16, ge=2)
    levels: list[int] = Field(default=[1, 2, 3, 4, 6, 8, 12, 15], min_length=1)
    target: Literal["frame", "complement_mean", "clean", "multi_frame"] = "frame"
    # ``multi_frame``: up to this many raw frames outside the subset are read
    # as targets at once (each through its own warp of the prediction) -- the
    # gradient noise of the frame target divided by the count, without ever
    # resampling a target.
    targets_per_sample: int = Field(default=4, ge=1)
    level_cap: int | None = Field(default=None, ge=1)
    align: Literal["registered", "none", "truth"] = "registered"
    registration: Path | None = None
    condition_on_level: bool = True
    warp_margin: int = Field(default=2, ge=0)
    registration_sigma: float = Field(default=2.0, ge=0.0)
    predenoised_sigma: float = Field(default=1.0, ge=0.0)
    registration_radius: int = Field(default=6, ge=1)
    debias_peak: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _check_levels(self) -> "FusionConfig":
        for level in self.levels:
            if not 1 <= level <= self.frames_per_burst - 1:
                raise ValueError(
                    f"fusion.levels entries must be in [1, frames_per_burst - 1] = "
                    f"[1, {self.frames_per_burst - 1}], got {level}"
                )
        if self.align == "registered" and self.registration is None:
            raise ValueError("fusion.align 'registered' requires fusion.registration (a table path)")
        if self.align != "registered" and self.registration is not None:
            raise ValueError("fusion.registration is only used with fusion.align 'registered'")
        return self


class ObjectiveConfig(_StrictModel):
    """What the network sees, what it predicts, and what the loss weighs.

    - ``image``:    input [y],            output x_hat (1 channel)
    - ``gradient``: input sobel(y),       output sobel(x)_hat (2 channels);
                    the image is recovered at inference by the exact FFT
                    least-squares inverse, with the DC offset taken from the
                    noisy input's crop mean.
    - ``hybrid``:   input [y, sobel(y)],  output x_hat (1 channel)

    ``target`` is what the fidelity terms chase: ``"noisy"`` is a fresh noisy
    replica (Noise2Noise), ``"clean"`` the clean image (supervised oracle), and
    ``"noisy_mean"`` the leave-one-out mean of every replica EXCEPT the input
    -- as unbiased as a fresh frame (the input's own noise never enters it) but
    with ~1/(N-1) of its variance, so the gradient signal for rare, low-contrast
    structure is ~(N-1)x cleaner per step.  It uses the burst only at training
    time; inference stays single-frame.

    ``lambda_image`` weighs the image-domain residual, ``lambda_gradient`` the
    Sobel-domain residual (for ``gradient`` the output already lives there, so
    ``lambda_image`` must be 0).  ``lambda_consistency`` adds the two-
    realization repeatability penalty ``|f(y_a) - f(y_b)|^2`` -- the direct
    precision lever; it needs a second independent frame per sample and trades
    bias for variance, so it defaults to off.

    ``gradient_target`` lets the GRADIENT term chase a different reference than
    the image term (the target-ladder experiment; the image term always keeps
    ``target``).  ``"target"`` is the historical behavior (same tensor for both
    terms); ``"clean"`` is the supervised oracle; ``"noisy_mean"`` is the
    leave-one-out average of every OTHER replica -- unbiased like a fresh frame
    but with ~1/(N-1) of its variance (the input replica must be excluded or
    the N2N cross-term argument breaks and the loss pulls toward identity);
    ``"file"`` reads a precomputed per-source image (float32 ``.npy`` in
    [0, 1]) from ``gradient_target_dir`` -- the distillation arm, produced by
    ``python -m edge_denoise distill-targets``.
    """

    representation: Literal["image", "gradient", "hybrid"] = "hybrid"
    target: Literal["clean", "noisy", "noisy_mean"] = "noisy"
    # With ``target: noisy_mean``: undo the clipping bias of the stored frames
    # (``min(Pois(peak * x), peak) / peak``) by pushing the leave-one-out mean
    # through the inverse of its expected response g(x) = E[min(K, peak)] / peak.
    # The mean of 15 clipped frames is precise enough for the 1-D inversion to
    # be well conditioned; the network then sees clipped inputs and unbiased
    # targets, like a clean-target oracle, without a clean image.  None = off.
    target_debias_peak: float | None = Field(default=None, gt=0.0)
    gradient_target: Literal["target", "clean", "noisy_mean", "file"] = "target"
    gradient_target_dir: Path | None = None
    lambda_image: float = Field(default=1.0, ge=0.0)
    lambda_gradient: float = Field(default=4.0, ge=0.0)
    lambda_consistency: float = Field(default=0.0, ge=0.0)
    loss: Literal["l2", "l1"] = "l2"
    # Burst fusion (registered subset mean in, one raw drifted frame as the
    # target, dose conditioning); None = the single-frame objectives above.
    fusion: FusionConfig | None = None

    @model_validator(mode="after")
    def _check_fusion(self) -> "ObjectiveConfig":
        if self.fusion is None:
            return self
        problems = []
        if self.representation == "gradient":
            problems.append("representation must be 'image' or 'hybrid'")
        if self.target != "noisy":
            problems.append("target must be 'noisy' (one raw frame of the burst)")
        if self.gradient_target != "target":
            problems.append("gradient_target must be 'target'")
        if self.lambda_consistency > 0.0:
            problems.append("lambda_consistency is not supported")
        if self.target_debias_peak is not None:
            problems.append("target_debias_peak is not supported (use fusion.debias_peak)")
        if problems:
            raise ValueError("objective.fusion: " + "; ".join(problems))
        return self

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
        if self.gradient_target != "target" and self.lambda_gradient <= 0.0:
            raise ValueError(
                "objective.gradient_target overrides the gradient term's reference, "
                "so lambda_gradient must be > 0 for it to have any effect"
            )
        if self.gradient_target == "file" and self.gradient_target_dir is None:
            raise ValueError(
                "objective.gradient_target 'file' requires objective.gradient_target_dir "
                "(a directory of per-source {source_index:05d}.npy targets)"
            )
        if self.gradient_target != "file" and self.gradient_target_dir is not None:
            raise ValueError(
                "objective.gradient_target_dir is only meaningful with "
                "objective.gradient_target 'file'"
            )
        if self.target_debias_peak is not None and self.target != "noisy_mean":
            raise ValueError(
                "objective.target_debias_peak applies to the leave-one-out mean target "
                "only (objective.target 'noisy_mean')"
            )
        return self


class ModelConfig(_StrictModel):
    """U-Net hyperparameters; identical fields and defaults to burst_diffusion
    so an equal-capacity comparison is a config diff, not a code diff."""

    ch: int = Field(default=64, ge=4)
    ch_mult: list[int] = Field(default=[1, 2, 2, 2], min_length=1)
    num_res_blocks: int = Field(default=2, ge=1)
    attention: bool = True  # includes the bottleneck attention block
    attn_resolutions: list[int] = Field(default=[16])
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    resamp_with_conv: bool = True
    num_groups: int | None = Field(default=None, ge=1)

    @property
    def effective_num_groups(self) -> int:
        return self.num_groups if self.num_groups is not None else min(32, self.ch)


class DefectAugmentConfig(_StrictModel):
    """Synthetic low-contrast defects added to the whole burst window of a
    training sample (input, target, agreement frame and gradient target all
    receive the SAME additive field), so the network learns that soft
    blemishes and scratches exist inside otherwise flat regions.

    Why: a conditional-mean estimator shows a weak feature at (contrast x
    posterior probability), and the posterior probability is driven by the
    learned prior -- a corpus of 76 scenes teaches "flats are flat", so rare
    real blemishes are treated as noise and erased.  Adding defects to every
    frame of the burst keeps the Noise2Noise argument intact (the field is
    independent of every frame's noise) and is exactly what one would do on
    real bursts.  The variance mismatch of not re-drawing Poisson noise for
    the added intensity is second order at these contrasts.

    Defects are Gaussian blobs (anisotropic sigma in ``blob_sigma`` px) or
    scratches (segments of length ``line_length`` with a ``line_sigma`` px
    Gaussian cross-profile), of either sign, with peak amplitude drawn from
    ``contrast`` (in [0, 1] intensity units).
    """

    probability: float = Field(default=0.5, ge=0.0, le=1.0)
    max_count: int = Field(default=3, ge=1)
    contrast: tuple[float, float] = (0.02, 0.08)
    blob_sigma: tuple[float, float] = (1.5, 6.0)
    line_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    line_length: tuple[float, float] = (8.0, 40.0)
    line_sigma: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def _check_ranges(self) -> "DefectAugmentConfig":
        for name in ("contrast", "blob_sigma", "line_length"):
            low, high = getattr(self, name)
            if not 0.0 < low <= high:
                raise ValueError(f"defect_augment.{name} must satisfy 0 < low <= high, got {low}, {high}")
        return self


class TrainingConfig(_StrictModel):
    run_dir: Path
    # Optional synthetic-defect augmentation of training samples (see
    # DefectAugmentConfig); absent = the historical data stream, bit for bit.
    defect_augment: DefectAugmentConfig | None = None
    # Weights-only warm start (the fine-tune protocol): model parameters are
    # initialized from this checkpoint's EMA weights (falling back to the live
    # weights) before step 0.  Accepts edge_denoise checkpoints and
    # burst_diffusion checkpoints with a matching backbone.  Optimizer, EMA,
    # RNG, and data-stream state all start fresh. Ignored when resuming a run.
    init_checkpoint: Path | None = None
    batch_size: int = Field(default=8, ge=1)
    accumulation_steps: int = Field(default=1, ge=1)
    precision: Literal["fp32", "bf16"] = "fp32"
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
        if self.objective.fusion is not None:
            return self.objective.fusion.frames_per_burst
        required = 1
        if self.objective.target == "noisy":
            required += 1
        if self.objective.lambda_consistency > 0.0:
            required += 1
        if "noisy_mean" in (self.objective.target, self.objective.gradient_target):
            # The leave-one-out average needs at least one replica besides the
            # input; it may share replicas with the consistency frame (each
            # term's zero-cross-term argument holds separately).
            required = max(required, 2)
        return required

    @model_validator(mode="after")
    def _check_structure(self) -> "Config":
        if self.objective.fusion is not None and self.training.defect_augment is not None:
            raise ValueError("training.defect_augment is not supported with objective.fusion")
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
        for resolution in self.model.attn_resolutions if self.model.attention else ():
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
