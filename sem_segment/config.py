"""Configuration schema for the SEM segmentation pipeline.

Every section forbids unknown keys, so a misspelled setting fails at load time
rather than being silently ignored.  The schema is split so the two halves of
the pipeline stay separable: ``segmentation`` decides *which pixels belong to
which feature*, while ``contours`` / ``refine`` / ``metrology`` decide *where
the boundary actually is* and *what to measure*.

Units are pixels everywhere inside the package.  ``input.pixel_size_nm`` is an
operator-supplied conversion applied only when results are reported; nothing
here reads a physical scale out of an image file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    """Reject unknown keys so misspelled settings fail loudly."""

    model_config = ConfigDict(extra="forbid")


class InputConfig(_StrictModel):
    """How a file on disk becomes the array the pipeline measures.

    ``crop`` is applied *before* the grayscale-consistency check.  That order
    matters for real instrument exports: SEM TIFFs routinely carry a colored
    frame or a databar, which would otherwise make an otherwise-grayscale file
    look like a color image and be rejected outright.
    """

    pixel_size_nm: float | None = Field(default=None, gt=0.0)
    crop: tuple[int, int, int, int] | None = Field(
        default=None, description="(y0, y1, x0, x1) half-open slice applied on load."
    )
    black_level: float | None = Field(default=None, ge=0.0)
    white_level: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _check_input(self) -> "InputConfig":
        if self.crop is not None:
            y0, y1, x0, x1 = self.crop
            if y0 < 0 or x0 < 0:
                raise ValueError(f"crop origin must be non-negative, got {self.crop}")
            if y1 - y0 < 16 or x1 - x0 < 16:
                raise ValueError(f"crop must keep at least 16 px on each axis, got {self.crop}")
        if (self.black_level is None) != (self.white_level is None):
            raise ValueError("black_level and white_level must be set together")
        if self.black_level is not None and self.white_level is not None:
            if self.white_level <= self.black_level:
                raise ValueError(
                    f"white_level must exceed black_level, got {self.white_level} <= {self.black_level}"
                )
        return self


#: SAM 3's vision tower is fixed at this resolution (patch 14).  The upstream
#: documentation warns that other resolutions degrade accuracy, so the package
#: never tiles at a "rounder" size such as 1024.
SAM3_NATIVE_PX = 1008


class LargeFrameConfig(_StrictModel):
    """What to do when a frame is larger than the model's native resolution.

    ``resize`` is the default, and deliberately so.  Refinement only needs the
    mask boundary to land within a few pixels of the true edge, which survives a
    2x downsample easily, and the refinement step then recovers full-resolution
    precision from the original pixels.  That costs one forward pass and has no
    seams at all.

    ``tile`` keeps the model at native scale but introduces seams, needs an
    overlap wider than the largest feature, and can drop a feature entirely if
    that constraint is violated.  It is worth it only when features would fall
    below roughly ten pixels across in the resized frame.
    """

    mode: Literal["resize", "tile", "none"] = "resize"
    tile_px: int = Field(default=SAM3_NATIVE_PX, ge=64)
    overlap_px: int = Field(default=256, ge=32)
    #: Warn when resizing would shrink a typical feature below this many pixels.
    min_feature_px_after_resize: float = Field(default=10.0, gt=0.0)

    @model_validator(mode="after")
    def _check_large_frame(self) -> "LargeFrameConfig":
        if self.overlap_px >= self.tile_px:
            raise ValueError(
                f"overlap_px must be smaller than tile_px, got {self.overlap_px} >= {self.tile_px}"
            )
        return self


class SegmentationConfig(_StrictModel):
    """Backend selection and prompting.

    ``contrast_stretch`` affects only the array handed to the model.  The
    measurement array is never stretched; see ``sem_segment.pipeline``.
    """

    backend: Literal["sam3_auto", "sam3_text", "sam3_prompt", "classical"] = "sam3_auto"
    model_id: str = "facebook/sam3"
    model_path: Path | None = None
    device: str = "auto"
    contrast_stretch: tuple[float, float] | None = (1.0, 99.0)
    text: str | None = None
    points: list[tuple[float, float]] | None = None
    boxes: list[tuple[float, float, float, float]] | None = None
    points_per_batch: int = Field(default=64, ge=1)
    points_per_crop: int = Field(default=32, ge=4)
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    mask_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_instances: int = Field(default=256, ge=1)
    #: ``classical`` only: split touching regions with a distance-transform watershed.
    split_touching: bool = True
    #: ``classical`` only: which side of the threshold is the feature.
    polarity: Literal["auto", "bright", "dark"] = "auto"
    large_frame: LargeFrameConfig = Field(default_factory=LargeFrameConfig)

    @model_validator(mode="after")
    def _check_segmentation(self) -> "SegmentationConfig":
        if self.backend == "sam3_text" and not self.text:
            raise ValueError("backend 'sam3_text' requires a non-empty 'text' prompt")
        if self.backend == "sam3_prompt" and not (self.points or self.boxes):
            raise ValueError("backend 'sam3_prompt' requires 'points' or 'boxes'")
        if self.contrast_stretch is not None:
            low, high = self.contrast_stretch
            if not 0.0 <= low < high <= 100.0:
                raise ValueError(
                    "contrast_stretch must be ascending percentiles within [0, 100], "
                    f"got {self.contrast_stretch}"
                )
        for box in self.boxes or ():
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"box must be (x1, y1, x2, y2) with x2 > x1 and y2 > y1, got {box}")
        return self


class MasksConfig(_StrictModel):
    """Which raw instance masks survive to become measurable regions.

    Two policies here are deliberately not the common shortcut.

    *Containment is checked separately from IoU.*  A mask nested inside another
    has a low IoU with it, so IoU-only deduplication keeps both and every
    feature is measured twice.  ``sam3_auto`` produces exactly this "the hole
    and the hole's dark core" pattern routinely.  ``containment_keep="larger"``
    then keeps the outer boundary rather than the higher-scoring one, because
    the tight inner core often scores higher and is not the feature.

    *Holes are filled only when small.*  Always-filling silently turns a genuine
    annulus - a ring, a via seen through a dielectric - into a disk.  Small
    speckle holes are filled; larger ones are kept, counted, and get their own
    refined boundary with the inside/outside sense inverted.
    """

    min_area_px: float = Field(default=25.0, gt=0.0)
    max_area_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    iou_dedupe: float = Field(default=0.7, gt=0.0, le=1.0)
    containment_dedupe: float = Field(default=0.9, gt=0.0, le=1.0)
    containment_keep: Literal["larger", "higher_score"] = "larger"
    max_fill_area_fraction: float = Field(default=0.1, ge=0.0, le=1.0)
    #: Border regions are always measured; this controls image-level aggregates only.
    include_border_regions: bool = False
    keep_largest_component: bool = True


class ContoursConfig(_StrictModel):
    """Method 1: the polygon traced around a mask.

    ``normal_smooth_px`` controls how much the vertex sequence is smoothed
    before differentiating to obtain a tangent.  A marching-squares polygon from
    a *binary* mask is a staircase, so an unsmoothed tangent oscillates between
    axis-aligned directions and the resulting normals are useless for profile
    sampling.  Too much smoothing biases a genuinely curved boundary inward.
    """

    spacing_px: float = Field(default=1.0, gt=0.0)
    normal_smooth_px: float = Field(default=2.0, ge=0.0)
    min_vertices: int = Field(default=12, ge=4)


class RefineConfig(_StrictModel):
    """Method 2: where the edge really is, measured on the unmodified image.

    ``threshold`` is the default because it is the same 50%-of-p10/p90
    convention the repository's existing CD harness uses, which keeps
    contour-derived numbers comparable with previously published measurements.

    ``interp_order`` defaults to cubic on purpose: bilinear sampling injects a
    once-per-pixel systematic of order 0.05-0.1 px, which is the entire
    precision budget this package exists to protect.
    """

    enabled: bool = True
    estimator: Literal["threshold", "gradient_peak", "erf"] = "threshold"
    search_px: float = Field(default=6.0, gt=1.0)
    step_px: float = Field(default=0.25, gt=0.0)
    interp_order: Literal[1, 3] = 3
    #: Gaussian smoothing applied along the contour before sampling normals.
    sigma_normal_px: float = Field(default=2.0, gt=0.0)
    #: Gaussian derivative width used by the ``gradient_peak`` estimator.
    deriv_sigma_px: float = Field(default=0.5, gt=0.0)
    min_contrast: float = Field(default=0.05, gt=0.0)
    max_residual: float = Field(default=0.15, gt=0.0)
    #: Accept a shift only below this; defaults to the full search radius.
    max_shift_px: float | None = Field(default=None, gt=0.0)
    allow_multiple_crossings: bool = False
    #: Rejected runs shorter than this are interpolated across; longer ones split the contour.
    max_gap_px: float = Field(default=5.0, ge=0.0)
    min_valid_fraction: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_refine(self) -> "RefineConfig":
        if self.max_shift_px is not None and self.max_shift_px > self.search_px:
            raise ValueError(
                f"max_shift_px={self.max_shift_px} cannot exceed search_px={self.search_px}"
            )
        samples = int(round(2.0 * self.search_px / self.step_px)) + 1
        if samples < 9:
            raise ValueError(
                f"search window yields only {samples} samples; need at least 9 to fit an edge "
                "(reduce step_px or raise search_px)"
            )
        return self


class MetrologyConfig(_StrictModel):
    """What gets measured on each closed contour.

    ``cd_definition`` defaults to the equivalent circular diameter rather than
    the minimum Feret caliper.  A minimum over many caliper angles is an
    extreme-value statistic: its distribution is skewed, biased low, and has a
    larger variance than the underlying shape warrants, which makes it a poor
    number to put in a repeatability table.

    Line-edge roughness is computed from the *refined* contour's residual about
    its own smooth trend - not from the refinement displacement.  The
    displacement measures how wrong the segmentation boundary was, which is a
    property of the model, not of the specimen.
    """

    chord_angles: int = Field(default=36, ge=4)
    cd_definition: Literal["equivalent_diameter", "feret_min", "feret_mean", "chord_mean"] = (
        "equivalent_diameter"
    )
    roughness_detrend: Literal["smooth", "none"] = "smooth"
    ler_highpass_cutoff_px: float = Field(default=8.0, gt=0.0)
    nearest_neighbours: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _check_metrology(self) -> "MetrologyConfig":
        return self


class ReportConfig(_StrictModel):
    """Figure rendering for the self-contained HTML report."""

    enabled: bool = True
    zoom_insets: int = Field(default=3, ge=0)
    zoom_half_width_px: int = Field(default=48, ge=8)
    max_table_rows: int = Field(default=200, ge=1)
    profile_samples: int = Field(default=6, ge=0)
    dpi: int = Field(default=130, ge=50, le=400)


class Config(_StrictModel):
    """Root configuration."""

    input: InputConfig = Field(default_factory=InputConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    masks: MasksConfig = Field(default_factory=MasksConfig)
    contours: ContoursConfig = Field(default_factory=ContoursConfig)
    refine: RefineConfig = Field(default_factory=RefineConfig)
    metrology: MetrologyConfig = Field(default_factory=MetrologyConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    @property
    def needs_model_weights(self) -> bool:
        """True when the selected backend loads a pretrained checkpoint."""
        return self.segmentation.backend.startswith("sam3")

    @model_validator(mode="after")
    def _check_structure(self) -> "Config":
        # A region must be wide enough for a normal search to find plateaus on
        # both sides of its edge, or every profile spans the whole feature.
        span = 2.0 * self.refine.search_px
        if self.refine.enabled and self.masks.min_area_px < 0.17 * span * span:
            raise ValueError(
                f"min_area_px={self.masks.min_area_px} is too small for search_px="
                f"{self.refine.search_px}: a feature that small is narrower than the profile "
                "window, so no edge can be fitted. Raise min_area_px or lower search_px."
            )
        # The detrend must remove something coarser than the contour smoothing
        # already did, or the high-pass band is empty and LER measures nothing.
        if self.metrology.roughness_detrend == "smooth":
            if self.metrology.ler_highpass_cutoff_px <= 2.0 * self.refine.sigma_normal_px:
                raise ValueError(
                    f"ler_highpass_cutoff_px={self.metrology.ler_highpass_cutoff_px} must exceed "
                    f"2 * sigma_normal_px={2.0 * self.refine.sigma_normal_px}; otherwise the "
                    "roughness band has already been smoothed away and LER would read as noise."
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
