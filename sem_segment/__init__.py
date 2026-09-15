"""SEM feature segmentation, contour extraction, and metrology.

Dependency policy: this package imports nothing else in this repository.  It is
an analysis-tier package like ``sem_noise`` - it needs no GPU for its core
pipeline and depends on the ``[analysis]`` extra (scipy, scikit-image,
matplotlib).  SAM 3 support additionally needs the ``[segment]`` extra, and the
``classical`` backend runs without either it or any model weights.

The pipeline has two halves, and keeping them distinct is the point:

*Segmentation* decides which pixels belong to which feature, and which side is
inside.  That is what SAM 3 contributes and what classical thresholding cannot.

*Refinement* decides where the boundary actually is, by fitting an edge model to
the original, unmodified pixels along each contour normal.  A mask boundary
interpolates the 0.5 level of a binary mask, so its precision is bounded by the
resolution the mask was produced at - roughly two original pixels for SAM 3 on a
frame larger than its native 1008 px.  On synthetic edges the refinement is
unbiased to better than 0.001 px.

Typical use::

    from sem_segment import load_config, load_image, segment_image

    config = load_config("sem_segment/configs/default.yml")
    image = load_image("frame.tif", crop=(2, 441, 2, 511))
    result = segment_image(image.measure01, config)
    rows = result.rows(pixel_size_nm=5.82812)

``segment_image`` is pure: it touches no filesystem and downloads nothing
implicitly.  Writing and rendering are separate, so a wrapper that aggregates
across images composes on the same API without this package having to guess what
a folder of images means.
"""

from __future__ import annotations

from .backends import BACKENDS, BackendUnavailable, InstanceMask, Segmenter, build_segmenter
from .config import (
    Config,
    ContoursConfig,
    InputConfig,
    MasksConfig,
    MetrologyConfig,
    RefineConfig,
    ReportConfig,
    SegmentationConfig,
    load_config,
)
from .contours import Contour, trace_all, trace_instance
from .image_io import LoadedImage, discover_images, load_image, to_model_rgb
from .masks import postprocess
from .metrology import RegionMetrology, ShapeMeasures, measure_shape, summarise_image
from .pipeline import Diagnostics, SegmentationResult, segment_image
from .refine import RefinedContour, refine_all, refine_contour
from .writers import write_result

__version__ = "0.1.0"

__all__ = [
    "BACKENDS",
    "BackendUnavailable",
    "Config",
    "Contour",
    "ContoursConfig",
    "Diagnostics",
    "InputConfig",
    "InstanceMask",
    "LoadedImage",
    "MasksConfig",
    "MetrologyConfig",
    "RefineConfig",
    "RefinedContour",
    "RegionMetrology",
    "ReportConfig",
    "SegmentationConfig",
    "SegmentationResult",
    "Segmenter",
    "ShapeMeasures",
    "build_segmenter",
    "discover_images",
    "load_config",
    "load_image",
    "measure_shape",
    "postprocess",
    "refine_all",
    "refine_contour",
    "segment_image",
    "summarise_image",
    "to_model_rgb",
    "trace_all",
    "trace_instance",
    "write_result",
]
