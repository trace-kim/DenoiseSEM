"""Standalone image loading for the segmentation pipeline.

This package imports nothing else in this repository, so the loaders live here
rather than being borrowed from ``edge_denoise`` or ``sem_noise``.  Two rules
are carried over deliberately because they protect measurements:

*No luminance conversion.*  An RGB file is accepted only when its three decoded
channels are identical, and then one channel is taken exactly.  Weighting an SEM
export by Rec.601 coefficients would silently rescale detector units.

*Crop before the grayscale check.*  Real instrument exports carry colored
overlays - a green frame, a databar, a scale bar - outside the imaging area.
Checking channel equality on the whole file rejects images whose imaging region
is perfectly grayscale, so the crop is applied first.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

#: Extensions the pipeline will attempt to read, matching the repository's
#: existing real-SEM contract.
IMAGE_SUFFIXES = frozenset({".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"})

_SIXTEEN_BIT_MODES = frozenset({"I;16", "I;16L", "I;16B", "I;16N", "I"})
_ACCEPTED_MODES = frozenset({"L", "F", "RGB", "RGBA"}) | _SIXTEEN_BIT_MODES

#: Smallest array the pipeline will work on; below this a normal-profile search
#: window spans the whole image.
MIN_SIDE_PX = 16


@dataclass(frozen=True)
class LoadedImage:
    """An image ready to measure, plus the provenance needed to reproduce it."""

    path: Path
    #: float64 [H, W] in [0, 1].  Never resized, stretched, or filtered.
    measure01: np.ndarray
    #: The decoded array as stored, after cropping and grayscale collapse.
    native: np.ndarray
    #: Value that maps to 1.0 in ``measure01`` (255, 65535, or a detector level).
    white_level: float
    black_level: float
    crop: tuple[int, int, int, int] | None
    file_sha256: str
    pixel_sha256: str

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.measure01.shape[0]), int(self.measure01.shape[1]))

    @property
    def stem(self) -> str:
        return self.path.stem


def file_hash(path: Path) -> str:
    """SHA-256 of the file bytes, for provenance."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_hash(array: np.ndarray) -> str:
    """Decoded-content identity including shape, independent of file encoding."""
    digest = hashlib.sha256(str(array.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
    return digest.hexdigest()


def _collapse_grayscale(array: np.ndarray, path: Path) -> np.ndarray:
    """Take the single channel of a repeated-grayscale array, or raise."""
    if array.ndim == 2:
        return array
    if array.ndim != 3:
        raise ValueError(f"{path.name}: expected a 2-D or 3-D array, got shape {array.shape}")
    if array.shape[2] == 4:
        if not np.all(array[..., 3] == array[..., 3].flat[0]):
            raise ValueError(f"{path.name}: RGBA with a varying alpha channel is unsupported")
        array = array[..., :3]
    if array.shape[2] == 1:
        return array[..., 0]
    if array.shape[2] != 3:
        raise ValueError(f"{path.name}: expected 1, 3, or 4 channels, got {array.shape[2]}")
    same = np.array_equal(array[..., 0], array[..., 1]) and np.array_equal(array[..., 0], array[..., 2])
    if not same:
        mismatch = int(np.count_nonzero(np.any(array != array[..., :1], axis=2)))
        raise ValueError(
            f"{path.name}: grayscale required; RGB channels differ at {mismatch} pixel(s) and no "
            "luminance conversion is performed. If the difference is a colored overlay or databar "
            "outside the imaging area, set input.crop (or --crop) to the imaging region; the crop "
            "is applied before this check."
        )
    # Repeated grayscale channels carry one signal. Select it exactly.
    return array[..., 0]


def _default_white_level(array: np.ndarray, mode: str) -> float:
    if array.dtype == np.uint8:
        return 255.0
    if array.dtype == np.uint16:
        return 65535.0
    if array.dtype.kind == "f":
        return 1.0
    if mode in _SIXTEEN_BIT_MODES:
        return 65535.0
    return float(np.iinfo(array.dtype).max)


def read_native(
    path: str | Path,
    *,
    crop: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, str]:
    """Decode one image to a 2-D array in its stored units.

    The crop is applied before the grayscale-consistency check so that a colored
    overlay outside the imaging area does not disqualify the file.
    """
    path = Path(path)
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"{path.name}: unsupported extension; expected one of {sorted(IMAGE_SUFFIXES)}")
    with Image.open(path) as image:
        mode = image.mode
        if mode not in _ACCEPTED_MODES:
            raise ValueError(
                f"{path.name}: unsupported image mode {mode!r}; expected one of {sorted(_ACCEPTED_MODES)}"
            )
        array = np.asarray(image).copy()

    if crop is not None:
        y0, y1, x0, x1 = crop
        if y1 > array.shape[0] or x1 > array.shape[1]:
            raise ValueError(
                f"{path.name}: crop {crop} extends outside the image of shape {array.shape[:2]}"
            )
        array = array[y0:y1, x0:x1]

    array = _collapse_grayscale(array, path)

    if array.dtype.kind not in "uif":
        raise ValueError(f"{path.name}: expected a real numeric image, got dtype {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path.name}: contains NaN or infinity")
    if min(array.shape) < MIN_SIDE_PX:
        raise ValueError(
            f"{path.name}: image/crop is {array.shape}; at least {MIN_SIDE_PX} px on each axis is required"
        )
    # PIL decodes 16-bit files as int32 mode 'I'. Narrow back when the values fit,
    # so the default white level is 65535 rather than 2**31.
    if mode == "I" and array.dtype != np.uint16:
        if array.min() >= 0 and array.max() <= 65535:
            array = array.astype(np.uint16)
    return np.ascontiguousarray(array), mode


def to_measure01(
    array: np.ndarray,
    *,
    black_level: float,
    white_level: float,
) -> np.ndarray:
    """Scale stored units into float64 [0, 1] for measurement."""
    if white_level <= black_level:
        raise ValueError(f"white_level must exceed black_level, got {white_level} <= {black_level}")
    scaled = (np.asarray(array, dtype=np.float64) - black_level) / (white_level - black_level)
    return np.clip(scaled, 0.0, 1.0)


def load_image(
    path: str | Path,
    *,
    crop: tuple[int, int, int, int] | None = None,
    black_level: float | None = None,
    white_level: float | None = None,
) -> LoadedImage:
    """Load one image into the measurement representation, with provenance."""
    path = Path(path)
    native, mode = read_native(path, crop=crop)
    white = float(white_level) if white_level is not None else _default_white_level(native, mode)
    black = float(black_level) if black_level is not None else 0.0
    measure01 = to_measure01(native, black_level=black, white_level=white)
    return LoadedImage(
        path=path,
        measure01=measure01,
        native=native,
        white_level=white,
        black_level=black,
        crop=crop,
        file_sha256=file_hash(path),
        pixel_sha256=pixel_hash(native),
    )


def to_model_rgb(
    measure01: np.ndarray,
    *,
    contrast_stretch: tuple[float, float] | None = (1.0, 99.0),
) -> np.ndarray:
    """Build the uint8 RGB array a segmentation model sees.

    This is the only place the image is allowed to be altered for the model's
    benefit.  The returned array is never measured; the caller keeps
    ``measure01`` for that.  SAM 3 was trained on natural-image contrast, so a
    low-contrast SEM frame often segments far better after a percentile stretch,
    and doing it here means that choice cannot reach a reported edge position.
    """
    image = np.asarray(measure01, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {image.shape}")
    if contrast_stretch is not None:
        low_pct, high_pct = contrast_stretch
        low, high = np.percentile(image, [low_pct, high_pct])
        if high > low:
            image = np.clip((image - low) / (high - low), 0.0, 1.0)
    gray = np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def discover_images(target: str | Path) -> list[Path]:
    """Return the images named by a file or contained in a folder.

    A folder is treated as a plain collection: the images are unrelated unless
    something outside this package says otherwise, so nothing here aggregates
    across them.
    """
    target = Path(target)
    if target.is_file():
        if target.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"{target.name}: unsupported extension; expected one of {sorted(IMAGE_SUFFIXES)}")
        return [target]
    if not target.is_dir():
        raise ValueError(f"input path does not exist: {target}")
    found = sorted(p for p in target.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not found:
        raise ValueError(f"no supported images found in {target} (looked for {sorted(IMAGE_SUFFIXES)})")
    return found
