"""Discover ordered sites and read grayscale pixels without normalization."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image
import tifffile


SUFFIXES = {".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg", ".npy"}


@dataclass
class Frame:
    site: str
    path: Path
    relative_path: str
    index: int
    page: int = 0
    timestamp_s: float | None = None
    include: bool = True
    metadata: dict[str, str] = field(default_factory=dict)


def natural_key(value: str) -> tuple:
    """Numeric filename ordering, with a deterministic tie break."""
    return tuple((1, int(p)) if p.isdigit() else (0, p.casefold())
                 for p in re.split(r"(\d+)", value)) + ((0, value),)


def _page_count(path: Path) -> int:
    if path.suffix.lower() in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as image:
            return len(image.pages)
    if path.suffix.lower() == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            if array.ndim not in {2, 3}:
                raise ValueError(f"{path.name}: NPY must be HxW or TxHxW")
            return 1 if array.ndim == 2 else len(array)
        finally:
            array._mmap.close()
    return 1


def discover_sites(root: str | Path, manifest: str | Path | None = None) -> dict[str, list[Frame]]:
    """Use explicit CSV order, or one directory/stack per site and natural order.

    CSV requires site,path,frame_index. Optional page,timestamp_s,include and
    arbitrary acquisition-setting columns are preserved in provenance.
    """
    root = Path(root).resolve()
    if not root.exists():
        raise ValueError(f"input does not exist: {root}")
    sites: dict[str, list[Frame]] = {}
    if manifest is not None:
        if not root.is_dir():
            raise ValueError("--input must be a directory when using --manifest")
        with Path(manifest).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"site", "path", "frame_index"} <= set(reader.fieldnames or []):
                raise ValueError("manifest requires site,path,frame_index columns")
            for row_number, row in enumerate(reader, 2):
                if None in row or any(v is None for v in row.values()):
                    raise ValueError(f"malformed manifest row {row_number}")
                site = row["site"].strip()
                path = (root / row["path"]).resolve()
                if not site or not path.is_relative_to(root) or not path.is_file():
                    raise ValueError(f"invalid site or input path at manifest row {row_number}")
                index, page = int(row["frame_index"]), int(row.get("page") or 0)
                if index < 0 or page < 0 or path.suffix.lower() not in SUFFIXES:
                    raise ValueError(f"invalid index, page, or format at row {row_number}")
                stamp = float(row["timestamp_s"]) if row.get("timestamp_s") else None
                if stamp is not None and not math.isfinite(stamp):
                    raise ValueError("timestamps must be finite seconds")
                included = row.get("include", "true").strip().lower() or "true"
                if included not in {"true", "false", "1", "0"}:
                    raise ValueError("include must be true/false or 1/0")
                known = {"site", "path", "frame_index", "page", "timestamp_s", "include"}
                sites.setdefault(site, []).append(Frame(
                    site, path, path.relative_to(root).as_posix(), index, page,
                    stamp, included in {"true", "1"},
                    {k: v for k, v in row.items() if k not in known},
                ))
        for site, frames in sites.items():
            frames.sort(key=lambda f: f.index)
            if len({f.index for f in frames}) != len(frames):
                raise ValueError(f"{site}: duplicate frame_index")
            if len({(f.path, f.page) for f in frames}) != len(frames):
                raise ValueError(f"{site}: repeated path/page in manifest")
            stamps = [f.timestamp_s for f in frames]
            if any(s is not None for s in stamps):
                if any(s is None for s in stamps) or any(b <= a for a, b in zip(stamps, stamps[1:])):
                    raise ValueError(f"{site}: timestamps must all be present and strictly increasing")
    else:
        if root.is_file():
            groups = [(root.stem, [root])]
        else:
            direct = sorted((p for p in root.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES),
                            key=lambda p: natural_key(p.name))
            groups = []
            for directory in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: natural_key(p.name)):
                paths = sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES),
                               key=lambda p: natural_key(p.name))
                if paths:
                    groups.append((directory.name, paths))
            if direct and groups:
                raise ValueError("input mixes root images and site folders; use a manifest")
            if direct:
                groups = [(root.name, direct)]
        for site, paths in groups:
            frames = []
            for path in paths:
                if path.suffix.lower() not in SUFFIXES:
                    raise ValueError(f"unsupported image: {path.name}")
                for page in range(_page_count(path)):
                    relative = path.relative_to(root if root.is_dir() else root.parent).as_posix()
                    frames.append(Frame(site, path, relative, len(frames), page))
            sites[site] = frames
    if not sites:
        raise ValueError("no images found; use one folder per site, a single stack, or a manifest")
    return sites


def read_frame(frame: Frame, roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Return a finite, grayscale 2-D frame; preserve dtype and raw detector units."""
    suffix = frame.path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        with tifffile.TiffFile(frame.path) as image:
            page = image.pages[frame.page]
            if page.samplesperpixel != 1:
                raise ValueError(f"{frame.relative_path}: color TIFF is unsupported")
            array = page.asarray()
    elif suffix == ".npy":
        data = np.load(frame.path, mmap_mode="r", allow_pickle=False)
        try:
            if data.ndim == 3:
                array = np.array(data[frame.page])
            elif data.ndim == 2 and frame.page == 0:
                array = np.array(data)
            else:
                raise ValueError("NPY must be HxW or TxHxW, with a valid page")
        finally:
            data._mmap.close()
    else:
        if frame.page != 0:
            raise ValueError("page must be zero for single-frame images")
        with Image.open(frame.path) as image:
            if image.mode not in {"L", "I", "F", "I;16", "I;16B", "I;16L", "RGB"}:
                raise ValueError(f"{frame.relative_path}: grayscale images required; no color conversion is performed")
            array = np.asarray(image).copy()
            if image.mode == "RGB":
                if not (np.array_equal(array[..., 0], array[..., 1])
                        and np.array_equal(array[..., 0], array[..., 2])):
                    raise ValueError(
                        f"{frame.relative_path}: grayscale images required; "
                        "RGB channels must be identical at every pixel"
                    )
                # Repeated grayscale channels carry one signal. Select it exactly,
                # without luminance conversion or changing decoded detector units.
                array = array[..., 0]
    if array.ndim != 2 or array.dtype.kind not in "uif":
        raise ValueError(f"{frame.relative_path}: expected a real numeric HxW grayscale image")
    if not np.isfinite(array).all():
        raise ValueError(f"{frame.relative_path}: contains NaN or infinity")
    if roi is not None:
        y0, y1, x0, x1 = roi
        if y1 > array.shape[0] or x1 > array.shape[1]:
            raise ValueError(f"{frame.relative_path}: ROI extends outside the image")
        array = array[y0:y1, x0:x1]
    if min(array.shape) < 16:
        raise ValueError("analysis requires images/ROI at least 16 pixels in each dimension")
    # float32 work arrays preserve every uint8/uint16 value. Reject unsafe integer
    # ranges instead of silently destroying low-amplitude noise on large offsets.
    if array.dtype.kind in "ui" and (array.min() < -(2**24) or array.max() > 2**24):
        raise ValueError("integer intensities beyond float32 exact range are unsupported")
    return np.ascontiguousarray(array)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_hash(array: np.ndarray) -> str:
    """Canonical decoded-content identity, including shape, independent of encoding."""
    digest = hashlib.sha256(str(array.shape).encode("ascii"))
    digest.update(np.asarray(array, dtype="<f8").tobytes())
    return digest.hexdigest()
