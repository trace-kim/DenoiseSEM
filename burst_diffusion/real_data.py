"""Read the immutable, memory-mapped real-SEM cache prepared by edge_denoise.

This module handles storage only. Registration and objectives belong to the
edge package. The old synthetic manifest and its uint8 contract are unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .data import BurstCache

REAL_MANIFEST = "real_dataset.json"
REAL_FORMAT = 1


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"prepared cache path escapes dataset: {relative}")
    return path


def load_real_cache(cache: BurstCache, *, min_replicas: int, min_size: int) -> None:
    from .data import BurstSource

    if cache.channels != 1:
        raise ValueError("Real SEM data supports grayscale only")
    path = cache.burst_dir / REAL_MANIFEST
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("format") != REAL_FORMAT or metadata.get("kind") != "real_sem":
        raise ValueError(f"unsupported real SEM cache format: {path}")
    black, white = metadata["normalization"]["black"], metadata["normalization"]["white"]
    if not np.isfinite([black, white]).all() or white <= black:
        raise ValueError("real SEM normalization requires finite black < white")
    cache.train_sources, cache.val_sources, cache.test_sources = [], [], []
    cache.dropped_replicas, cache.dropped_size = [], []
    cache.duplicate_groups = metadata.get("duplicate_groups", [])
    seen: set[int] = set()
    for site in metadata["sites"]:
        index = site["source_index"]
        if index in seen or site["split"] not in ("train", "val", "test"):
            raise ValueError("duplicate source index or invalid split in real SEM cache")
        seen.add(index)
        # Hash before exposing any split, also detecting accidental edits to
        # prepared pixels or targets. No copies of the arrays enter Python RAM.
        arrays = {}
        for name in ("raw", "aligned", "sum"):
            record = site[name]
            stored = cache_path(cache.burst_dir, record["path"])
            if file_digest(stored) != record["sha256"]:
                raise ValueError(f"prepared cache content changed: {stored}; prepare a new dataset")
            arrays[name] = np.load(stored, mmap_mode="r", allow_pickle=False)
        raw = arrays["raw"]
        if raw.ndim != 3 or raw.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
            raise ValueError(f"invalid native frame stack for site {index}")
        if arrays["aligned"].shape != raw.shape or arrays["sum"].shape != raw.shape[1:]:
            raise ValueError(f"inconsistent prepared array shapes for site {index}")
        if arrays["aligned"].dtype != np.float32 or arrays["sum"].dtype != np.float32:
            raise ValueError("registered frames and sums must be float32")
        shifts = np.asarray(site["shifts"], dtype=np.float64)
        if shifts.shape != (len(raw), 2) or not np.isfinite(shifts).all():
            raise ValueError(f"invalid registration shifts for site {index}")
        bounds = np.asarray(site["bounds"])
        if (len(site["frames"]) != len(raw) or bounds.shape != (4,) or bounds.dtype.kind not in "iu"
                or (bounds[:2] < 0).any() or (bounds[2:] > raw.shape[1:]).any()
                or (bounds[2:] - bounds[:2] < min_size).any()):
            raise ValueError(f"invalid frame records or common overlap for site {index}")
        if len(raw) < min_replicas or min(raw.shape[1:]) < min_size:
            raise ValueError(f"site {index} cannot supply {min_replicas} frames of size {min_size}")
        source = BurstSource(source_index=index, clean=None, frames=raw)
        getattr(cache, f"{site['split']}_sources").append(source)
    if not cache.train_sources:
        raise ValueError("prepared real SEM cache has no training sites")
    cache.real_metadata = metadata
    cache.real_fingerprint = file_digest(path)


def normalize_native(array: np.ndarray, black: float, white: float) -> np.ndarray:
    return np.clip((array.astype(np.float32) - black) / (white - black), 0.0, 1.0)
