"""Prepare real SEM repeats and sample native-input, registered-target pairs.

Only translation is estimated. Charging, local distortion and changing
specimens require acquisition QC; they are not silently fitted away.
"""

from __future__ import annotations

import csv
import json
import math
import re
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from burst_diffusion.data import BurstCache, BurstSource, content_key
from burst_diffusion.real_data import (
    REAL_FORMAT, REAL_MANIFEST, cache_path, file_digest, normalize_native,
)

from .config import Config
from .data import PairBatch, PairFactory, PairInfo, ValPairBatch
from .image_io import collapse_grayscale_rgb
from .register import coarse_shift, gaussian_smooth, refine_shift, warp_frame

IMAGE_EXTENSIONS = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}
LOSS_MARGIN = 3


def read_native(path: Path) -> np.ndarray:
    """Read decoded grayscale 8/16-bit pixels, accepting identical RGB channels."""
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"export one frame per file (multi-page image): {path}")
        array = np.asarray(image).copy()
        array = collapse_grayscale_rgb(array, mode=image.mode, path=path)
        if image.mode == "I" and array.min() >= 0 and array.max() <= 65535:
            array = array.astype(np.uint16)
    if array.ndim != 2 or array.dtype.kind != "u" or array.dtype.itemsize not in (1, 2):
        raise ValueError(f"expected grayscale uint8/uint16 image: {path} ({array.shape}, {array.dtype})")
    return array.astype(np.uint8 if array.dtype.itemsize == 1 else np.uint16, copy=False)


def _natural_key(path: Path) -> list:
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name)]


def _assign_splits(
    sites: list[dict], val_fraction: float, test_fraction: float, seed: int,
    assignments: dict[str, str] | None,
) -> list[list[int]]:
    """Keep sites sharing ANY decoded frame together, before fixing holdouts."""
    parent = list(range(len(sites)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owners: dict[str, int] = {}
    for index, site in enumerate(sites):
        for frame in site["frames"]:
            digest = frame["sha256"]
            if digest in owners:
                parent[root(index)] = root(owners[digest])
            owners[digest] = index
    groups: dict[int, list[int]] = {}
    for index in range(len(sites)):
        groups.setdefault(root(index), []).append(index)
    members = list(groups.values())
    if assignments is not None:
        if set(assignments) != {site["name"] for site in sites}:
            raise ValueError("split file must assign every site folder exactly once")
        for site in sites:
            split = assignments[site["name"]]
            if split not in ("train", "val", "test"):
                raise ValueError(f"invalid split {split!r}")
            site["split"] = split
        if any(len({sites[i]["split"] for i in group}) != 1 for group in members):
            raise ValueError("duplicate image content crosses the requested site splits")
    else:
        required = 1 + int(val_fraction > 0) + int(test_fraction > 0)
        if len(members) < required:
            raise ValueError(f"need at least {required} distinct site groups for these holdouts")
        order = list(reversed(np.random.RandomState(seed).permutation(len(members)).tolist()))
        for site in sites:
            site["split"] = "train"
        cursor = 0
        for split, fraction in (("test", test_fraction), ("val", val_fraction)):
            target = max(1, round(fraction * len(sites))) if fraction else 0
            selected = 0
            # Reserve one group for training and, while assigning test, one
            # for validation. Duplicate groups are never broken to hit a count.
            reserve = 1 + int(split == "test" and val_fraction > 0)
            while selected < target and cursor < len(order) - reserve:
                group = members[order[cursor]]
                for index in group:
                    sites[index]["split"] = split
                selected += len(group)
                cursor += 1
    if not any(site["split"] == "train" for site in sites):
        raise ValueError("a training site is required")
    return [group for group in members if len(group) > 1]


def estimate_translations(
    raw: np.ndarray, *, black: float, white: float, sigma: float,
    radius: int, max_shift: float, device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Reuse existing shift estimators, streaming one full frame at a time."""
    resolved = torch.device(device)
    reference = gaussian_smooth(
        torch.from_numpy(normalize_native(raw[0], black, white))[None, None].to(resolved), sigma
    )
    shifts, uncertainty = np.zeros((len(raw), 2)), np.zeros((len(raw), 2))
    guess = (0, 0)
    with torch.no_grad():
        for index in range(1, len(raw)):
            moving = gaussian_smooth(
                torch.from_numpy(normalize_native(raw[index], black, white))[None, None].to(resolved), sigma
            )
            initial = coarse_shift(reference, moving, radius=radius, guess=guess)
            shift, covariance = refine_shift(reference, moving, initial)
            if not np.isfinite(shift).all() or np.abs(shift).max() > max_shift:
                raise ValueError(f"frame {index}: implausible shift {shift}; inspect acquisition or increase --max-shift")
            shifts[index] = shift
            uncertainty[index] = np.sqrt(np.maximum(np.diag(covariance), 0.0))
            guess = tuple(int(round(value)) for value in shift)
    return shifts, uncertainty


def _preview(raw: np.ndarray, aligned_mean: np.ndarray, black: float, white: float, path: Path) -> None:
    Image.fromarray(aligned_mean.astype(np.float32)).save(path.with_suffix(".tif"))
    images = [normalize_native(raw[0], black, white), normalize_native(raw[-1], black, white), aligned_mean]
    panel = Image.new("L", (3 * 384, 414), 255)
    draw = ImageDraw.Draw(panel)
    for index, (array, label) in enumerate(zip(images, ("first raw", "last raw", "registered mean (reference)"))):
        image = Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8))
        image.thumbnail((384, 384))  # Preview only; stored training pixels stay native.
        panel.paste(image, (index * 384, 30))
        draw.text((index * 384 + 4, 8), label, fill=0)
    panel.save(path)


def prepare_real_dataset(
    source_dir: str | Path, output_dir: str | Path, *, image_size: int = 512,
    black_level: float = 0.0, white_level: float | None = None,
    val_fraction: float = 0.1, test_fraction: float = 0.1, split_seed: int = 2019,
    split_file: str | Path | None = None, align: str = "translation",
    sigma: float = 2.0, radius: int = 6, max_shift: float = 32.0,
    frame_start: int = 0, frame_stop: int | None = None,
    device: str = "cpu", progress: Callable[[str], None] | None = None,
) -> Path:
    """Build a new immutable cache from ``source_dir/site_name/frame.tif``.

    Original inputs are preserved in native integer stacks. Registered copies
    and sums are float32, memory-mapped, and used only for targets/references.
    The output must not exist; a failed preparation never publishes a dataset.
    """
    source_root, destination = Path(source_dir).resolve(), Path(output_dir).resolve()
    if not source_root.is_dir():
        raise ValueError(f"missing site-folder directory: {source_root}")
    if destination.exists() or destination.is_relative_to(source_root):
        raise ValueError("output must be a new directory outside the raw site folders")
    if not (0 <= val_fraction < 1 and 0 <= test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("holdout fractions must be nonnegative and sum to less than one")
    if image_size < 8 or frame_start < 0 or (frame_stop is not None and frame_stop <= frame_start):
        raise ValueError("invalid patch size or frame interval")
    if align not in ("translation", "none") or sigma < 0 or radius < 1 or max_shift <= 0:
        raise ValueError("invalid registration settings")
    folders = sorted((p for p in source_root.iterdir() if p.is_dir()), key=_natural_key)
    if not folders:
        raise ValueError("expected one subfolder per SEM site")
    assignments = None if split_file is None else json.loads(Path(split_file).read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sites, qc_rows = [], []
    dtype = None
    white = white_level
    # The temporary directory is created directly under the resolved output
    # parent; cleanup is restricted to this newly created staging directory.
    with tempfile.TemporaryDirectory(prefix=".sem-prepare-", dir=destination.parent) as temporary, ExitStack() as mappings:
        staging = Path(temporary)
        (staging / "arrays").mkdir()
        (staging / "previews").mkdir()
        for source_index, folder in enumerate(folders):
            files = sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS), key=_natural_key)
            files = files[frame_start:frame_stop]
            if len(files) < 2:
                raise ValueError(f"site {folder.name} needs at least two image files in the selected interval")
            first = read_native(files[0])
            if dtype is None:
                dtype = first.dtype
                white = float(np.iinfo(dtype).max) if white is None else float(white)
            if not np.isfinite([black_level, white]).all() or white <= black_level:
                raise ValueError("normalization requires finite black_level < white_level")
            base = f"arrays/{source_index:05d}"
            raw = np.lib.format.open_memmap(staging / f"{base}_raw.npy", mode="w+", dtype=dtype, shape=(len(files), *first.shape))
            mappings.callback(raw._mmap.close)
            frames = []
            frame_hashes: set[str] = set()
            for index, file in enumerate(files):
                array = first if index == 0 else read_native(file)
                if array.shape != first.shape or array.dtype != dtype:
                    raise ValueError(f"all frames must have matching site dimensions and dataset bit depth: {file}")
                digest = content_key(array)
                if digest in frame_hashes:
                    raise ValueError(f"duplicate frame pixels within site {folder.name}: {file.name}")
                frame_hashes.add(digest)
                raw[index] = array
                frames.append({"name": file.relative_to(source_root).as_posix(), "sha256": digest})
            raw.flush()
            if progress:
                progress(f"{folder.name}: {len(files)} native {first.shape[0]}x{first.shape[1]} {dtype} frames; registering")
            shifts, uncertainty = (
                estimate_translations(raw, black=black_level, white=white, sigma=sigma, radius=radius, max_shift=max_shift, device=device)
                if align == "translation" else (np.zeros((len(raw), 2)), np.zeros((len(raw), 2))))
            lower = np.ceil(4 - shifts.min(axis=0)).astype(int)
            upper = np.floor(np.array(first.shape) - 4 - shifts.max(axis=0)).astype(int)
            if (upper - lower < image_size).any():
                raise ValueError(f"site {folder.name}: common registration overlap cannot fit {image_size}px crops")
            aligned = np.lib.format.open_memmap(staging / f"{base}_aligned.npy", mode="w+", dtype=np.float32, shape=raw.shape)
            mappings.callback(aligned._mmap.close)
            # One full-frame double accumulator avoids systematic rounding
            # across long bright bursts; the compact stored sum stays float32.
            total = np.zeros(first.shape, dtype=np.float64)
            for index, frame in enumerate(raw):
                unit = normalize_native(frame, black_level, white)
                moved = unit if not np.any(shifts[index]) else warp_frame(unit, shifts[index], device=device, clip=False)[0]
                aligned[index] = moved
                total += aligned[index]
                qc_rows.append({
                    "site": folder.name, "frame": index, "file": frames[index]["name"],
                    "dy": shifts[index, 0], "dx": shifts[index, 1],
                    "uncertainty_dy": uncertainty[index, 0], "uncertainty_dx": uncertainty[index, 1],
                    "mean01": float(unit.mean()), "std01": float(unit.std()),
                    "clipped_fraction": float(((frame < black_level) | (frame > white)).mean()),
                    "endpoint_fraction": float(((unit == 0) | (unit == 1)).mean()),
                })
            aligned.flush()
            np.save(staging / f"{base}_sum.npy", total.astype(np.float32), allow_pickle=False)
            _preview(raw, total / len(raw), black_level, white, staging / "previews" / f"{source_index:05d}.png")
            site = {"name": folder.name, "source_index": source_index, "frames": frames,
                    "shifts": shifts.tolist(), "bounds": [*lower.tolist(), *upper.tolist()]}
            # Release Windows mmap handles before the atomic directory rename.
            del frame
            aligned._mmap.close()
            raw._mmap.close()
            del aligned, raw
            for name in ("raw", "aligned", "sum"):
                relative = f"{base}_{name}.npy"
                site[name] = {"path": relative, "sha256": file_digest(staging / relative)}
            sites.append(site)
        duplicates = _assign_splits(sites, val_fraction, test_fraction, split_seed, assignments)
        metadata = {
            "format": REAL_FORMAT, "kind": "real_sem", "sites": sites,
            "normalization": {"black": black_level, "white": white},
            "registration": {"mode": align, "sigma": sigma, "radius": radius, "max_shift": max_shift,
                             "interpolation": "bicubic", "row_deformation": False},
            "split_seed": split_seed, "duplicate_groups": duplicates,
            "val_fraction": val_fraction, "test_fraction": test_fraction,
        }
        with (staging / "qc.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(qc_rows[0]))
            writer.writeheader()
            writer.writerows(qc_rows)
        (staging / REAL_MANIFEST).write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staging.rename(destination)
    return destination / REAL_MANIFEST


def sample_region(
    array: np.ndarray, top: float, left: float, size: int,
    *, normalization: tuple[float, float] | None = None,
) -> np.ndarray:
    """Read a float32 region with a subpixel origin, using a small halo only."""
    r, c = math.floor(top), math.floor(left)
    if top == r and left == c and r >= 0 and c >= 0 and r + size <= array.shape[0] and c + size <= array.shape[1]:
        crop = np.asarray(array[r:r + size, c:c + size], dtype=np.float32)
        return crop if normalization is None else normalize_native(crop, *normalization)
    pad = 2
    r0, c0 = r - pad, c - pad
    extent = size + 2 * pad + 1
    r1, c1 = r0 + extent, c0 + extent
    patch = np.asarray(array[max(0, r0):min(array.shape[0], r1), max(0, c0):min(array.shape[1], c1)], dtype=np.float32)
    padding = ((max(0, -r0), max(0, r1 - array.shape[0])), (max(0, -c0), max(0, c1 - array.shape[1])))
    if any(value for pair in padding for value in pair):
        patch = np.pad(patch, padding, mode="edge")
    if normalization is not None:
        patch = normalize_native(patch, *normalization)
    base = torch.arange(size, dtype=torch.float32)
    ys = (base + pad + top - r) / (extent - 1) * 2 - 1
    xs = (base + pad + left - c) / (extent - 1) * 2 - 1
    grid = torch.stack([xs[None].expand(size, -1), ys[:, None].expand(-1, size)], dim=-1)[None]
    return F.grid_sample(torch.from_numpy(patch.copy())[None, None], grid, mode="bicubic", align_corners=True)[0, 0].numpy()


def fixed_windows(bounds: list[int], size: int) -> list[tuple[int, int]]:
    y0, x0, y1, x1 = bounds
    if min(y1 - y0, x1 - x0) < size:
        raise ValueError(f"common overlap {bounds} cannot fit a {size}px crop")
    return list(dict.fromkeys([
        ((y0 + y1 - size) // 2, (x0 + x1 - size) // 2),
        (y0, x0), (y0, x1 - size), (y1 - size, x0), (y1 - size, x1 - size),
    ]))


class RealPairFactory(PairFactory):
    """Use native raw inputs, registered targets, and aligned consistency."""

    def __init__(self, cache: BurstCache, config: Config, *, seed: int):
        assert cache.real_metadata is not None
        objective = config.objective
        if (objective.representation != "image" or objective.target not in ("noisy", "noisy_mean")
                or objective.gradient_target != "target" or objective.fusion is not None
                or objective.target_debias_peak is not None or config.training.defect_augment is not None):
            raise ValueError("Real SEM supports image representation, noisy/noisy_mean targets, gradient_target=target, and optional consistency; synthetic oracle/debias/augmentation/fusion settings are unsupported")
        self.cache, self.image_size, self.batch_size = cache, config.data.image_size, config.training.batch_size
        self.target, self.need_second = objective.target, objective.lambda_consistency > 0
        self._rng = np.random.default_rng(seed)
        self.sites = {site["source_index"]: site for site in cache.real_metadata["sites"]}
        self.black = cache.real_metadata["normalization"]["black"]
        self.white = cache.real_metadata["normalization"]["white"]
        self.aligned, self.totals = {}, {}
        for source in cache.train_sources + cache.val_sources:
            if len(source.frames) < config.min_replicas:
                raise ValueError(f"site {source.source_index} needs at least {config.min_replicas} distinct frames")
            site = self.sites[source.source_index]
            fixed_windows(site["bounds"], self.image_size)
            if self.target == "noisy_mean":
                self.aligned[source.source_index] = np.load(cache_path(cache.burst_dir, site["aligned"]["path"]), mmap_mode="r")
                self.totals[source.source_index] = np.load(cache_path(cache.burst_dir, site["sum"]["path"]), mmap_mode="r")

    def _pair(self, source: BurstSource, window: tuple[int, int], a: int, b: int, second: int | None) -> tuple:
        size, site = self.image_size, self.sites[source.source_index]
        shifts = np.asarray(site["shifts"])
        origin = np.rint(np.asarray(window) + shifts[a]).astype(int)
        raw = source.frames[a][origin[0]:origin[0] + size, origin[1]:origin[1] + size]
        inputs = normalize_native(raw, self.black, self.white)
        reference_origin = origin - shifts[a]
        if self.target == "noisy":
            target_origin = origin + shifts[b] - shifts[a]
            targets = sample_region(source.frames[b], *target_origin, size, normalization=(self.black, self.white))
        else:
            total = sample_region(self.totals[source.source_index], *reference_origin, size)
            own = sample_region(self.aligned[source.source_index][a], *reference_origin, size)
            targets = (total - own) / (len(source.frames) - 1)
        second_input, second_shift = None, np.zeros(2, dtype=np.float32)
        if second is not None:
            second_origin = np.rint(np.asarray(window) + shifts[second]).astype(int)
            second_input = normalize_native(source.frames[second][second_origin[0]:second_origin[0] + size,
                                                                      second_origin[1]:second_origin[1] + size], self.black, self.white)
            second_shift = shifts[a] - shifts[second] + second_origin - origin
        info = PairInfo(source.source_index, tuple(origin.tolist()), a, b if self.target == "noisy" else None, second)
        convert = lambda image: (np.asarray(image, dtype=np.float32) * 2 - 1)[None]
        return convert(inputs), convert(targets), None if second_input is None else convert(second_input), second_shift, info

    def _batch(self, samples: list[tuple], *, validation: bool = False):
        inputs, targets, seconds, shifts, infos = zip(*samples)
        values = dict(inputs=torch.from_numpy(np.stack(inputs)), targets=torch.from_numpy(np.stack(targets)),
                      second=None if seconds[0] is None else torch.from_numpy(np.stack(seconds)),
                      second_shifts=torch.tensor(np.asarray(shifts), dtype=torch.float32), loss_margin=LOSS_MARGIN)
        batch = ValPairBatch(**values, clean=None) if validation else PairBatch(**values)
        return batch, list(infos)

    def sample_batch(self, *, count: int | None = None, return_info: bool = False) -> PairBatch | tuple[PairBatch, list[PairInfo]]:
        count = self.batch_size if count is None else count
        if count < 1:
            raise ValueError("count must be positive")
        samples = []
        for _ in range(count):
            source = self.cache.train_sources[int(self._rng.integers(len(self.cache.train_sources)))]
            y0, x0, y1, x1 = self.sites[source.source_index]["bounds"]
            window = (int(self._rng.integers(y0, y1 - self.image_size + 1)), int(self._rng.integers(x0, x1 - self.image_size + 1)))
            order = self._rng.permutation(len(source.frames)).tolist()
            second = order[2 if self.target == "noisy" else 1] if self.need_second else None
            samples.append(self._pair(source, window, order[0], order[1], second))
        batch, info = self._batch(samples)
        return (batch, info) if return_info else batch

    def val_batch(self, *, count: int) -> ValPairBatch:
        if not self.cache.val_sources or count < 1:
            raise ValueError("positive count and validation sites are required")
        samples = []
        for index in range(count):
            source = self.cache.val_sources[index % len(self.cache.val_sources)]
            windows = fixed_windows(self.sites[source.source_index]["bounds"], self.image_size)
            window = windows[(index // len(self.cache.val_sources)) % len(windows)]
            samples.append(self._pair(source, window, 0, min(2, len(source.frames) - 1), 1))
        return self._batch(samples, validation=True)[0]
