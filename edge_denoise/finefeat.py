"""Fine-feature retention diagnostic: what does each estimator do to structure
that lives BELOW the edge scale?

Every learned arm of this project outputs images visibly smoother than the
clean reference (pilot report section 6).  "Smoother" hides two different
things, and this module measures both, per spatial scale, instead of assuming:

1. **Scale-resolved transfer in flat regions.**  The clean image and every
   output are split into difference-of-Gaussian bands (about 1-2 px, 2-4 px,
   4-8 px, 8-16 px, 16-32 px).  Inside the regions between strong edges
   (blemishes included -- only gradients above REGION_T count as structure)
   the output band is regressed on the clean
   band (blurs are confined to each connected region between strong edges --
   normalized convolution per region -- so neither an edge ramp nor the
   neighbouring plateau can ring into the statistic): the slope is the
   *empirical transfer gain* (1 = the structure is transmitted at full
   contrast, 0 = erased), the correlation says whether
   what is emitted is the true structure or something else, and the RMS ratio
   says how much band energy the output carries regardless of correctness --
   energy without correlation is hallucination.  Next to each arm the module
   prints the *linear (Wiener) bound* for that band, computed from the clean
   band power and the MEASURED single-frame noise band power: for one frame
   and for a 16-frame average.  A conditional-mean estimator with a good prior
   can beat the linear bound on sparse structure; nothing can transmit a band
   whose Wiener gain is ~0 as a per-pixel realization.

2. **Structured features.**  Connected components of the clean image's 2-12 px
   band that stand well above the grain (a robust-sigma threshold) inside the
   flat mask are the blemishes, scratches and stains that the user cares about.
   Each gets a single-frame matched-filter SNR (contrast x sqrt(area) / local
   Poisson sigma), and each arm's *retention* = its own band contrast over the
   same pixels divided by the clean contrast.  Retention is reported against
   SNR (recoverability), across retakes (is the feature shown consistently, or
   only in some acquisitions?), and complemented by a *false-feature* count:
   components in the output's band map that overlap no clean feature.

Denoisers are plain callables (frames ``[B, 1, S, S]`` in [-1, 1] -> images,
same shape, CPU), so classical, edge, burst and generative arms all go through
the same full-frame blended tiling (:func:`edge_denoise.distill.denoise_full_frame`).
numpy/torch/PIL only (repo convention: no scipy).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from burst_diffusion.data import BurstCache, BurstSource
from burst_diffusion.metrics import psnr

from .config import Config
from .distill import DenoiseFn, denoise_full_frame

#: Difference-of-Gaussian band edges (sigma in px); band k spans (edges[k], edges[k+1]].
BAND_SIGMAS: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)
BAND_LABELS: tuple[str, ...] = ("1-2px", "2-4px", "4-8px", "8-16px", "16-32px")
#: The "structured feature" band: 2-12 px structures (blemishes, scratches).
FEATURE_SIGMAS: tuple[float, float] = (1.0, 6.0)
FLAT_T = 0.006  # |grad| of the smoothed clean below this -> flat (pilot convention; grain stats)
EDGE_T = 0.030
#: |grad| of the smoothed clean above this (plus a margin) is *structure*; what
#: remains -- the regions between strong edges, blemishes included -- is where
#: bands and features are read.  Soft blemishes of contrast <= ~0.06 have
#: smoothed gradients below this; the line edges of the SEM patterns are 3-6x above.
REGION_T = 0.015
REGION_MARGIN = 3  # dilation of the strong-edge mask before it is excluded
MIN_REGION_AREA = 64  # smaller inter-edge regions carry no usable statistics
FEATURE_THRESHOLD_SIGMAS = 4.0  # feature = |band| > k * robust sigma of the band in flats
FEATURE_THRESHOLD_FLOOR = 0.004  # [0, 1] units; below the MIIC grain's 4-sigma (~0.012)
FLAT_MARGIN = 3  # erosion of the flat mask (px) before any statistic is read
MIN_FEATURE_AREA = 8
POISSON_PEAK_DEFAULT = 10.0


# ---------------------------------------------------------------------------
# image operators (torch, CPU, reflect padding, no scipy)


def _reflect_pad(tensor: torch.Tensor, radius: int) -> torch.Tensor:
    """Reflect-pad ``[1, 1, H, W]`` by ``radius`` on every side, in stages when
    the radius exceeds a side (torch requires pad < dim per application)."""
    remaining = radius
    while remaining > 0:
        step = min(remaining, tensor.shape[-1] - 1, tensor.shape[-2] - 1)
        tensor = F.pad(tensor, (step, step, step, step), mode="reflect")
        remaining -= step
    return tensor


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur of a 2D float array with reflect padding."""
    if sigma <= 0.0:
        return image.astype(np.float64, copy=True)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float64)
    kernel = torch.exp(-(offsets**2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float64))[None, None]
    padded = _reflect_pad(tensor, radius)
    padded = F.conv2d(padded, kernel.view(1, 1, 1, -1))
    padded = F.conv2d(padded, kernel.view(1, 1, -1, 1))
    return padded[0, 0].numpy()


def masked_blur(image: np.ndarray, labels: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur confined to each labelled region (normalized convolution
    per region: ``G(image * [labels == l]) / G([labels == l])``).  A region is
    a connected flat area between strong edges, so neither the edge ramp nor
    the neighbouring plateau at a different level can bleed into it -- a plain
    or single-mask blur would ring across the edge at every scale above the
    gap.  Pixels outside every region get the plain blur (never read)."""
    out = gaussian_blur(image, sigma)
    if sigma <= 0.0:
        return image.astype(np.float64, copy=True)
    for label in np.unique(labels[labels > 0]):
        member = labels == label
        weight = gaussian_blur(member.astype(np.float64), sigma)
        numerator = gaussian_blur(image * member, sigma)
        inside = member & (weight > 1e-6)
        out[inside] = numerator[inside] / weight[inside]
    return out


def band_maps(
    image: np.ndarray, labels: np.ndarray, sigmas: Sequence[float] = BAND_SIGMAS
) -> list[np.ndarray]:
    """Difference-of-region-masked-Gaussian bands ``G_{s_k} - G_{s_{k+1}}``."""
    blurred = [masked_blur(image, labels, s) for s in sigmas]
    return [blurred[k] - blurred[k + 1] for k in range(len(sigmas) - 1)]


def feature_band(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The structured-feature band (2-12 px), confined to each region."""
    return masked_blur(image, labels, FEATURE_SIGMAS[0]) - masked_blur(image, labels, FEATURE_SIGMAS[1])


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    k = 2 * radius + 1
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    return (-F.max_pool2d(-t, k, stride=1, padding=radius))[0, 0].numpy() > 0.5


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    k = 2 * radius + 1
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    return F.max_pool2d(t, k, stride=1, padding=radius)[0, 0].numpy() > 0.5


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected component labels (1..n) of a boolean mask; 0 = background.

    Label propagation by repeated 3x3 max-pooling of unique pixel ids until a
    fixed point: dependency-free and fast for 512^2 masks.
    """
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int64), 0
    height, width = mask.shape
    ids = (np.arange(height * width, dtype=np.float32).reshape(height, width) + 1.0) * mask
    t = torch.from_numpy(ids)[None, None]
    m = torch.from_numpy(mask.astype(np.float32))[None, None]
    while True:
        propagated = F.max_pool2d(t, 3, stride=1, padding=1) * m
        if torch.equal(propagated, t):
            break
        t = propagated
    raw = t[0, 0].numpy().astype(np.int64)
    unique = np.unique(raw[raw > 0])
    remap = np.zeros(int(raw.max()) + 1, dtype=np.int64)
    remap[unique] = np.arange(1, len(unique) + 1)
    return remap[raw], int(len(unique))


def robust_sigma(values: np.ndarray) -> float:
    """1.4826 x median absolute deviation (a grain-scale sigma immune to outliers)."""
    if values.size == 0:
        return float("nan")
    median = float(np.median(values))
    return 1.4826 * float(np.median(np.abs(values - median)))


# ---------------------------------------------------------------------------
# masks and features


def flat_and_edge_masks(clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(flat, edge) masks of the pilot's smoothness diagnostic: flat = the
    grain-only pixels (any gradient above FLAT_T excluded, blemishes included),
    edge = the contours."""
    smooth = gaussian_blur(clean, 1.5)
    gy, gx = np.gradient(smooth)
    magnitude = np.hypot(gx, gy)
    return magnitude < FLAT_T, magnitude > EDGE_T


def region_labels(clean: np.ndarray) -> np.ndarray:
    """Connected regions between strong edges (0 = structure / too small).

    Strong edges (smoothed gradient above REGION_T) are dilated by
    REGION_MARGIN and removed; what remains is labelled, and regions below
    MIN_REGION_AREA are dropped.  Blemishes and scratches, whose gradients are
    weaker, stay inside their region and can be measured."""
    smooth = gaussian_blur(clean, 1.5)
    gy, gx = np.gradient(smooth)
    structure = _dilate(np.hypot(gx, gy) > REGION_T, REGION_MARGIN)
    labels, count = label_components(~structure)
    if count == 0:
        return labels
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    small = areas < MIN_REGION_AREA
    small[0] = True
    labels[small[labels]] = 0
    return labels


@dataclass
class Feature:
    """One structured feature of the clean image (a connected blob in the
    2-12 px band that stands above the grain inside the flat mask)."""

    label: int
    area: int
    centroid_yx: tuple[float, float]
    contrast: float  # mean signed band value over the component, [0, 1] units
    local_intensity: float  # mean smoothed clean intensity there
    snr_single: float  # contrast * sqrt(area) / poisson sigma at that intensity


def find_features(
    clean: np.ndarray,
    labels: np.ndarray,
    *,
    peak: float,
    threshold_sigmas: float = FEATURE_THRESHOLD_SIGMAS,
    min_area: int = MIN_FEATURE_AREA,
) -> tuple[list[Feature], np.ndarray, np.ndarray, float]:
    """Detect structured features; returns (features, label map, feature band, threshold).

    The threshold is ``threshold_sigmas`` x the robust sigma of the band in the
    flats (the grain level), floored at :data:`FEATURE_THRESHOLD_FLOOR` so a
    perfectly flat synthetic field does not turn numerical ripple into
    features."""
    band = feature_band(clean, labels)
    interior = _erode(labels > 0, FLAT_MARGIN)
    sigma = robust_sigma(band[interior])
    threshold = max(threshold_sigmas * sigma, FEATURE_THRESHOLD_FLOOR)
    candidate = interior & (np.abs(band) > threshold)
    labels, count = label_components(candidate)
    smooth = gaussian_blur(clean, 3.0)
    features: list[Feature] = []
    kept = np.zeros_like(labels)
    for label in range(1, count + 1):
        member = labels == label
        area = int(member.sum())
        if area < min_area:
            continue
        ys, xs = np.nonzero(member)
        contrast = float(band[member].mean())
        intensity = float(np.clip(smooth[member].mean(), 1e-3, 1.0))
        sigma_n = math.sqrt(intensity / peak)
        features.append(
            Feature(
                label=len(features) + 1,
                area=area,
                centroid_yx=(float(ys.mean()), float(xs.mean())),
                contrast=contrast,
                local_intensity=intensity,
                snr_single=abs(contrast) * math.sqrt(area) / sigma_n,
            )
        )
        kept[member] = len(features)
    return features, kept, band, threshold


# ---------------------------------------------------------------------------
# per-source analysis


def _band_stats(
    out_bands: Sequence[np.ndarray], clean_bands: Sequence[np.ndarray], masks: Sequence[np.ndarray]
) -> list[dict]:
    rows = []
    for out_b, clean_b, mask in zip(out_bands, clean_bands, masks):
        c = clean_b[mask]
        o = out_b[mask]
        cc = float(np.dot(c, c))
        gain = float(np.dot(o, c) / cc) if cc > 0 else float("nan")
        corr = float(np.corrcoef(o, c)[0, 1]) if cc > 0 and o.std() > 0 else float("nan")
        rows.append(
            {
                "gain": gain,
                "corr": corr,
                "rms_ratio": float(np.sqrt(np.mean(o**2) / max(np.mean(c**2), 1e-30))),
                "rms_out": float(np.sqrt(np.mean(o**2))),
            }
        )
    return rows


#: A burst arm: ``(source, retake) -> full-frame [H, W] output in [0, 1]``.
BurstDenoiseFn = Callable[[BurstSource, int], np.ndarray]


def _retake_rows(
    frames: Sequence[np.ndarray], num_seeds: int, frames_per_retake: int
) -> tuple[list[np.ndarray], dict[str, list[np.ndarray]], int]:
    """Seed frames and classical rows.  With ``frames_per_retake > 1`` the
    frames are consecutive retakes (drifting bursts): seed ``r`` is the first
    frame of retake ``r`` and the averages are taken INSIDE each retake, one
    realization per retake -- what an instrument averaging a drifting burst
    delivers.  With 1 the historical layout applies (frame k is seed k; the
    averages are disjoint groups of consecutive frames)."""
    if frames_per_retake > 1:
        retakes = len(frames) // frames_per_retake
        seeds = min(num_seeds, retakes)
        seed_frames = [frames[r * frames_per_retake] for r in range(seeds)]
        rows: dict[str, list[np.ndarray]] = {}
        for count in (4, 16):
            if frames_per_retake >= count:
                rows[f"avg_of_{count}"] = [
                    np.mean(frames[r * frames_per_retake : r * frames_per_retake + count], axis=0)
                    for r in range(seeds)
                ]
        return seed_frames, rows, seeds
    seeds = min(num_seeds, len(frames))
    rows = {
        "avg_of_4": [np.mean(frames[4 * g : 4 * g + 4], axis=0) for g in range(min(seeds, len(frames) // 4))],
        "avg_of_16": [np.mean(frames[:16], axis=0)],
    }
    return list(frames[:seeds]), rows, seeds


def analyze_source(
    source: BurstSource,
    arms: Mapping[str, DenoiseFn],
    *,
    tile: int,
    stride: int,
    num_seeds: int,
    peak: float,
    tile_batch: int = 64,
    burst_arms: Mapping[str, BurstDenoiseFn] | None = None,
    frames_per_retake: int = 1,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Full-frame analysis of one source; returns (record, frame-0 outputs).

    ``burst_arms`` are scored per retake (see :func:`_retake_rows`); every
    other arm denoises the retake's first frame.
    """
    clean = source.clean.astype(np.float64) / 255.0
    all_frames = [f.astype(np.float64) / 255.0 for f in source.frames]
    frames, classical_rows, seeds = _retake_rows(all_frames, num_seeds, frames_per_retake)
    flat, edge = flat_and_edge_masks(clean)
    regions = region_labels(clean)
    clean_bands = band_maps(clean, regions)
    interior = _erode(regions > 0, FLAT_MARGIN)
    band_masks = [interior] * len(BAND_LABELS)
    features, feature_labels, _, threshold = find_features(clean, regions, peak=peak)
    feature_interior = interior

    # Measured noise band power (single frame) -> linear Wiener bounds per band.
    noise_bands = band_maps(frames[0] - clean, regions)
    wiener_1 = []
    wiener_16 = []
    clean_rms = []
    noise_rms = []
    for cb, nb, mask in zip(clean_bands, noise_bands, band_masks):
        s = float(np.mean(cb[mask] ** 2))
        n = float(np.mean(nb[mask] ** 2))
        clean_rms.append(math.sqrt(s))
        noise_rms.append(math.sqrt(n))
        wiener_1.append(s / (s + n) if s + n > 0 else float("nan"))
        wiener_16.append(s / (s + n / 16.0) if s + n > 0 else float("nan"))

    # Realizations: classical rows from the frames, arms by full-frame tiling,
    # burst arms per retake.
    outputs: dict[str, list[np.ndarray]] = {"single_frame": list(frames)}
    outputs.update(classical_rows)
    for name, fn in arms.items():
        outputs[name] = [
            denoise_full_frame(fn, frames[k], tile=tile, stride=stride, tile_batch=tile_batch)
            for k in range(seeds)
        ]
    for name, burst_fn in (burst_arms or {}).items():
        if name in outputs:
            raise ValueError(f"arm name {name!r} used twice")
        outputs[name] = [burst_fn(source, retake) for retake in range(seeds)]

    record: dict = {
        "source_index": source.source_index,
        "flat_fraction": float(flat.mean()),
        "edge_fraction": float(edge.mean()),
        "region_fraction": float((regions > 0).mean()),
        "bands": BAND_LABELS,
        "clean_band_rms": clean_rms,
        "noise_band_rms_single": noise_rms,
        "wiener_gain_single": wiener_1,
        "wiener_gain_avg16": wiener_16,
        "feature_threshold": threshold,
        "features": [
            {
                "label": f.label,
                "area": f.area,
                "centroid_yx": list(f.centroid_yx),
                "contrast": f.contrast,
                "local_intensity": f.local_intensity,
                "snr_single": f.snr_single,
            }
            for f in features
        ],
        "methods": {},
    }
    grain = clean - gaussian_blur(clean, 1.5)
    grain_mask = _erode(flat, 3)
    for name, realizations in outputs.items():
        out0 = realizations[0]
        out_bands = band_maps(out0, regions)
        residual = out0 - clean
        texture = out0 - gaussian_blur(out0, 1.5)
        # Feature retention: per feature, per realization.
        out_feature_bands = [feature_band(r, regions) for r in realizations]
        retention: list[dict] = []
        for f in features:
            member = feature_labels == f.label
            values = [float(b[member].mean()) / f.contrast for b in out_feature_bands]
            retention.append(
                {
                    "label": f.label,
                    "retention_mean": float(np.mean(values)),
                    "retention_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    "retention_per_seed": values,
                }
            )
        # False features: components in the output band above the CLEAN threshold
        # that overlap no clean feature (frame-0 realization).
        false_mask = feature_interior & (np.abs(out_feature_bands[0]) > threshold)
        false_labels, false_count = label_components(false_mask)
        false_features = 0
        false_area = 0
        for label in range(1, false_count + 1):
            member = false_labels == label
            if int(member.sum()) < MIN_FEATURE_AREA:
                continue
            if (feature_labels[member] > 0).any():
                continue
            false_features += 1
            false_area += int(member.sum())
        record["methods"][name] = {
            "realizations": len(realizations),
            "psnr_frame0": psnr(clean[..., None], out0[..., None]),
            "rms_flat": float(np.sqrt(np.mean(residual[flat] ** 2))),
            "rms_edge": float(np.sqrt(np.mean(residual[edge] ** 2))),
            "texture_rms_flat": float(np.sqrt(np.mean(texture[grain_mask] ** 2))),
            "corr_residual_grain": float(np.corrcoef(residual[grain_mask], grain[grain_mask])[0, 1]),
            "band": _band_stats(out_bands, clean_bands, band_masks),
            "feature_retention": retention,
            "false_features": false_features,
            "false_feature_area": false_area,
            "flat_pixels_searched": int(feature_interior.sum()),
        }
    frame0 = {name: realizations[0] for name, realizations in outputs.items()}
    return record, frame0


# ---------------------------------------------------------------------------
# aggregation and reports


def _nanmean(values: Sequence[float]) -> float | None:
    array = np.asarray([v for v in values if v is not None], dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else None


def summarize(records: Sequence[dict]) -> dict:
    """Average band statistics and pool features across sources."""
    methods = list(records[0]["methods"]) if records else []
    # Classical rows first, then the learned arms (JSON round-trips sort keys).
    classical = [m for m in ("single_frame", "avg_of_4", "avg_of_16") if m in methods]
    methods = classical + [m for m in methods if m not in classical]
    summary: dict = {
        "bands": list(BAND_LABELS),
        "clean_band_rms": [_nanmean([r["clean_band_rms"][k] for r in records]) for k in range(len(BAND_LABELS))],
        "noise_band_rms_single": [
            _nanmean([r["noise_band_rms_single"][k] for r in records]) for k in range(len(BAND_LABELS))
        ],
        "wiener_gain_single": [_nanmean([r["wiener_gain_single"][k] for r in records]) for k in range(len(BAND_LABELS))],
        "wiener_gain_avg16": [_nanmean([r["wiener_gain_avg16"][k] for r in records]) for k in range(len(BAND_LABELS))],
        "features_total": int(sum(len(r["features"]) for r in records)),
        "methods": {},
    }
    # Bins chosen for the MIIC dev population (median single-frame SNR ~0.45,
    # the largest stains ~2); a feature at SNR 1 is a 3-sigma detection in a
    # 16-frame average, at SNR 2 a 3-sigma detection in ~2 frames.
    snr_bins = [(0.0, 0.4), (0.4, 0.6), (0.6, 1.0), (1.0, float("inf"))]
    summary["snr_bins"] = [[lo, hi] for lo, hi in snr_bins]
    for name in methods:
        entries = [r["methods"][name] for r in records]
        band_gain = [_nanmean([e["band"][k]["gain"] for e in entries]) for k in range(len(BAND_LABELS))]
        band_corr = [_nanmean([e["band"][k]["corr"] for e in entries]) for k in range(len(BAND_LABELS))]
        band_rms_ratio = [_nanmean([e["band"][k]["rms_ratio"] for e in entries]) for k in range(len(BAND_LABELS))]
        pooled: list[tuple[float, float, float | None]] = []  # (snr, retention, std)
        for record, entry in zip(records, entries):
            for feature, ret in zip(record["features"], entry["feature_retention"]):
                pooled.append((feature["snr_single"], ret["retention_mean"], ret["retention_std"]))
        by_bin = []
        for lo, hi in snr_bins:
            members = [p for p in pooled if lo <= p[0] < hi]
            by_bin.append(
                {
                    "count": len(members),
                    "retention_median": float(np.median([p[1] for p in members])) if members else None,
                    "retention_mean": _nanmean([p[1] for p in members]),
                    "retention_std_mean": _nanmean([p[2] for p in members]),
                }
            )
        flat_searched = sum(e["flat_pixels_searched"] for e in entries)
        summary["methods"][name] = {
            "psnr_frame0": _nanmean([e["psnr_frame0"] for e in entries]),
            "rms_flat": _nanmean([e["rms_flat"] for e in entries]),
            "rms_edge": _nanmean([e["rms_edge"] for e in entries]),
            "texture_rms_flat": _nanmean([e["texture_rms_flat"] for e in entries]),
            "corr_residual_grain": _nanmean([e["corr_residual_grain"] for e in entries]),
            "band_gain": band_gain,
            "band_corr": band_corr,
            "band_rms_ratio": band_rms_ratio,
            "feature_retention_median": float(np.median([p[1] for p in pooled])) if pooled else None,
            "feature_retention_mean": _nanmean([p[1] for p in pooled]),
            "feature_retention_over_half": (
                float(np.mean([p[1] > 0.5 for p in pooled])) if pooled else None
            ),
            "feature_retention_std_mean": _nanmean([p[2] for p in pooled]),
            "retention_by_snr": by_bin,
            "false_features_per_1000_flat_px": (
                1000.0 * sum(e["false_features"] for e in entries) / flat_searched if flat_searched else None
            ),
            "false_features_total": int(sum(e["false_features"] for e in entries)),
        }
    return summary


def _fmt(value: float | None, spec: str) -> str:
    return "-" if value is None or (isinstance(value, float) and math.isnan(value)) else format(value, spec)


def write_summary_markdown(summary: dict, records: Sequence[dict], path: Path) -> None:
    lines = [
        "# Fine-feature retention summary",
        "",
        f"sources: {len(records)} | structured features found in clean flats: "
        f"{summary['features_total']} | bands: difference-of-Gaussian, flat regions only, "
        "eroded by 2x the band scale",
        "",
        "**Transfer gain per band** = regression slope of the output band on the clean band "
        "(1 = transmitted at full contrast, 0 = erased); **corr** says whether the emitted "
        "structure is the true one; **rms ratio** = output band energy / clean band energy "
        "(energy without correlation = hallucination). `wiener` rows are the linear "
        "(Wiener) bound from the measured clean and noise band powers, for one frame and "
        "for a 16-frame average.",
        "",
        "| method | " + " | ".join(f"gain {b}" for b in BAND_LABELS) + " | " + " | ".join(f"corr {b}" for b in BAND_LABELS) + " |",
        "|---|" + "---|" * (2 * len(BAND_LABELS)),
        "| clean band rms x1e-3 | " + " | ".join(_fmt(v * 1e3 if v is not None else None, ".2f") for v in summary["clean_band_rms"]) + " |" + " - |" * len(BAND_LABELS),
        "| noise band rms x1e-3 (1 frame) | " + " | ".join(_fmt(v * 1e3 if v is not None else None, ".2f") for v in summary["noise_band_rms_single"]) + " |" + " - |" * len(BAND_LABELS),
        "| wiener bound, 1 frame | " + " | ".join(_fmt(v, ".3f") for v in summary["wiener_gain_single"]) + " |" + " - |" * len(BAND_LABELS),
        "| wiener bound, 16 frames | " + " | ".join(_fmt(v, ".3f") for v in summary["wiener_gain_avg16"]) + " |" + " - |" * len(BAND_LABELS),
    ]
    for name, m in summary["methods"].items():
        lines.append(
            f"| {name} | "
            + " | ".join(_fmt(v, ".3f") for v in m["band_gain"])
            + " | "
            + " | ".join(_fmt(v, ".3f") for v in m["band_corr"])
            + " |"
        )
    lines += [
        "",
        "**Band energy ratio** (output rms / clean rms per band):",
        "",
        "| method | " + " | ".join(BAND_LABELS) + " |",
        "|---|" + "---|" * len(BAND_LABELS),
    ]
    for name, m in summary["methods"].items():
        lines.append(f"| {name} | " + " | ".join(_fmt(v, ".3f") for v in m["band_rms_ratio"]) + " |")
    bins = summary["snr_bins"]
    bin_labels = [f"SNR {lo:g}-{hi:g}" if math.isfinite(hi) else f"SNR >= {lo:g}" for lo, hi in bins]
    counts = [b["count"] for b in next(iter(summary["methods"].values()))["retention_by_snr"]] if summary["methods"] else []
    lines += [
        "",
        "**Structured-feature retention** (output contrast / clean contrast on the clean "
        "feature's pixels, 2-12 px band; `std` = spread of that retention across retakes; "
        "`false/1000px` = output-only features per 1000 searched flat pixels):",
        "",
        "| method | PSNR frame0 | texture rms flat x1e-3 | corr(resid, grain) | retention median | retention mean | > 0.5 | retake std | false/1000px | "
        + " | ".join(f"{label} (n={n})" for label, n in zip(bin_labels, counts))
        + " |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|" + "---:|" * len(bins),
    ]
    for name, m in summary["methods"].items():
        lines.append(
            f"| {name} | {_fmt(m['psnr_frame0'], '.2f')} | "
            f"{_fmt(m['texture_rms_flat'] * 1e3 if m['texture_rms_flat'] is not None else None, '.2f')} | "
            f"{_fmt(m['corr_residual_grain'], '+.2f')} | {_fmt(m['feature_retention_median'], '.3f')} | "
            f"{_fmt(m['feature_retention_mean'], '.3f')} | {_fmt(m['feature_retention_over_half'], '.0%')} | "
            f"{_fmt(m['feature_retention_std_mean'], '.3f')} | {_fmt(m['false_features_per_1000_flat_px'], '.2f')} | "
            + " | ".join(_fmt(b["retention_median"], ".3f") for b in m["retention_by_snr"])
            + " |"
        )
    lines.append("")
    lines.append(
        "Per-source records (per feature, per band, per realization) are in "
        "`fine_features.json`."
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_feature_plate(
    source_records: Sequence[tuple[dict, np.ndarray, dict[str, np.ndarray]]],
    path: Path,
    *,
    crop: int = 128,
    upscale: int = 3,
    gain: float = 8.0,
) -> None:
    """For each source: the clean crop around its strongest feature, then the
    2-12 px band map of clean and of every method (contrast x ``gain``), with
    detected clean features outlined."""
    font = ImageFont.load_default()
    rows: list[list[tuple[str, np.ndarray, np.ndarray | None]]] = []
    for record, clean, outputs in source_records:
        features = record["features"]
        if features:
            strongest = max(features, key=lambda f: f["snr_single"])
            cy, cx = strongest["centroid_yx"]
        else:
            cy, cx = clean.shape[0] / 2.0, clean.shape[1] / 2.0
        top = int(np.clip(round(cy - crop / 2), 0, clean.shape[0] - crop))
        left = int(np.clip(round(cx - crop / 2), 0, clean.shape[1] - crop))
        window = np.s_[top : top + crop, left : left + crop]

        regions = region_labels(clean)

        def band(image: np.ndarray) -> np.ndarray:
            b = feature_band(image, regions)
            return np.clip(b[window] * gain + 0.5, 0.0, 1.0)

        _, labels, _, _ = find_features(clean, regions, peak=POISSON_PEAK_DEFAULT)
        outline = (_dilate(labels > 0, 2) & ~(labels > 0))[window]
        row = [(f"src {record['source_index']}: clean", clean[window], None), ("clean band x8", band(clean), outline)]
        for name, image in outputs.items():
            row.append((f"{name} band x8", band(image), outline))
        rows.append(row)
    side = crop * upscale
    columns = max(len(r) for r in rows)
    pad, cap = 4, 14
    canvas = Image.new("RGB", (columns * (side + pad) + pad, len(rows) * (side + cap + pad) + pad), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for r, row in enumerate(rows):
        for c, (label, array, outline) in enumerate(row):
            rgb = np.stack([np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)] * 3, axis=-1)
            if outline is not None:
                rgb[outline] = (255, 80, 40)
            tile = Image.fromarray(rgb.repeat(upscale, 0).repeat(upscale, 1))
            x, y = pad + c * (side + pad), pad + r * (side + cap + pad)
            canvas.paste(tile, (x, y))
            draw.text((x, y + side + 1), label, fill=(230, 230, 230), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=True)


def fine_features(
    config: Config,
    arms: Mapping[str, DenoiseFn],
    *,
    out_dir: str | Path,
    split: str = "val",
    sources: Sequence[int] | None = None,
    num_seeds: int = 4,
    stride: int = 48,
    peak: float = POISSON_PEAK_DEFAULT,
    plate_sources: int = 3,
    extra_metadata: Mapping[str, object] | None = None,
    progress: Callable[[str], None] | None = None,
    burst_arms: Mapping[str, BurstDenoiseFn] | None = None,
    frames_per_retake: int = 1,
) -> dict:
    """Run the diagnostic; writes fine_features.json, summary.md and a plate."""
    if split not in ("val", "train", "test"):
        raise ValueError(f"split must be 'val', 'train', or 'test', got {split!r}")
    if frames_per_retake < 1:
        raise ValueError(f"frames_per_retake must be >= 1, got {frames_per_retake}")
    tile = config.data.image_size
    # min_replicas must not drop any source: the content-group split is
    # computed over the KEPT sources, so dropping the (16-frame) training
    # sources of a drifting dataset would silently re-split the holdout ones.
    # Sources with fewer retakes than requested simply yield fewer.
    cache = BurstCache(
        config.data.dataset_dir,
        channels=config.data.channels,
        min_replicas=2,
        min_size=tile,
        val_fraction=config.data.val_fraction,
        test_fraction=config.data.test_fraction,
        split_seed=config.data.split_seed,
    )
    available = cache.sources_for_split(split)
    if sources is not None:
        wanted = set(int(s) for s in sources)
        available = [s for s in available if s.source_index in wanted]
        missing = wanted - {s.source_index for s in available}
        if missing:
            raise ValueError(f"sources not in the {split!r} split: {sorted(missing)}")
    if not available:
        raise ValueError(f"no sources selected in the {split!r} split")
    records: list[dict] = []
    plates: list[tuple[dict, np.ndarray, dict[str, np.ndarray]]] = []
    for source in available:
        record, frame0 = analyze_source(
            source,
            arms,
            tile=tile,
            stride=stride,
            num_seeds=num_seeds,
            peak=peak,
            burst_arms=burst_arms,
            frames_per_retake=frames_per_retake,
        )
        records.append(record)
        if len(plates) < plate_sources:
            plates.append((record, source.clean.astype(np.float64) / 255.0, frame0))
        if progress is not None:
            progress(f"source {source.source_index}: {len(record['features'])} features")
    summary = summarize(records)
    results = {
        "dataset_dir": str(config.data.dataset_dir),
        "split": split,
        "num_seeds": num_seeds,
        "frames_per_retake": frames_per_retake,
        "stride": stride,
        "peak": peak,
        "summary": summary,
        "per_source": records,
    }
    if extra_metadata:
        results.update(dict(extra_metadata))
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "fine_features.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_summary_markdown(summary, records, destination / "summary.md")
    if plates:
        write_feature_plate(plates, destination / "feature_plate.png")
    return results
