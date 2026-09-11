"""Descriptive estimators with explicit spatial, temporal, and sampling domains."""

from __future__ import annotations

import numpy as np
from scipy import fft, ndimage, signal, stats

from .config import AnalysisConfig
from .registration import translate


def distribution(values: np.ndarray, limit: int, rng: np.random.Generator) -> dict:
    """Summarize a bounded sample, without IID significance tests on pooled pixels."""
    values = np.asarray(values).ravel()
    if len(values) > limit:
        values = values[rng.choice(len(values), limit, replace=False)]
    values = values.astype(float)
    mean = float(values.mean())
    sigma = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    median = float(np.median(values))
    mad_sigma = float(np.median(np.abs(values - median)) * 1.4826022185)
    hist, edges = np.histogram(values, bins=80, density=True)
    probabilities = np.linspace(0.005, 0.995, 199)
    return {
        "sample_count": len(values), "mean_dn": mean, "std_dn": sigma,
        "median_dn": median, "mad_sigma_dn": mad_sigma,
        "skewness": float(stats.skew(values, bias=False)) if sigma > 0 else None,
        "excess_kurtosis": float(stats.kurtosis(values, bias=False)) if sigma > 0 else None,
        "fraction_beyond_3_sigma": float(np.mean(np.abs(values - mean) > 3 * sigma)) if sigma > 0 else 0.0,
        "histogram_centers_dn": ((edges[:-1] + edges[1:]) / 2).tolist(),
        "histogram_density": hist.tolist(),
        "normal_quantiles": stats.norm.ppf(probabilities).tolist(),
        "observed_quantiles_dn": np.quantile(values, probabilities).tolist(),
    }


def mean_variance(samples: np.ndarray, bins: int) -> tuple[list[dict], dict]:
    """Estimate signal on even positions, variance on disjoint odd positions.

    The affine fit is in exported DN, not electron counts. A negative intercept
    is retained: detector offsets, processing, and model mismatch can cause it.
    """
    means = samples[::2].mean(axis=0, dtype=np.float64)
    variances = samples[1::2].var(axis=0, ddof=1, dtype=np.float64)
    edges = np.unique(np.quantile(means, np.linspace(0, 1, bins + 1)))
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (means >= lo) & ((means <= hi) if i == len(edges) - 2 else (means < hi))
        if mask.sum() < 8:
            continue
        rows.append({"bin": i, "signal_dn": float(means[mask].mean()),
                     "variance_dn2": float(variances[mask].mean()), "pixels": int(mask.sum()),
                     "lower_dn": float(lo), "upper_dn": float(hi)})
    if len(rows) < 3:
        return rows, {"available": False, "reason": "insufficient distinct intensity bins/pixels"}
    x = np.array([r["signal_dn"] for r in rows])
    y = np.array([r["variance_dn2"] for r in rows])
    weights = np.array([r["pixels"] for r in rows], dtype=float)
    center = np.average(x, weights=weights)
    scale = float(np.ptp(x))
    if scale <= max(1e-8, abs(center) * 1e-7):
        return rows, {"available": False, "reason": "signal range is too small"}
    signal_uncertainty = float(np.sqrt(np.median(variances) / len(samples[::2])))
    if scale < 6 * signal_uncertainty:
        return rows, {"available": False, "reason": "signal range is not resolved above repeat-mean noise"}
    design = np.column_stack(((x - center) / scale, np.ones_like(x)))
    coefficient = np.linalg.lstsq(design * np.sqrt(weights[:, None]), y * np.sqrt(weights), rcond=None)[0]
    slope = coefficient[0] / scale
    intercept = coefficient[1] - slope * center
    prediction = slope * x + intercept
    denominator = float(np.sum(weights * (y - np.average(y, weights=weights))**2))
    return rows, {"available": True, "slope_dn": float(slope), "intercept_dn2": float(intercept),
                  "r_squared": float(1 - np.sum(weights * (y - prediction)**2) / denominator) if denominator > 0 else None,
                  "signal_range_dn": [float(x.min()), float(x.max())],
                  "negative_predicted_variance": bool(np.any(prediction < 0)),
                  "interpretation": "phenomenological variance = slope * signal + intercept; no electron calibration"}


def _runs(indices: np.ndarray) -> list[np.ndarray]:
    return np.split(np.arange(len(indices)), np.flatnonzero(np.diff(indices) != 1) + 1)


def temporal_diagnostics(samples: np.ndarray, indices: np.ndarray, frame_means: np.ndarray,
                         config: AnalysisConfig, interval_s: float | None,
                         irregular_time: bool = False) -> dict:
    """Respect missing/excluded acquisition indices in every lag/block estimate."""
    values = np.asarray(samples, dtype=np.float64)
    residual = values - values.mean(axis=0)
    centered = residual - residual.mean(axis=1, keepdims=True)
    denom = float(np.mean(residual**2))
    centered_denom = float(np.mean(centered**2))
    lookup = {int(index): i for i, index in enumerate(indices)}
    acf = []
    for lag in range(1, min(config.max_lag, int(indices[-1] - indices[0])) + 1):
        pairs = [(i, lookup[int(index + lag)]) for i, index in enumerate(indices) if int(index + lag) in lookup]
        if not pairs:
            continue
        a, b = np.array(pairs).T
        acf.append({"lag_frames": lag, "lag_s": lag * interval_s if interval_s else None,
                    "pairs": len(pairs),
                    "pixel_acf": float(np.mean(residual[a] * residual[b]) / denom) if denom > 0 else None,
                    "pixel_acf_without_frame_offset": float(np.mean(centered[a] * centered[b]) / centered_denom) if centered_denom > 0 else None})
    runs = _runs(indices)
    longest = max(map(len, runs))
    shuffled = values[np.random.default_rng(config.seed).permutation(len(values))]
    curves = []
    m = 1
    while m <= longest // 4:
        squares, mean_squares, shuffled_squares, windows = 0.0, 0.0, 0.0, 0
        independent_pairs = 0
        for run in runs:
            if len(run) < 2 * m:
                continue
            # Prefix sums avoid repeatedly materializing m-frame image blocks.
            def differences(series: np.ndarray) -> np.ndarray:
                prefix = np.concatenate((np.zeros((1,) + series.shape[1:]), np.cumsum(series, axis=0)))
                averages = (prefix[m:] - prefix[:-m]) / m
                return averages[m:] - averages[:-m]

            difference = differences(values[run])
            mean_difference = differences(frame_means[run, None])
            control_difference = differences(shuffled[run])
            squares += float(np.sum(difference**2) / values.shape[1])
            mean_squares += float(np.sum(mean_difference**2))
            shuffled_squares += float(np.sum(control_difference**2) / values.shape[1])
            windows += len(difference)
            independent_pairs += len(run) // (2 * m)
        curves.append({"block_frames": m, "tau_s": m * interval_s if interval_s else None,
                       "overlapping_pairs": windows, "disjoint_pairs": independent_pairs,
                       "pixel_allan_deviation_dn": float(np.sqrt(squares / (2 * windows))),
                       "mean_allan_deviation_dn": float(np.sqrt(mean_squares / (2 * windows))),
                       "shuffled_pixel_allan_deviation_dn": float(np.sqrt(shuffled_squares / (2 * windows)))})
        m *= 2
    if curves:
        for row in curves:
            row["white_noise_reference_dn"] = curves[0]["pixel_allan_deviation_dn"] / np.sqrt(row["block_frames"])
    frequency, power = [], []
    if len(runs) == 1 and len(frame_means) >= 8 and not irregular_time:
        frequency_array, power_array = signal.periodogram(
            frame_means, fs=1 / interval_s if interval_s else 1.0,
            detrend="linear", window="hann", scaling="density",
        )
        frequency, power = frequency_array[1:].tolist(), power_array[1:].tolist()
    return {"acf": acf, "averaging": curves, "mean_periodogram_frequency": frequency,
            "mean_periodogram_density": power,
            "frequency_unit": "Hz" if interval_s else "cycles/frame",
            "periodogram_available": bool(frequency), "longest_contiguous_run": longest,
            "time_sampling": "irregular; frame-lag diagnostics only" if irregular_time else
                             ("uniform seconds" if interval_s else "acquisition index; timing unknown")}


def spatial_diagnostics(stack: np.ndarray, shifts: np.ndarray, positions: np.ndarray,
                        indices: np.ndarray, crop: tuple[slice, slice], integer: bool,
                        config: AnalysisConfig) -> tuple[dict, np.ndarray]:
    """Pair-difference spectra on an explicitly reported central crop, without masks."""
    possible = [i for i in range(0, len(indices) - 1, 2) if indices[i + 1] - indices[i] == 1]
    selected = np.unique(np.linspace(0, len(possible) - 1, min(len(possible), config.spatial_pairs)).astype(int)) if possible else []
    shape = stack[0][crop].shape
    side_y, side_x = min(shape[0], config.spatial_max_side), min(shape[1], config.spatial_max_side)
    y0, x0 = (shape[0] - side_y) // 2, (shape[1] - side_x) // 2
    window = np.outer(np.hanning(side_y), np.hanning(side_x))
    psd = np.zeros(window.shape, dtype=float)
    max_lag = min(config.max_lag, min(window.shape) // 4)
    covariance = np.zeros((2, max_lag))
    variance, row_energy, col_energy = 0.0, 0.0, 0.0
    for selected_i in selected:
        j = possible[selected_i]
        a, b = positions[j], positions[j + 1]
        difference = (translate(stack[b], shifts[b], integer=integer)[crop].astype(float) -
                      translate(stack[a], shifts[a], integer=integer)[crop]) / np.sqrt(2)
        patch = difference[y0:y0 + side_y, x0:x0 + side_x]
        patch -= patch.mean()
        variance += float(np.mean(patch**2))
        row_energy += float(np.var(patch.mean(axis=1))) * side_x
        col_energy += float(np.var(patch.mean(axis=0))) * side_y
        psd += np.abs(fft.fftshift(fft.fft2(patch * window)))**2 / np.sum(window**2)
        for lag in range(1, max_lag + 1):
            covariance[0, lag - 1] += np.mean(patch[lag:] * patch[:-lag])
            covariance[1, lag - 1] += np.mean(patch[:, lag:] * patch[:, :-lag])
    count = len(selected)
    rows = [{"lag_px": i + 1, "y_acf": float(covariance[0, i] / variance) if variance > 0 else None,
             "x_acf": float(covariance[1, i] / variance) if variance > 0 else None} for i in range(max_lag)] if count else []
    return {"pair_count": count, "crop_y0_y1_x0_x1": [int(crop[0].start + y0), int(crop[0].start + y0 + side_y),
                                                     int(crop[1].start + x0), int(crop[1].start + x0 + side_x)],
            "pair_sigma_dn": float(np.sqrt(variance / count)) if count else None,
            "row_banding_ratio": row_energy / variance if variance > 0 else None,
            "column_banding_ratio": col_energy / variance if variance > 0 else None,
            "acf": rows, "spectrum_unit": "DN^2 / (cycles/pixel)^2; two-sided Hann periodogram"}, psd / max(count, 1)


def analyze_mode(stack: np.ndarray, shifts: np.ndarray, accepted: np.ndarray,
                 indices: np.ndarray, crop: tuple[slice, slice], integer: bool,
                 levels: tuple[float | None, float | None], config: AnalysisConfig,
                 interval_s: float | None, irregular_time: bool) -> tuple[dict, dict[str, np.ndarray], list[dict]]:
    """Stream moments over a disk-backed stack and retain bounded pixel time series."""
    positions = np.flatnonzero(accepted)
    shape = stack[0][crop].shape
    mean, m2 = np.zeros(shape, dtype=float), np.zeros(shape, dtype=float)
    first_sum, last_sum = np.zeros(shape, dtype=float), np.zeros(shape, dtype=float)
    valid = np.ones(shape, dtype=bool)
    frame_rows = []
    quarter = max(1, len(positions) // 4)
    for j, i in enumerate(positions):
        frame = translate(stack[i], shifts[i], integer=integer)[crop].astype(float)
        delta = frame - mean
        mean += delta / (j + 1)
        m2 += delta * (frame - mean)
        if j < quarter:
            first_sum += frame
        if j >= len(positions) - quarter:
            last_sum += frame
        pixel_valid = np.ones(stack.shape[1:], dtype=np.float32)
        if levels[0] is not None:
            pixel_valid *= stack[i] > levels[0]
        if levels[1] is not None:
            pixel_valid *= stack[i] < levels[1]
        valid &= translate(pixel_valid, shifts[i], integer=integer)[crop] > 0.99999
        frame_rows.append({"frame_position": int(i), "mean_dn": float(frame.mean()),
                           "contrast_dn": float(frame.std()),
                           "laplacian_rms_dn": float(np.sqrt(np.mean(ndimage.laplace(frame)**2)))})
    variance = m2 / (len(positions) - 1)
    smooth = ndimage.gaussian_filter(mean, sigma=2)
    gy, gx = np.gradient(smooth)
    gradient = np.hypot(gy, gx)
    if not valid.any():
        raise ValueError("no pixels remain after clipping masks; check black/white levels and ROI")
    threshold = np.quantile(gradient[valid], config.flat_fraction)
    flat = ndimage.binary_erosion(valid & (gradient <= threshold), iterations=1)
    if flat.sum() < 32:
        raise ValueError("fewer than 32 unsaturated low-gradient pixels; expand ROI or flat_fraction")
    rng = np.random.default_rng(config.seed)
    candidates = np.flatnonzero(flat)
    chosen = rng.choice(candidates, min(config.sample_pixels, len(candidates)), replace=False)
    ys, xs = np.unravel_index(chosen, shape)
    samples = np.empty((len(positions), len(chosen)), dtype=stack.dtype)
    template = mean - mean.mean()
    template_energy = float(np.sum(template**2))
    for j, i in enumerate(positions):
        frame = translate(stack[i], shifts[i], integer=integer)[crop]
        samples[j] = frame[ys, xs]
        slope = float(np.sum((frame - frame.mean()) * template) / template_energy) if template_energy > 0 else None
        row = frame_rows[j]
        row.update(gain_to_mean=slope, offset_to_mean_dn=float(frame.mean() - slope * mean.mean()) if slope is not None else None,
                   residual_rms_dn=float(np.sqrt(np.mean((frame - mean)**2))),
                   flat_residual_rms_dn=float(np.sqrt(np.mean((frame[ys, xs] - mean[ys, xs])**2))))
    # The sqrt correction restores variance lost by subtracting a mean that
    # includes the observation under independent, stationary replicates.
    residual = (samples.astype(float) - samples.mean(axis=0, dtype=float)) * np.sqrt(len(positions) / (len(positions) - 1))
    adjacent = np.flatnonzero(np.diff(indices) == 1)
    pair_residual = (samples[adjacent + 1].astype(float) - samples[adjacent]) / np.sqrt(2)
    bins, fit = mean_variance(samples, config.intensity_bins)
    temporal = temporal_diagnostics(samples, indices, np.array([r["mean_dn"] for r in frame_rows]), config, interval_s, irregular_time)
    spatial, psd = spatial_diagnostics(stack, shifts, positions, indices, crop, integer, config)
    sigma_map = np.sqrt(variance)
    flat_sigma = float(np.sqrt(np.mean(variance[flat])))
    edge = valid & (gradient >= np.quantile(gradient[valid], 0.9))
    edge_sigma = float(np.sqrt(np.mean(variance[edge])))
    robust_center = float(np.median(sigma_map[flat]))
    robust_scale = float(np.median(np.abs(sigma_map[flat] - robust_center)) * 1.4826)
    unstable = valid & (sigma_map > max(robust_center * 5, robust_center + 8 * robust_scale))
    summary = {
        "pixel_domain": "integer translations; native pixel values" if integer else "bilinear subpixel translations",
        "frames": len(positions), "valid_pixels": int(valid.sum()), "flat_pixels": int(flat.sum()),
        "sampled_flat_pixels": len(chosen), "flat_gradient_threshold_dn_per_px": float(threshold),
        "temporal_sigma_dn": float(np.sqrt(np.mean(variance[valid]))), "flat_temporal_sigma_dn": flat_sigma,
        "edge_temporal_sigma_dn": edge_sigma, "edge_to_flat_sigma_ratio": edge_sigma / flat_sigma if flat_sigma > 0 else None,
        "unstable_pixel_candidates": int(unstable.sum()),
        "early_late_mean_delta_dn": float(np.mean((last_sum - first_sum) / quarter)),
        "early_late_rms_delta_dn": float(np.sqrt(np.mean(((last_sum - first_sum) / quarter)**2))),
        "early_late_frames_per_group": quarter,
        "residual_distribution": distribution(residual, config.distribution_samples, rng),
        "adjacent_difference_distribution": distribution(pair_residual, config.distribution_samples, rng) if len(adjacent) else None,
        "intensity_bins": bins, "mean_variance_fit": fit, "temporal": temporal, "spatial": spatial,
    }
    maps = {"mean": mean.astype(np.float32), "std": sigma_map.astype(np.float32),
            "early_late_delta": ((last_sum - first_sum) / quarter).astype(np.float32),
            "valid_mask": valid, "flat_mask": flat, "unstable_mask": unstable, "psd": psd.astype(np.float32)}
    return summary, maps, frame_rows
