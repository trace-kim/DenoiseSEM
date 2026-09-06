"""Prototype (NOT adopted): evidence-weighted residual restoration on top of the
fused K-frame output.

    python tools/burst_fusion_refine_prototype.py <fusion_ckpt> [--sources 48,50,...]

Idea: the fusion network is a conditional mean and shrinks weak structure;
the registered K-frame average is unbiased but noisy.  Split their difference
into difference-of-Gaussian bands (2-4, 4-8, 8-16 px), estimate the local
band power of that residual against the local Poisson noise power of a
K-frame mean, and add the residual back with the local Wiener gain
``max(0, 1 - N / P)``.  Measured on the ten dev scenes with the main fusion
checkpoint at K = 16: per-scene median blemish retention +0.02 .. +0.10
(e.g. scene 48: 0.69 -> 0.74 / 0.78 for 5 / 3 px power smoothing, scene 5:
0.56 -> 0.65), PSNR -0.1 .. -0.3 dB, and no change for the faint features
whose residual never rises above the local noise floor.  Not adopted: the
gain is small, the weak features are evidence-limited at 16 frames, and the
step adds noise where the network was right.  Kept as a record of the trial.
"""
from __future__ import annotations

import argparse

import numpy as np

from burst_diffusion.data import BurstCache
from edge_denoise.finefeat import feature_band, find_features, gaussian_blur, region_labels
from edge_denoise.fusion import FusionDenoiser
from edge_denoise.register import RegistrationTable, align_burst, fuse_mean

SIGMAS = (1.0, 2.0, 4.0, 8.0, 16.0)


def dog_bands(image: np.ndarray) -> list[np.ndarray]:
    blurs = [gaussian_blur(image, s) for s in SIGMAS]
    return [image - blurs[0]] + [blurs[i] - blurs[i + 1] for i in range(len(SIGMAS) - 1)]


def refine(fused: np.ndarray, registered_mean: np.ndarray, frames: int, gains: list[float], *, peak: float = 10.0, smooth: float = 5.0, bands=(1, 2, 3)) -> np.ndarray:
    residual_bands = dog_bands(registered_mean - fused)
    intensity = np.clip(gaussian_blur(fused, 4.0), 0.02, 1.0)
    out = fused.copy()
    for b in bands:
        noise = gains[b] * intensity / peak / frames
        power = gaussian_blur(residual_bands[b] ** 2, smooth)
        weight = np.clip(1.0 - noise / np.maximum(power, 1e-12), 0.0, 1.0)
        out = out + weight * residual_bands[b]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--dataset", default="data/MIIC-burst-p10-drift")
    parser.add_argument("--registration", default="runs/edge_denoise/registration_drift_pre.json")
    parser.add_argument("--sources", default="48,50,87,28,21,5,80,83,90")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--smooth", type=float, default=5.0)
    args = parser.parse_args()
    table = RegistrationTable.load(args.registration)
    cache = BurstCache(args.dataset, channels=1, min_replicas=args.frames, min_size=64, val_fraction=0.1, test_fraction=0.1, split_seed=2019)
    by = {s.source_index: s for s in cache.val_sources}
    denoiser = FusionDenoiser.from_checkpoint(args.checkpoint)
    white = np.random.default_rng(0).normal(0.0, 1.0, (512, 512))
    gains = [float(np.var(b)) for b in dog_bands(white)]
    for index in (int(v) for v in args.sources.split(",")):
        source = by[index]
        clean = source.clean.astype(np.float64) / 255.0
        regions = region_labels(clean)
        features, labels, _, _ = find_features(clean, regions, peak=10.0)
        if not features:
            continue
        frames01 = [f.astype(np.float64) / 255.0 for f in source.frames[: args.frames]]
        aligned, valid = align_burst(frames01, table.bursts[index][0], indices=range(args.frames), device=denoiser.device)
        mean = fuse_mean(aligned, valid, frames01[0])
        fused = denoiser.denoise_mean(mean, args.frames)
        refined = refine(fused, mean, args.frames, gains, smooth=args.smooth)

        def retention(image: np.ndarray) -> float:
            band = feature_band(image, regions)
            return float(np.median([band[labels == f.label].mean() / f.contrast for f in features]))

        def psnr(image: np.ndarray) -> float:
            return float(10.0 * np.log10(1.0 / np.mean((image - clean) ** 2)))

        print(f"scene {index}: retention fused {retention(fused):.3f} -> refined {retention(refined):.3f} | PSNR {psnr(fused):.2f} -> {psnr(refined):.2f} dB")


if __name__ == "__main__":
    main()
